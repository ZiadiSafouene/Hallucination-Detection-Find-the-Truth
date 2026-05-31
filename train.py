from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from config import (
    MODEL_REGISTRY,
    MODEL_SAVE_DIR,
    SYSTEM_PROMPT,
    TASK_REGISTRY,
    resolve_tasks,
    task_output_dir,
)
from predict import evaluate
from utils import example_to_messages, load_json_dataset


def load_model_and_tokenizer(model_key: str):
    cfg = MODEL_REGISTRY[model_key]
    model_ids = [cfg["hf_id"]]
    if fallback := cfg.get("fallback_hf_id"):
        model_ids.append(fallback)

    last_err = None
    for model_id in model_ids:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )
            if model_id != cfg["hf_id"]:
                print(f"  Using fallback weights: {model_id}")
            break
        except OSError as e:
            last_err = e
    else:
        raise last_err

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer, cfg


def build_sft_text(tokenizer, messages: list[dict]) -> dict:
    prompt_messages = messages[:-1]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    full = tokenizer(full_text, truncation=False, add_special_tokens=False)
    prompt_ids = tokenizer(prompt_text, truncation=False, add_special_tokens=False)["input_ids"]
    input_ids = full["input_ids"]

    if len(prompt_ids) < len(input_ids) and input_ids[: len(prompt_ids)] == prompt_ids:
        # NOTE: comparing two plain Python lists with == gives True/False correctly
        # only when both are lists of ints (not tensors). input_ids here comes from
        # tokenizer(...)["input_ids"] which is a plain Python list, so this is safe.
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    else:
        # Fallback: train on the full sequence (no masking)
        labels = list(input_ids)

    return {"input_ids": input_ids, "labels": labels}


def build_dataset(examples: list[dict], tokenizer, max_length: int) -> Dataset:
    rows = []
    for ex in examples:
        messages = example_to_messages(ex, SYSTEM_PROMPT)
        item = build_sft_text(tokenizer, messages)
        if len(item["input_ids"]) > max_length:
            item["input_ids"] = item["input_ids"][:max_length]
            item["labels"] = item["labels"][:max_length]
        rows.append(item)
    return Dataset.from_list(rows)


class SFTCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict]) -> dict:
        max_len = min(self.max_length, max(len(f["input_ids"]) for f in features))
        input_ids, labels, attention_mask = [], [], []
        for f in features:
            ids = f["input_ids"][:max_len]
            lbs = f["labels"][:max_len]
            pad = max_len - len(ids)
            input_ids.append(ids + [self.tokenizer.pad_token_id] * pad)
            labels.append(lbs + [-100] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def train_one(
    task: str,
    model_key: str,
    train_examples: list[dict],
    dev_examples: list[dict],
    output_dir: str,
    *,
    num_epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 16,
    learning_rate: float = 1e-5,
    max_length: int = 2048,
    max_new_tokens: int = 128,
    skip_eval: bool = False,
) -> dict:
    task_dir = task_output_dir(output_dir, task)
    save_path = os.path.join(task_dir, model_key, MODEL_SAVE_DIR)
    os.makedirs(save_path, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Task {task} — {TASK_REGISTRY[task]['name']}")
    print(f"  {MODEL_REGISTRY[model_key]['display']}  ({model_key})")
    print(f"  → {save_path}")
    print(f"{'=' * 60}")

    model, tokenizer, cfg = load_model_and_tokenizer(model_key)
    max_length = min(max_length, cfg.get("max_length", 2048))

    train_ds = build_dataset(train_examples, tokenizer, max_length)
    collator = SFTCollator(tokenizer, max_length)

    training_args = TrainingArguments(
        output_dir=os.path.join(task_dir, model_key, "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        optim="adamw_torch",
        report_to="none",
        seed=42,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    trainer.train()

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  Saved → {save_path}")

    if skip_eval:
        return {}

    print("  Evaluating on dev …")
    metrics = evaluate(model, tokenizer, dev_examples, max_new_tokens=max_new_tokens)
    print(f"  Combined score  : {metrics['combined_score']:.4f}")
    print(f"  Macro F1        : {metrics['macro_f1']:.4f}")
    print(f"  Option accuracy : {metrics['option_accuracy']:.4f}")

    with open(os.path.join(task_dir, model_key, "dev_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Full fine-tune Arabic LLMs (Subtask 3)")
    parser.add_argument(
        "--task",
        nargs="+",
        choices=["1", "2", "all"],
        default=["all"],
        help="Task 1 (Islamic) or Task 2 (General Culture), or both",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default=None,
        help="Override train JSON (only when running a single --task)",
    )
    parser.add_argument(
        "--dev_path",
        type=str,
        default=None,
        help="Override dev JSON (only when running a single --task)",
    )
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        default=["all"],
    )
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    tasks = resolve_tasks(args.task)
    if (args.train_path or args.dev_path) and len(tasks) > 1:
        parser.error("--train_path / --dev_path only work with a single --task")

    if not torch.cuda.is_available():
        print("WARNING: CUDA not found. Full fine-tuning 7–9B models needs a GPU.\n")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    model_keys = list(MODEL_REGISTRY.keys()) if "all" in args.models else args.models
    all_results: dict[str, dict] = {}

    for task in tasks:
        task_cfg = TASK_REGISTRY[task]
        train_path = args.train_path or task_cfg["train_path"]
        dev_path = args.dev_path or task_cfg["dev_path"]

        train_examples = load_json_dataset(train_path)
        dev_examples = load_json_dataset(dev_path)

        print(f"Task {task} — {task_cfg['name']}")
        print(f"  Train: {len(train_examples)}  ({train_path})")
        print(f"  Dev  : {len(dev_examples)}  ({dev_path})")

        task_results = {}
        for key in model_keys:
            task_results[key] = train_one(
                task,
                key,
                train_examples,
                dev_examples,
                args.output_dir,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                learning_rate=args.learning_rate,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                skip_eval=args.skip_eval,
            )
        all_results[task] = task_results

    if not args.skip_eval:
        print(f"\n{'=' * 62}")
        print("  DEV SUMMARY")
        print(f"{'=' * 62}")
        for task, task_results in all_results.items():
            print(f"\n  Task {task} — {TASK_REGISTRY[task]['name']}")
            print(f"  {'Model':<16} {'Combined':>10} {'Macro F1':>10} {'Opt Acc':>10}")
            print(f"  {'-' * 50}")
            for key, m in task_results.items():
                if m:
                    print(
                        f"  {key:<16}"
                        f"{m['combined_score']:>10.4f}"
                        f"{m['macro_f1']:>10.4f}"
                        f"{m['option_accuracy']:>10.4f}"
                    )

        summary_path = os.path.join(args.output_dir, "all_dev_results.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults → {summary_path}")


if __name__ == "__main__":
    main()
