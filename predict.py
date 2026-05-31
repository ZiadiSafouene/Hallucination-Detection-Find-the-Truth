from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    MODEL_REGISTRY,
    MODEL_SAVE_DIR,
    SYSTEM_PROMPT,
    TASK_REGISTRY,
    resolve_tasks,
    task_output_dir,
)
from utils import (
    compute_metrics,
    example_to_messages,
    load_json_dataset,
    parse_model_output,
    predictions_to_submission,
)


def load_finetuned_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


@torch.no_grad()
def predict_examples(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    *,
    max_new_tokens: int = 128,
    verbose: bool = True,
) -> list[dict[str, str]]:
    device = next(model.parameters()).device
    preds: list[dict[str, str]] = []

    for i, ex in enumerate(examples):
        messages = example_to_messages(ex, SYSTEM_PROMPT)
        if messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        try:
            preds.append(parse_model_output(text))
        except ValueError:
            preds.append({"predicted_label": "hallucination", "predicted_answer": "A"})
            if verbose:
                print(f"  [warn] parse failed for {ex['id']}: {text[:80]!r}")

        if verbose and (i + 1) % 50 == 0:
            print(f"  … {i + 1}/{len(examples)}")

    return preds


def evaluate(
    model,
    tokenizer,
    dev_examples: list[dict[str, Any]],
    *,
    max_new_tokens: int = 128,
) -> dict[str, float]:
    """Used by train.py for quick post-training dev scoring."""
    was_training = model.training
    model.eval()

    was_use_cache = getattr(model.config, "use_cache", None)
    if was_use_cache is not None:
        model.config.use_cache = True

    preds = predict_examples(
        model, tokenizer, dev_examples,
        max_new_tokens=max_new_tokens, verbose=False,
    )

    if was_use_cache is not None:
        model.config.use_cache = was_use_cache

    if was_training:
        model.train()
    return compute_metrics(dev_examples, preds)


def run_model(
    task: str,
    model_key: str,
    examples: list[dict[str, Any]],
    output_dir: str,
    *,
    model_root: str = "./outputs",
    has_labels: bool = True,
    max_new_tokens: int = 128,
    codabench_format: bool = False,
) -> dict[str, float] | None:
    model_dir = os.path.join(task_output_dir(model_root, task), model_key, MODEL_SAVE_DIR)
    if not os.path.isdir(model_dir):
        print(f"\n  [SKIP] {model_key} — not found: {model_dir}")
        return None

    print(f"\n  Task {task} — {MODEL_REGISTRY[model_key]['display']}")
    t0 = time.time()
    model, tokenizer = load_finetuned_model(model_dir)
    preds = predict_examples(model, tokenizer, examples, max_new_tokens=max_new_tokens)
    elapsed = time.time() - t0

    submission = predictions_to_submission(
        examples, preds, codabench_format=codabench_format
    )
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"predictions_{model_key}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
    print(f"  Predictions → {out_path}  ({elapsed:.1f}s)")

    if not has_labels:
        return None

    metrics = compute_metrics(examples, preds)
    print(f"  Combined score  : {metrics['combined_score']:.4f}")
    print(f"  Macro F1        : {metrics['macro_f1']:.4f}")
    print(f"  Option accuracy : {metrics['option_accuracy']:.4f}")

    with open(os.path.join(output_dir, f"results_{model_key}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Predict on dev with fine-tuned models")
    parser.add_argument(
        "--task",
        nargs="+",
        choices=["1", "2", "all"],
        default=["all"],
        help="Task 1 (Islamic) or Task 2 (General Culture)",
    )
    parser.add_argument(
        "--dev_path",
        default=None,
        help="Override dev JSON (only with a single --task)",
    )
    parser.add_argument("--output_dir", default="./predictions")
    parser.add_argument("--model_root", default="./outputs")
    parser.add_argument(
        "--model",
        nargs="+",
        dest="models",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        default=["all"],
    )
    parser.add_argument("--no_metrics", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--codabench_format", action="store_true")
    args = parser.parse_args()

    tasks = resolve_tasks(args.task)
    if args.dev_path and len(tasks) > 1:
        parser.error("--dev_path only works with a single --task")

    model_keys = list(MODEL_REGISTRY.keys()) if "all" in args.models else args.models
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    for task in tasks:
        dev_path = args.dev_path or TASK_REGISTRY[task]["dev_path"]
        examples = load_json_dataset(dev_path)
        has_labels = not args.no_metrics and examples and "label" in examples[0]
        pred_dir = os.path.join(args.output_dir, f"task{task}")

        print(f"\nTask {task} — {TASK_REGISTRY[task]['name']}")
        print(f"Dev file: {dev_path} ({len(examples)} examples)")

        task_metrics = {}
        for key in model_keys:
            metrics = run_model(
                task,
                key,
                examples,
                pred_dir,
                model_root=args.model_root,
                has_labels=has_labels,
                max_new_tokens=args.max_new_tokens,
                codabench_format=args.codabench_format,
            )
            if metrics:
                task_metrics[key] = metrics

        if task_metrics:
            print(f"\n  Task {task} summary")
            for key, m in task_metrics.items():
                print(
                    f"  {key:<16} combined={m['combined_score']:.4f}  "
                    f"f1={m['macro_f1']:.4f}  acc={m['option_accuracy']:.4f}"
                )


if __name__ == "__main__":
    main()
