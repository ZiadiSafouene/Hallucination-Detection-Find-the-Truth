"""
Shared helpers for data loading, prompts, parsing, and metrics.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score

from config import (
    GOLD_TO_SUBMISSION_LABEL,
    SUBMISSION_TO_GOLD_LABEL,
    VALID_OPTIONS,
    EVAL_ALPHA,
    EVAL_BETA,
)


def load_json_dataset(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def gold_label_to_submission(label: str) -> str:
    if label not in GOLD_TO_SUBMISSION_LABEL:
        raise ValueError(f"Unknown label: {label!r}")
    return GOLD_TO_SUBMISSION_LABEL[label]


def submission_label_to_gold(label: str) -> str:
    key = label.strip().lower().replace(" ", "-")
    if key in SUBMISSION_TO_GOLD_LABEL:
        return SUBMISSION_TO_GOLD_LABEL[key]
    if label in SUBMISSION_TO_GOLD_LABEL:
        return SUBMISSION_TO_GOLD_LABEL[label]
    raise ValueError(f"Unknown submission label: {label!r}")


def format_options(options: dict[str, str]) -> str:
    lines = []
    for letter in VALID_OPTIONS:
        if letter in options:
            lines.append(f"{letter}. {options[letter]}")
    return "\n".join(lines)


def build_user_prompt(example: dict[str, Any]) -> str:
    options_text = format_options(example["options"])
    return (
        f"السؤال:\n{example['question']}\n\n"
        f"الإجابة المولّدة:\n{example['generated_answer']}\n\n"
        f"الخيارات:\n{options_text}\n\n"
        "حدّد تصنيف الهلوسة والخيار الصحيح."
    )


def build_target_response(example: dict[str, Any]) -> str:
    payload = {
        "predicted_label": gold_label_to_submission(example["label"]),
        "predicted_answer": example["answer"].strip().upper(),
    }
    return json.dumps(payload, ensure_ascii=False)


def example_to_messages(example: dict[str, Any], system_prompt: str) -> list[dict[str, str]]:
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(example)},
    ]
    if "label" in example and "answer" in example:
        msgs.append({"role": "assistant", "content": build_target_response(example)})
    return msgs


def parse_model_output(text: str) -> dict[str, str]:
    """Extract predicted_label and predicted_answer from model text."""
    text = text.strip()
    # Try JSON block first
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            label = obj.get("predicted_label") or obj.get("pred_label")
            answer = obj.get("predicted_answer") or obj.get("pred_option")
            if label and answer:
                return _normalize_prediction(label, answer)
        except json.JSONDecodeError:
            pass

    label = None
    answer = None
    lower = text.lower()
    if any(x in lower for x in ("no-hallucinate", "no_hallucination", "no hallucinate", "no hallucination")) or re.search(r"\bcorrect\b", lower):
        label = "no-hallucinate"
    elif "hallucination" in lower:
        label = "hallucination"

    answer_match = re.search(r"\b([A-F])\b", text)
    if answer_match:
        answer = answer_match.group(1).upper()

    if label and answer:
        return _normalize_prediction(label, answer)

    raise ValueError(f"Could not parse model output: {text[:200]!r}")


def _normalize_prediction(label: str, answer: str) -> dict[str, str]:
    label = label.strip().lower().replace("_", "-")
    if label in ("correct", "no-hallucinate", "no hallucinate", "no-hallucination", "no hallucination"):
        label = "no-hallucinate"
    elif label == "hallucination":
        label = "hallucination"
    else:
        raise ValueError(f"Invalid predicted_label: {label!r}")

    answer = answer.strip().upper()
    if answer not in VALID_OPTIONS:
        raise ValueError(f"Invalid predicted_answer: {answer!r}")
    return {"predicted_label": label, "predicted_answer": answer}


def predictions_to_submission(
    examples: list[dict[str, Any]],
    preds: list[dict[str, str]],
    *,
    codabench_format: bool = False,
) -> list[dict[str, str]]:
    rows = []
    for ex, pred in zip(examples, preds):
        if codabench_format:
            rows.append({
                "id": ex["id"],
                "pred_label": pred["predicted_label"].replace("-", "_"),
                "pred_option": pred["predicted_answer"],
            })
        else:
            rows.append({
                "id": ex["id"],
                "predicted_label": pred["predicted_label"],
                "predicted_answer": pred["predicted_answer"],
            })
    return rows


def compute_metrics(
    gold_examples: list[dict[str, Any]],
    preds: list[dict[str, str]],
    *,
    alpha: float = EVAL_ALPHA,
    beta: float = EVAL_BETA,
) -> dict[str, float]:
    """Compute hallucination detection and option accuracy metrics.

    Args:
        gold_examples: List of gold examples with 'label' and 'answer' fields.
        preds: List of prediction dicts with 'predicted_label' and 'predicted_answer'.
        alpha: Weight for Macro F1 (default: EVAL_ALPHA = 0.4).
        beta: Weight for Option Accuracy (default: EVAL_BETA = 0.6).
              Must satisfy alpha + beta == 1.0.
    """
    if abs(alpha + beta - 1.0) > 1e-9:
        raise ValueError(f"alpha + beta must equal 1.0, got {alpha} + {beta} = {alpha + beta}")

    gold_labels = [
        "correct" if ex["label"].strip().lower()
        in ("correct", "no-hallucinate", "no_hallucinate", "no-hallucination", "no_hallucination")
        else "hallucination"
        for ex in gold_examples
    ]
    pred_labels = [submission_label_to_gold(p["predicted_label"]) for p in preds]
    gold_options = [ex["answer"].strip().upper() for ex in gold_examples]
    pred_options = [p["predicted_answer"].strip().upper() for p in preds]

    macro_f1 = f1_score(gold_labels, pred_labels, average="macro", zero_division=0)
    option_acc = accuracy_score(gold_options, pred_options)
    combined = alpha * macro_f1 + beta * option_acc

    return {
        "macro_f1": macro_f1,
        "option_accuracy": option_acc,
        "combined_score": combined,
        "label_accuracy": accuracy_score(gold_labels, pred_labels),
    }

