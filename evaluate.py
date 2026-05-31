"""
evaluate.py
───────────
Score dev predictions against gold labels.

Usage
-----
python evaluate.py --task 1 --pred predictions/task1/predictions_silma.json
python evaluate.py --task 2 --pred predictions/task2/predictions_qwen.json
"""

from __future__ import annotations

import argparse
import json

from config import TASK_REGISTRY, EVAL_ALPHA, EVAL_BETA
from utils import compute_metrics, load_json_dataset


def load_predictions(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    keyed = {}
    for row in rows:
        pid = row["id"]
        label = row.get("predicted_label") or row.get("pred_label", "")
        answer = row.get("predicted_answer") or row.get("pred_option", "")
        if "pred_label" in row:
            label = label.replace("_", "-")
        keyed[pid] = {
            "predicted_label": label,
            "predicted_answer": answer.strip().upper(),
        }
    return keyed


def main():
    parser = argparse.ArgumentParser(description="Evaluate dev predictions")
    parser.add_argument("--task", type=str, choices=["1", "2"], required=True)
    parser.add_argument("--dev", type=str, default=None,
                        help="Dev JSON (default: from task registry)")
    parser.add_argument("--pred", type=str, required=True,
                        help="Predictions JSON from predict.py")
    parser.add_argument("--alpha", type=float, default=EVAL_ALPHA,
                        help=f"Weight on Macro F1 (default: {EVAL_ALPHA})")
    parser.add_argument("--beta", type=float, default=EVAL_BETA,
                        help=f"Weight on Option Accuracy (default: {EVAL_BETA})")
    args = parser.parse_args()

    dev_path = args.dev or TASK_REGISTRY[args.task]["dev_path"]
    gold = load_json_dataset(dev_path)
    print(f"Task {args.task} — {TASK_REGISTRY[args.task]['name']}")
    print(f"Gold: {dev_path}")
    pred_map = load_predictions(args.pred)

    aligned_gold = []
    aligned_pred = []
    missing = 0
    for ex in gold:
        if ex["id"] not in pred_map:
            missing += 1
            continue
        aligned_gold.append(ex)
        aligned_pred.append(pred_map[ex["id"]])

    if missing:
        print(f"Warning: {missing} gold ids missing from predictions")

    metrics = compute_metrics(aligned_gold, aligned_pred, alpha=args.alpha, beta=args.beta)
    print(f"Instances       : {len(aligned_gold)}")
    print(f"Macro F1        : {metrics['macro_f1']:.4f}")
    print(f"Option accuracy : {metrics['option_accuracy']:.4f}")
    print(f"Label accuracy  : {metrics['label_accuracy']:.4f}")
    print(f"Combined score  : {metrics['combined_score']:.4f}  (α={args.alpha}, β={args.beta})")


if __name__ == "__main__":
    main()
