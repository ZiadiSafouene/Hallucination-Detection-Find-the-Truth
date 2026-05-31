from __future__ import annotations

from pathlib import Path
from typing import Any

from .data import Example, load_examples, load_json as load_data_json
from .evaluate import evaluate_examples
from .utils import save_json


def write_prediction_file(predictions: list[dict[str, Any]], path: str | Path) -> None:
    save_json(predictions, path)


def load_codabench_gold(path: str | Path) -> list[Example]:
    return load_examples(path)


def load_codabench_predictions(path: str | Path) -> list[dict[str, Any]]:
    rows = load_data_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Prediction file must contain a JSON list: {path}")
    return list(rows)


def score_codabench(
    prediction_path: str | Path,
    gold_path: str | Path,
    output_scores_path: str | Path,
    alpha: float = 0.5,
    beta: float | None = None,
) -> dict[str, Any]:
    gold = load_codabench_gold(gold_path)
    predictions = load_codabench_predictions(prediction_path)
    results = evaluate_examples(gold, predictions, alpha=alpha, beta=beta)
    scores = {
        "combined_score": float(results["combined_score"]),
        "detection_f1": float(results["detection_f1"]),
        "detection_accuracy": float(results["label_accuracy"]),
        "macro_f1": float(results["macro_f1"]),
        "option_accuracy": float(results["option_accuracy"]),
        "eval_alpha": float(results["alpha"]),
        "num_gold_examples": int(results["num_gold_examples"]),
        "num_prediction_examples": int(results["num_prediction_examples"]),
        "num_valid_predictions": int(results["num_valid_predictions"]),
        "num_missing_predictions": int(results["num_missing_predictions"]),
        "num_extra_predictions": int(results["num_extra_predictions"]),
        "num_invalid_predictions": int(results["num_invalid_predictions"]),
    }
    save_json(scores, output_scores_path)
    return scores
