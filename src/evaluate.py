from __future__ import annotations

from typing import Any, Iterable

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .data import Example
from .utils import normalize_label


VALID_OPTIONS = {"A", "B", "C", "D", "E", "F"}


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def compute_label_metrics(gold: list[str], pred: list[str]) -> dict[str, float]:
    labels = ["no_hallucination", "hallucination"]
    accuracy = float(accuracy_score(gold, pred)) if gold else 0.0
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        gold,
        pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    tp = sum(1 for t, p in zip(gold, pred) if t == "hallucination" and p == "hallucination")
    fp = sum(1 for t, p in zip(gold, pred) if t == "no_hallucination" and p == "hallucination")
    fn = sum(1 for t, p in zip(gold, pred) if t == "hallucination" and p == "no_hallucination")
    tn = sum(1 for t, p in zip(gold, pred) if t == "no_hallucination" and p == "no_hallucination")

    detection_precision = safe_div(tp, tp + fp)
    detection_recall = safe_div(tp, tp + fn)
    detection_f1 = safe_div(2 * detection_precision * detection_recall, detection_precision + detection_recall)

    return {
        "label_accuracy": accuracy,
        "detection_accuracy": accuracy,
        "detection_precision": detection_precision,
        "detection_recall": detection_recall,
        "detection_f1": detection_f1,
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def compute_option_accuracy(gold: list[str], pred: list[str]) -> float:
    if not gold:
        return 0.0
    return float(accuracy_score(gold, pred))


def compute_combined_score(
    label_f1: float,
    option_accuracy: float,
    alpha: float = 0.5,
    beta: float | None = None,
) -> float:
    if beta is None:
        beta = 1.0 - alpha
    if abs(alpha + beta - 1.0) > 1e-9:
        raise ValueError("alpha + beta must equal 1.0")
    return alpha * label_f1 + beta * option_accuracy


def build_gold_label(example: Example) -> str:
    value = example.raw.get("gold_label") or example.label
    return normalize_label(str(value))


def build_gold_option(example: Example) -> str:
    value = (
        example.raw.get("correct_option")
        or example.raw.get("answer")
        or getattr(example, "correct_option", "")
        or example.gold_answer
    )
    return str(value or "").strip().upper()


def prediction_label(prediction: dict[str, Any]) -> str:
    value = prediction.get("pred_label")
    if value is None:
        value = prediction.get("predicted_label")
    if value is None:
        value = prediction.get("prediction_label")
    if value is None:
        value = prediction.get("label")
    return normalize_label(str(value))


def prediction_option(prediction: dict[str, Any], label: str | None = None) -> str:
    value = prediction.get("pred_option")
    if value is None:
        value = prediction.get("answer")
    if value is None:
        value = prediction.get("selected_option")
    if value is None:
        value = prediction.get("predicted_answer")
    return str(value or "").strip().upper()


def _valid_options_for(example: Example) -> set[str]:
    return {str(key).upper() for key in example.options} or VALID_OPTIONS


def _validate_prediction(example: Example, prediction: dict[str, Any]) -> tuple[str, str, str | None]:
    try:
        pred_label = prediction_label(prediction)
    except Exception as exc:
        return "", "", f"Invalid pred_label: {exc}"

    pred_option = prediction_option(prediction, pred_label)
    valid_options = _valid_options_for(example)
    if pred_option not in valid_options:
        return pred_label, pred_option, f"Invalid pred_option: {pred_option!r}"

    return pred_label, pred_option, None


def evaluate_examples(
    examples: Iterable[Example],
    predictions: Iterable[dict[str, Any]],
    alpha: float = 0.5,
    beta: float | None = None,
) -> dict[str, Any]:
    example_list = list(examples)
    prediction_list = list(predictions)
    gold_by_id = {str(example.id): example for example in example_list}
    pred_by_id = {str(prediction.get("id")): prediction for prediction in prediction_list if isinstance(prediction, dict)}

    missing_predictions: list[str] = []
    extra_predictions = [pred_id for pred_id in pred_by_id if pred_id not in gold_by_id]
    invalid_predictions: list[dict[str, str]] = []

    gold_labels: list[str] = []
    pred_labels: list[str] = []
    gold_options: list[str] = []
    pred_options: list[str] = []

    for qid, example in gold_by_id.items():
        prediction = pred_by_id.get(qid)
        if prediction is None:
            missing_predictions.append(qid)
            continue

        pred_label, pred_option, error = _validate_prediction(example, prediction)
        if error is not None:
            invalid_predictions.append({"id": qid, "error": error})
            continue

        gold_labels.append(build_gold_label(example))
        pred_labels.append(pred_label)
        gold_options.append(build_gold_option(example))
        pred_options.append(pred_option)

    label_metrics = compute_label_metrics(gold_labels, pred_labels)
    option_accuracy = compute_option_accuracy(gold_options, pred_options)
    combined_score = compute_combined_score(
        label_metrics["detection_f1"],
        option_accuracy,
        alpha=alpha,
        beta=beta,
    )

    results = {
        "num_gold_examples": len(example_list),
        "num_prediction_examples": len(prediction_list),
        "num_valid_predictions": len(gold_labels),
        "num_missing_predictions": len(missing_predictions),
        "num_extra_predictions": len(extra_predictions),
        "num_invalid_predictions": len(invalid_predictions),
        "label_accuracy": label_metrics["label_accuracy"],
        "detection_f1": label_metrics["detection_f1"],
        "detection_precision": label_metrics["detection_precision"],
        "detection_recall": label_metrics["detection_recall"],
        "macro_f1": label_metrics["macro_f1"],
        "macro_precision": label_metrics["macro_precision"],
        "macro_recall": label_metrics["macro_recall"],
        "option_accuracy": option_accuracy,
        "combined_score": combined_score,
        "alpha": alpha,
        "beta": 1.0 - alpha if beta is None else beta,
        "hallucination_detection": {
            "positive_class": "hallucination",
            "true_positive": label_metrics["true_positive"],
            "false_positive": label_metrics["false_positive"],
            "false_negative": label_metrics["false_negative"],
            "true_negative": label_metrics["true_negative"],
            "precision": label_metrics["detection_precision"],
            "recall": label_metrics["detection_recall"],
            "f1": label_metrics["detection_f1"],
            "accuracy": label_metrics["detection_accuracy"],
            "macro_f1": label_metrics["macro_f1"],
        },
        "option_selection": {
            "correct": sum(1 for gold, pred in zip(gold_options, pred_options) if gold == pred),
            "total": len(gold_options),
            "accuracy": option_accuracy,
        },
        "missing_predictions": missing_predictions,
        "extra_predictions": extra_predictions,
        "invalid_predictions": invalid_predictions,
    }
    return results
