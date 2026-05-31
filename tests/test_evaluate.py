from src.data import Example
from src.evaluate import compute_label_metrics, compute_option_accuracy, compute_combined_score, evaluate_examples


def test_compute_label_metrics():
    gold = ["no_hallucination", "hallucination", "no_hallucination"]
    pred = ["no_hallucination", "hallucination", "hallucination"]
    metrics = compute_label_metrics(gold, pred)
    assert round(metrics["label_accuracy"], 3) == 0.667
    assert round(metrics["macro_f1"], 3) == 0.667


def test_compute_option_accuracy():
    gold = ["A", "B", "C"]
    pred = ["A", "C", "C"]
    assert compute_option_accuracy(gold, pred) == 2 / 3


def test_compute_combined_score():
    assert compute_combined_score(0.5, 0.8, alpha=0.7, beta=0.3) == 0.5 * 0.7 + 0.8 * 0.3


def test_evaluate_examples_matches_codabench_option_scoring():
    examples = [
        Example("1", "q1", "a1", "gold", "m", "no_hallucination", {"A": "a"}, {}, "A"),
        Example("2", "q2", "a2", "gold", "m", "hallucination", {"A": "a", "B": "b"}, {}, "B"),
    ]
    predictions = [
        {"id": "1", "prediction_label": "no_hallucination", "answer": "A"},
        {"id": "2", "prediction_label": "hallucination", "answer": "B"},
    ]

    results = evaluate_examples(examples, predictions)

    assert results["option_accuracy"] == 1.0
    assert results["detection_f1"] == 1.0
    assert results["macro_f1"] == 1.0
