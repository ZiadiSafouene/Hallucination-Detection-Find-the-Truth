from __future__ import annotations

from collections import Counter
from typing import Iterable, Any

from .utils import normalize_label


def combine_predictions(
    candidate_rows: Iterable[Any],
    option_keys: list[str],
    threshold: float = 0.75,
) -> Any:
    rows = list(candidate_rows)
    labels = [normalize_label(row.predicted_label) for row in rows]
    label_counts = Counter(labels)
    total = len(rows)

    most_common_label, count = label_counts.most_common(1)[0]
    if count / total >= threshold:
        selected_label = most_common_label
    else:
        selected_label = "hallucination" if "hallucination" in label_counts else most_common_label

    option_votes = [row.selected_option for row in rows if row.selected_option]
    option_counts = Counter(option_votes)
    selected_option = None
    if selected_label == "hallucination" and option_votes:
        candidate, _ = option_counts.most_common(1)[0]
        if candidate in option_keys:
            selected_option = candidate

    raw_output = " | ".join(row.raw_output for row in rows if row.raw_output)

    from .predict import PredictionRow

    return PredictionRow(
        id=rows[0].id,
        question=rows[0].question,
        model_answer=rows[0].model_answer,
        predicted_label=selected_label,
        selected_option=selected_option,
        raw_output=raw_output,
    )
