from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    generated_answer: str
    gold_answer: str
    generator_model: str
    label: str
    options: Dict[str, str]
    raw: Dict[str, Any]
    correct_option: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "generated_answer": self.generated_answer,
            "gold_answer": self.gold_answer,
            "generator_model": self.generator_model,
            "label": self.label,
            "options": self.options,
            "correct_option": self.correct_option,
        }


def load_json(path: str | Path) -> Any:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    payload = json.loads(text)
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
        return payload["data"]

    return payload


def load_examples(path: str | Path, limit: Optional[int] = None) -> list[Example]:
    path = Path(path)
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")

    examples: list[Example] = []
    for index, row in enumerate(rows):
        if limit is not None and len(examples) >= limit:
            break

        options = {str(k): str(v) for k, v in dict(row.get("options", {})).items()}
        example = Example(
            id=str(row["id"]),
            question=str(row.get("question", "")),
            generated_answer=str(row.get("generated_answer", "")),
            gold_answer=str(row.get("gold_answer", "")),
            generator_model=str(row.get("generator_model", "")),
            label=str(row.get("label", row.get("gold_label", ""))),
            options=options,
            raw=row,
            correct_option=str(row.get("correct_option", row.get("answer", ""))),
        )
        examples.append(example)
    return examples


def validate_examples(examples: Iterable[Example]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()

    for index, example in enumerate(examples):
        prefix = f"example[{index}] id={example.id}"
        if not example.id.strip():
            errors.append(f"{prefix}: missing id")
            continue
        if example.id in seen_ids:
            errors.append(f"{prefix}: duplicate id")
        seen_ids.add(example.id)

        is_codabench_gold = "gold_label" in example.raw and "correct_option" in example.raw
        if not is_codabench_gold and not example.question.strip():
            errors.append(f"{prefix}: empty question")
        if not example.gold_answer.strip() and not example.correct_option.strip():
            errors.append(f"{prefix}: empty gold answer/correct option")
        if not is_codabench_gold and not example.generated_answer.strip():
            errors.append(f"{prefix}: empty generated_answer")
        if not is_codabench_gold and not example.generator_model.strip():
            errors.append(f"{prefix}: empty generator_model")

        normalized_label = example.label.strip().lower().replace("-", "_")
        if normalized_label not in {"no_hallucination", "hallucination", "correct", "hallucinate"}:
            errors.append(f"{prefix}: invalid label {example.label!r}")

        if example.options and not isinstance(example.options, dict):
            errors.append(f"{prefix}: options must be a dictionary")

    return errors
