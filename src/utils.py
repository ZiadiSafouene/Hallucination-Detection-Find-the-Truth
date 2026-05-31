from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{(?:[^{}]|\n|\r|\t)*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def normalize_label(value: str) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"no_hallucination", "no_hallucinate", "correct"}:
        return "no_hallucination"
    if text in {"hallucination", "hallucinate", "hallucinating"}:
        return "hallucination"
    raise ValueError(f"Unknown label: {value!r}")


def normalize_option(value: Any, option_keys: list[str]) -> str | None:
    if value is None:
        return None
    token = str(value).strip().upper()
    if token in option_keys:
        return token

    # Accept answers like "A" or "B" embedded in user text.
    for option in option_keys:
        if re.search(rf"\b{re.escape(option)}\b", token, re.IGNORECASE):
            return option

    match = re.search(r"\b([A-Z])\b", token)
    if match:
        candidate = match.group(1).upper()
        if candidate in option_keys:
            return candidate
    return None


def parse_model_output(text: str, option_keys: list[str]) -> dict[str, Any]:
    text = str(text).strip()
    parse_text = text.rsplit("JSON:", 1)[-1].strip() if "JSON:" in text else text
    json_object = extract_json_object(parse_text)
    if json_object is not None:
        label = json_object.get("predicted_label") or json_object.get("label")
        option = (
            json_object.get("selected_option")
            or json_object.get("predicted_answer")
            or json_object.get("answer")
            or json_object.get("pred_option")
        )
        if label is not None:
            normalized_label = normalize_label(label)
            selected_option = normalize_option(option, option_keys) if option is not None else None
            return {
                "predicted_label": normalized_label,
                "selected_option": selected_option,
                "raw_output": text,
            }

    lower = parse_text.lower()
    if "no_hallucination" in lower or "no hallucination" in lower or "no_hallucinate" in lower or "correct" in lower:
        label = "no_hallucination"
    elif "hallucination" in lower:
        label = "hallucination"
    else:
        raise ValueError(f"Could not parse predicted label from: {text!r}")

    selected_option = normalize_option(parse_text, option_keys)
    return {
        "predicted_label": label,
        "selected_option": selected_option,
        "raw_output": text,
    }
