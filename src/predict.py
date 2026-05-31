from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .data import Example, load_examples, validate_examples
from .ensemble import combine_predictions
from .model import ModelPredictor, load_model_registry
from .prompts import build_prompt
from .utils import ensure_dir, normalize_label, parse_model_output, save_json


@dataclass(frozen=True)
class PredictionRow:
    id: str
    question: str
    model_answer: str
    predicted_label: str
    selected_option: str | None
    raw_output: str

    def to_dict(self, codabench: bool = False) -> dict[str, Any]:
        if codabench:
            return {
                "id": self.id,
                "pred_label": self.predicted_label.replace("_", "_"),
                "pred_option": self.selected_option,
            }
        output = {
            "id": self.id,
            "question": self.question,
            "model_answer": self.model_answer,
            "prediction_label": self.predicted_label,
        }
        if self.selected_option:
            output["answer"] = self.selected_option
        return output


def _load_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    return {row["id"]: row for row in data if isinstance(row, dict) and "id" in row}


def _predict_examples(predictor: ModelPredictor, examples: list[Example]) -> list[PredictionRow]:
    prompts = [build_prompt(example) for example in examples]
    raw_outputs = predictor.generate(prompts)
    rows: list[PredictionRow] = []
    for example, text in zip(examples, raw_outputs):
        option_keys = list(example.options.keys())
        row = parse_model_output(text, option_keys)
        selected_option = row["selected_option"] or (option_keys[0] if option_keys else None)
        rows.append(
            PredictionRow(
                id=example.id,
                question=example.question,
                model_answer=example.generated_answer,
                predicted_label=row["predicted_label"],
                selected_option=selected_option,
                raw_output=row["raw_output"],
            )
        )
    return rows


def save_predictions(rows: list[PredictionRow], path: Path, codabench: bool) -> None:
    payload = [row.to_dict(codabench=codabench) for row in rows]
    save_json(payload, path)


def _predict_model(
    predictor: ModelPredictor,
    examples: list[Example],
) -> dict[str, PredictionRow]:
    rows = _predict_examples(predictor, examples)
    return {row.id: row for row in rows}


def _ensemble_predictions(
    prediction_sets: list[dict[str, PredictionRow]],
    examples: list[Example],
    threshold: float,
) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    for example in examples:
        candidate_rows = [preds[example.id] for preds in prediction_sets if example.id in preds]
        if not candidate_rows:
            raise ValueError(f"Missing ensemble predictions for example {example.id}")
        rows.append(combine_predictions(candidate_rows, list(example.options.keys()), threshold=threshold))
    return rows


def run_prediction(
    config_path: str,
    domain_name: str,
    model_name: str | None,
    mode: str,
    codabench: bool = False,
    resume: bool = True,
    save_every: int = 20,
) -> Path:
    config_path = Path(config_path).resolve()
    config = ProjectConfig.load(config_path)
    registry_path = config_path.parent / "model_registry.yaml"
    registry = load_model_registry(registry_path)

    if mode == "ensemble":
        if not config.ensemble.enabled:
            raise ValueError("Ensemble mode is disabled in the configuration")
        model_names = config.ensemble.models
        output_name = "ensemble"
    else:
        if not model_name:
            model_name = "fanar"
        if model_name not in registry:
            raise KeyError(f"Unknown model: {model_name}")
        model_names = [model_name]
        output_name = model_name

    domain_path = config.get_domain_dir(domain_name)
    if domain_path.is_dir():
        examples_path = domain_path / config.data.file_pattern
    else:
        examples_path = domain_path
    examples = load_examples(examples_path)
    validation_errors = validate_examples(examples)
    if validation_errors:
        raise ValueError("Validation errors:\n" + "\n".join(validation_errors))

    prediction_output_base = Path(config.experiment.output_dir) / config.experiment.name / "predictions"
    output_path = prediction_output_base / f"prediction_{domain_name}_{output_name}.json"
    ensure_dir(output_path.parent)

    if mode == "ensemble":
        predictions_by_model: list[dict[str, PredictionRow]] = []
        for name in model_names:
            if name not in registry:
                raise KeyError(f"Unknown model in ensemble: {name}")
            print(f"Running ensemble model: {name}", flush=True)
            predictor = ModelPredictor(registry[name])
            predictions_by_model.append(_predict_model(predictor, examples))

        rows = _ensemble_predictions(
            prediction_sets=predictions_by_model,
            examples=examples,
            threshold=config.ensemble.hallucination_confidence_threshold,
        )
        save_predictions(rows, output_path, codabench=codabench)
        print(f"Ensemble prediction complete. Saved {len(rows)} rows.", flush=True)
        return output_path

    saved_rows: list[PredictionRow] = []
    if resume and output_path.exists():
        existing = _load_existing_predictions(output_path)
        example_map = {example.id: example for example in examples}
        for row in existing.values():
            example = example_map.get(str(row["id"]))
            predicted_label = str(row.get("prediction_label", row.get("predicted_label", row.get("pred_label", ""))))
            predicted_label = normalize_label(predicted_label)
            selected_option = row.get("answer", row.get("selected_option", row.get("pred_option")))
            saved_rows.append(PredictionRow(
                id=str(row["id"]),
                question=str(row.get("question", example.question if example else "")),
                model_answer=str(row.get("model_answer", row.get("raw_output", ""))),
                predicted_label=predicted_label,
                selected_option=selected_option,
                raw_output=str(row.get("raw_output", "")),
            ))

    done_ids = {row.id for row in saved_rows}
    new_rows: list[PredictionRow] = []

    predictor = ModelPredictor(registry[model_name])
    pending = [ex for ex in examples if ex.id not in done_ids]
    total_pending = len(pending)
    for index, example in enumerate(pending, start=1):
        print(f"[{index}/{total_pending}] Predicting id={example.id}...", flush=True)
        row = _predict_examples(predictor, [example])[0]
        new_rows.append(row)
        if len(new_rows) % save_every == 0:
            save_predictions(saved_rows + new_rows, output_path, codabench=codabench)
            print(f"  Saved {len(saved_rows) + len(new_rows)} predictions so far.", flush=True)

    if new_rows:
        save_predictions(saved_rows + new_rows, output_path, codabench=codabench)
    print(f"Prediction complete. Total predictions saved: {len(saved_rows) + len(new_rows)}", flush=True)
    return output_path
