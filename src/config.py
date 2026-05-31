from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int | None
    output_dir: str
    log_dir: str


@dataclass(frozen=True)
class DomainConfig:
    enabled: bool
    data_dir: str


@dataclass(frozen=True)
class DataConfig:
    file_pattern: str


@dataclass(frozen=True)
class EnsembleConfig:
    enabled: bool
    models: list[str]
    disagreement_strategy: str
    hallucination_confidence_threshold: float


@dataclass(frozen=True)
class CombinedScoreConfig:
    enabled: bool
    label_macro_f1_weight: float
    option_accuracy_weight: float


@dataclass(frozen=True)
class EvaluationConfig:
    main_metric: str
    compute_option_accuracy: bool
    combined_score: CombinedScoreConfig


@dataclass(frozen=True)
class CodabenchConfig:
    prediction_path: str
    gold_path: str
    output_scores_path: str


@dataclass(frozen=True)
class ProjectConfig:
    experiment: ExperimentConfig
    benchmark_project_dir: str
    domains: dict[str, DomainConfig]
    data: DataConfig
    models: dict[str, bool]
    ensemble: EnsembleConfig
    evaluation: EvaluationConfig
    codabench: CodabenchConfig

    @classmethod
    def load(cls, config_path: str | Path) -> "ProjectConfig":
        raw = load_yaml(config_path)
        experiment = raw.get("experiment", {})
        paths = raw.get("paths", {})
        domains = raw.get("domains", {})
        data = raw.get("data", {})
        models = raw.get("models", {})
        ensemble = raw.get("ensemble", {})
        evaluation = raw.get("evaluation", {})
        codabench = raw.get("codabench", {})

        return cls(
            experiment=ExperimentConfig(
                name=str(experiment.get("name", "arabic_hallucination_detection_find_truth")),
                seed=(None if experiment.get("seed") is None else int(experiment.get("seed"))),
                output_dir=str(experiment.get("output_dir", "outputs")),
                log_dir=str(experiment.get("log_dir", "logs")),
            ),
            benchmark_project_dir=str(paths.get("benchmark_project_dir", "")),
            domains={
                key: DomainConfig(
                    enabled=bool(value.get("enabled", True)),
                    data_dir=str(value["data_dir"]),
                )
                for key, value in domains.items()
            },
            data=DataConfig(
                file_pattern=str(data.get("file_pattern", "dev_*_gold.json")),
            ),
            models={str(key): bool(value) for key, value in models.items()},
            ensemble=EnsembleConfig(
                enabled=bool(ensemble.get("enabled", False)),
                models=[str(x) for x in ensemble.get("models", [])],
                disagreement_strategy=str(ensemble.get("disagreement_strategy", "high_confidence")),
                hallucination_confidence_threshold=float(
                    ensemble.get("hallucination_confidence_threshold", 0.75)
                ),
            ),
            evaluation=EvaluationConfig(
                main_metric=str(evaluation.get("main_metric", "macro_f1")),
                compute_option_accuracy=bool(evaluation.get("compute_option_accuracy", True)),
                combined_score=CombinedScoreConfig(
                    enabled=bool(evaluation.get("combined_score", {}).get("enabled", True)),
                    label_macro_f1_weight=float(
                        evaluation.get("combined_score", {}).get("label_macro_f1_weight", 0.5)
                    ),
                    option_accuracy_weight=float(
                        evaluation.get("combined_score", {}).get("option_accuracy_weight", 0.5)
                    ),
                ),
            ),
            codabench=CodabenchConfig(
                prediction_path=str(codabench.get("prediction_path", "/app/input/res/prediction.json")),
                gold_path=str(codabench.get("gold_path", "/app/input/ref/gold.json")),
                output_scores_path=str(codabench.get("output_scores_path", "/app/output/scores.json")),
            ),
        )

    def domain_names(self) -> list[str]:
        return list(self.domains.keys())

    def enabled_domain_names(self) -> list[str]:
        return [name for name, domain in self.domains.items() if domain.enabled]

    def get_domain_dir(self, domain_name: str) -> Path:
        value = self.domains.get(domain_name)
        if not value:
            raise KeyError(f"Unknown domain: {domain_name}")

        return Path(value.data_dir)

    def resolve_model_registry_path(self) -> Path:
        return Path("configs/model_registry.yaml")

    def get_enabled_model_registry_key(self, registry: dict[str, Any]) -> str | None:
        for model_name_or_path, enabled in self.models.items():
            if not enabled:
                continue
            normalized_name = model_name_or_path.strip()
            for key, model_config in registry.items():
                if (
                    model_config.model_name_or_path == normalized_name
                    or model_config.tokenizer_name_or_path == normalized_name
                    or key == normalized_name
                    or model_config.model_name_or_path.endswith(normalized_name)
                    or model_config.tokenizer_name_or_path.endswith(normalized_name)
                ):
                    return key
        return None
