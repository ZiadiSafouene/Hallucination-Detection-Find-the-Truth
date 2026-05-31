from __future__ import annotations

import argparse
from pathlib import Path

from .config import ProjectConfig
from .codabench import score_codabench
from .data import load_examples, validate_examples
from .evaluate import evaluate_examples
from .model import load_model_registry, ModelPredictor
from .predict import run_prediction
from .prompts import build_prompt
from .utils import load_json, save_json


def command_predict(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    config = ProjectConfig.load(config_path)
    registry = load_model_registry(config_path.parent / "model_registry.yaml")
    model_name = args.model
    if not model_name:
        model_name = config.get_enabled_model_registry_key(registry)
    if not model_name:
        model_name = "fanar"

    domains = config.enabled_domain_names()
    if not domains:
        raise ValueError("No enabled domains found in configuration")

    for domain in domains:
        output_path = run_prediction(
            config_path=args.config,
            domain_name=domain,
            model_name=model_name,
            mode=args.mode,
            codabench=args.codabench_format,
            resume=args.resume,
            save_every=args.save_every,
        )
        print(f"Saved predictions: {output_path}")


def command_evaluate(args: argparse.Namespace) -> None:
    config = ProjectConfig.load(args.config)
    path = Path(args.prediction)
    predictions = load_json(path)
    if not isinstance(predictions, list):
        raise ValueError("Prediction file must contain a JSON list")

    domain_path = config.get_domain_dir(args.domain)
    gold_path = args.gold or (domain_path if domain_path.is_file() else domain_path / config.data.file_pattern)
    dev_examples = load_examples(gold_path)
    validation_errors = validate_examples(dev_examples)
    if validation_errors:
        raise ValueError("Gold validation errors:\n" + "\n".join(validation_errors))

    results = evaluate_examples(dev_examples, predictions, alpha=args.alpha, beta=args.beta)
    print("Evaluation results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


def command_codabench(args: argparse.Namespace) -> None:
    results = score_codabench(
        prediction_path=args.prediction_path,
        gold_path=args.gold_path,
        output_scores_path=args.output_scores_path,
        alpha=args.alpha,
        beta=args.beta,
    )
    print("Codabench scores written to:", args.output_scores_path)
    for key, value in results.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Arabic hallucination detection and answer selection pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict", help="Run model inference on dev files")
    predict_parser.add_argument("--config", required=True)
    predict_parser.add_argument("--model", choices=["fanar", "allam"], help="Model name to use")
    predict_parser.add_argument("--mode", choices=["single", "ensemble"], default="single")
    predict_parser.add_argument("--codabench-format", action="store_true")
    predict_parser.add_argument("--resume", action="store_true", default=True)
    predict_parser.add_argument("--no-resume", dest="resume", action="store_false")
    predict_parser.add_argument("--save-every", type=int, default=20)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate predictions against gold dev files")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--prediction", required=True)
    eval_parser.add_argument("--domain", choices=["islamic", "general_culture"], required=True)
    eval_parser.add_argument("--gold", help="Optional gold file override")
    eval_parser.add_argument("--alpha", type=float, default=0.5)
    eval_parser.add_argument("--beta", type=float, default=None)

    codabench_parser = subparsers.add_parser("codabench", help="Score Codabench-style predictions")
    codabench_parser.add_argument("--prediction-path", required=True)
    codabench_parser.add_argument("--gold-path", required=True)
    codabench_parser.add_argument("--output-scores-path", required=True)
    codabench_parser.add_argument("--alpha", type=float, default=0.5)
    codabench_parser.add_argument("--beta", type=float, default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "predict":
        command_predict(args)
    elif args.command == "evaluate":
        command_evaluate(args)
    elif args.command == "codabench":
        command_codabench(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
