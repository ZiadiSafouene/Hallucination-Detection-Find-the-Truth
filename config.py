"""
Shared configuration for HalluScoring 2026 Subtask 3.
Hallucination Detection & Find the Truth — two tracks, unified two-step task.
"""

from __future__ import annotations

import os

# ── Competition labels ────────────────────────────────────────────────────────
# Gold labels in released JSON use "correct" / "hallucination".
# Codabench submission uses "no-hallucinate" / "hallucination".
GOLD_TO_SUBMISSION_LABEL = {
    "correct": "no-hallucinate",
    "hallucination": "hallucination",
    "no-hallucinate": "no-hallucinate",
    "no_hallucination": "no-hallucinate",
}

SUBMISSION_TO_GOLD_LABEL = {
    "no-hallucinate": "correct",
    "no_hallucination": "correct",
    "hallucination": "hallucination",
    "correct": "correct",
}

VALID_OPTIONS = ("A", "B", "C", "D", "E", "F")

# Checkpoint folder: outputs/task<N>/<model_key>/best_model/
MODEL_SAVE_DIR = "best_model"

# ── Tasks (Codabench Subtask 3) ───────────────────────────────────────────────
# Task 1: Islamic domain          — data/islamic/
# Task 2: General cultural domain — data/general_culture/
TASK_REGISTRY: dict[str, dict] = {
    "1": {
        "name": "Islamic Domain",
        "train_path": "data/islamic/islamic_train.json",
        "dev_path": "data/islamic/islamic_dev.json",
    },
    "2": {
        "name": "General Cultural Domain",
        "train_path": "data/general_culture/train_subtask3-2.json",
        "dev_path": "data/general_culture/dev_subtask3-2.json",
    },
}


def task_output_dir(base: str, task: str) -> str:
    """e.g. ./outputs/task1"""
    return os.path.join(base, f"task{task}")


def resolve_tasks(task_args: list[str]) -> list[str]:
    if "all" in task_args:
        return ["1", "2"]
    return task_args

# Final score weights (Codabench overview page)
# combined = EVAL_ALPHA * macro_f1 + EVAL_BETA * option_accuracy
EVAL_ALPHA = 0.4  # weight on Macro F1 (hallucination detection)
EVAL_BETA = 0.6   # weight on option accuracy (find the truth)
# Note: compute_metrics uses (1.0 - alpha) for EVAL_BETA to stay consistent.

# ── Base LLMs (< 13B, competition rule) ───────────────────────────────────────
MODEL_REGISTRY: dict[str, dict] = {
    "silma": {
        "hf_id": "silma-ai/SILMA-9B-Instruct-v1.0",
        "display": "SILMA 1.0 (9B)",
        "max_length": 2048,
    },
    "falcon_arabic": {
        # Built on Falcon3-7B; request access on Hugging Face if gated.
        "hf_id": "tiiuae/Falcon-Arabic-7B-Instruct",
        "fallback_hf_id": "tiiuae/Falcon3-7B-Instruct",
        "display": "Falcon Arabic (7B)",
        "max_length": 2048,
    },
    "qwen": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "display": "Qwen 2.5 (7B)",
        "max_length": 2048,
    },
    "allam": {
        "hf_id": "humain-ai/ALLaM-7B-Instruct-preview",
        "display": "ALLaM (7B)",
        "max_length": 2048,
    },
}

SYSTEM_PROMPT = """أنت نظام للتحقق من صحة إجابات النماذج اللغوية باللغة العربية.
مهمتك خطوتان:
1) تحديد ما إذا كانت الإجابة المولّدة موثوقة (no-hallucinate) أم تحتوي هلوسة (hallucination).
2) اختيار الإجابة الصحيحة من الخيارات A–F.

أجب بصيغة JSON فقط، بدون أي نص إضافي:
{"predicted_label": "no-hallucinate" | "hallucination", "predicted_answer": "A"|"B"|"C"|"D"|"E"|"F"}"""
