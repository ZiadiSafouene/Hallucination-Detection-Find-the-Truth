from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_name_or_path: str
    tokenizer_name_or_path: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    batch_size: int = 4
    trust_remote_code: bool = True


class ModelPredictor:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
        dtype = getattr(torch, config.dtype, None)
        if dtype is None or self.device.type != "cuda":
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=dtype,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=config.trust_remote_code,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompts: list[str]) -> list[str]:
        outputs: list[str] = []
        if not prompts:
            return outputs

        for offset in range(0, len(prompts), self.config.batch_size):
            batch = prompts[offset : offset + self.config.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            generated = generated[:, inputs["input_ids"].shape[1]:]
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend(text.strip() for text in decoded)

        return outputs


def build_model_config(name: str, raw: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        name=name,
        model_name_or_path=str(raw["model_name_or_path"]),
        tokenizer_name_or_path=str(raw["tokenizer_name_or_path"]),
        device=str(raw.get("device", "cuda")),
        dtype=str(raw.get("dtype", "bfloat16")),
        max_new_tokens=int(raw.get("max_new_tokens", 512)),
        temperature=float(raw.get("temperature", 0.0)),
        top_p=float(raw.get("top_p", 1.0)),
        batch_size=int(raw.get("batch_size", 4)),
        trust_remote_code=bool(raw.get("trust_remote_code", True)),
    )


def load_model_registry(path: str | Path) -> dict[str, ModelConfig]:
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        raw = json.loads(raw_text)
    else:
        raw = yaml.safe_load(raw_text)
    if not isinstance(raw, dict):
        raise ValueError(f"Model registry must contain a mapping of models: {path}")
    models = raw.get("models", {})
    return {key: build_model_config(key, value) for key, value in models.items()}
