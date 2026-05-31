from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import SYSTEM_PROMPT
from utils import example_to_messages, load_json_dataset


def process_file(in_path: str, out_path: str, system_prompt: str) -> None:
    examples = load_json_dataset(in_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for ex in examples:
            row = {
                "id": ex["id"],
                "messages": example_to_messages(ex, system_prompt),
                "metadata": {
                    "split": ex.get("split"),
                    "label": ex.get("label"),
                    "answer": ex.get("answer"),
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  {in_path}  →  {out_path}  ({len(examples)} examples)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Subtask 3 JSON to instruction-tuning JSONL")
    parser.add_argument("--input", action="append", dest="inputs", metavar="PATH")
    parser.add_argument("--output", action="append", dest="outputs", metavar="PATH")
    parser.add_argument("--system_prompt_file",
        type=str,
        default=None,
        help="Optional file overriding the default Arabic system prompt",
    )
    args = parser.parse_args()

    if not args.inputs or not args.outputs:
        parser.error("Provide at least one --input/--output pair.")
    if len(args.inputs) != len(args.outputs):
        parser.error("Number of --input and --output flags must match.")

    system_prompt = SYSTEM_PROMPT
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")

    print(f"Processing {len(args.inputs)} file(s)...")
    for in_path, out_path in zip(args.inputs, args.outputs):
        process_file(in_path, out_path, system_prompt)
    print("Done.")


if __name__ == "__main__":
    main()
