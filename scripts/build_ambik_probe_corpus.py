from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from clarifysae_llama.data.ambik_loader import load_ambik_clarification_dataset
from clarifysae_llama.data.prompting import build_clarification_prompt


def rows_from_file(path: Path, *, split: str, max_questions: int) -> list[dict[str, Any]]:
    frame = load_ambik_clarification_dataset(path)
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        variant = str(row.get("pair_variant", "")).strip().lower()
        if variant not in {"ambiguous", "clear"}:
            raise ValueError(
                f"{path} must be a paired AmbiK CSV with pair_variant; got {variant!r}"
            )
        source_id = str(row.get("source_id", row["id"]))
        label = 1 if variant == "ambiguous" else 0
        rows.append(
            {
                "id": f"ambik:{source_id}:{variant}",
                "pair_id": f"ambik:{source_id}:ambiguity_state",
                "scenario_id": f"ambik:{source_id}",
                "split": split,
                "concept": "ambik_ambiguity_state",
                "label": label,
                "variant": f"{variant}_prompt",
                "text": build_clarification_prompt(
                    description=str(row["environment_full"]),
                    task=str(row["ambiguous_task"]),
                    max_questions=max_questions,
                ),
                "recommended_pooling": "last_nonpad",
                "metadata": {
                    "source_id": source_id,
                    "pair_variant": variant,
                    "ambiguity_type": str(row["ambiguity_type"]),
                    "source_csv": str(path),
                },
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-questions", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    dev_path = Path(args.dev)
    for path in (train_path, dev_path):
        if not path.exists():
            raise FileNotFoundError(path)

    rows = [
        *rows_from_file(train_path, split="train", max_questions=args.max_questions),
        *rows_from_file(dev_path, split="dev", max_questions=args.max_questions),
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['split']}:{row['variant']}"
        counts[key] = counts.get(key, 0) + 1
    metadata = {
        "output": str(output),
        "train_csv": str(train_path),
        "dev_csv": str(dev_path),
        "rows": len(rows),
        "counts": counts,
        "concept": "ambik_ambiguity_state",
        "prompt_template": "build_clarification_prompt",
        "max_questions": int(args.max_questions),
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
