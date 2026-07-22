from __future__ import annotations

"""Build an in-domain clarification actor corpus from disjoint AmbiK splits.

The target model's 1B writer was not reliable enough to create nine-field
synthetic JSON scenarios.  This builder uses AmbiK's human-authored matched
prompts, gold clarification questions, and clear-task plans instead.  It keeps
the detector/actor factorization intact while eliminating corpus-generation
noise from the corrective pilot.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from clarifysae_llama.runners.build_ambik_concept_corpus import (
    REQUIRED,
    _clean,
    _rows_for_example,
)


def _read_split(
    path: Path,
    *,
    split: str,
    generic_question: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], int]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    accepted = 0
    for _, source_row in frame.iterrows():
        source_id = _clean(source_row.get("id", ""))
        try:
            expanded = _rows_for_example(
                source_row,
                split=split,
                generic_question=generic_question,
            )
            if len(expanded) != 10:
                raise ValueError(
                    f"Expected 10 factorized rows, received {len(expanded)}"
                )
            rows.extend(expanded)
            accepted += 1
        except Exception as exc:
            failures.append(
                {
                    "source_id": source_id,
                    "split": split,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows, failures, accepted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--generic-question",
        default="Could you provide more details?",
    )
    parser.add_argument("--expected-train", type=int, default=60)
    parser.add_argument("--expected-dev", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    dev_path = Path(args.dev)
    output_path = Path(args.output)

    train_rows, train_failures, train_accepted = _read_split(
        train_path,
        split="train",
        generic_question=str(args.generic_question),
    )
    dev_rows, dev_failures, dev_accepted = _read_split(
        dev_path,
        split="dev",
        generic_question=str(args.generic_question),
    )
    rows = [*train_rows, *dev_rows]
    failures = [*train_failures, *dev_failures]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    split_source_counts = {
        "train": int(train_accepted),
        "dev": int(dev_accepted),
    }
    split_row_counts = {
        "train": int(len(train_rows)),
        "dev": int(len(dev_rows)),
    }
    metadata = {
        "corpus_type": "ambik_annotated_actor",
        "train_csv": str(train_path),
        "dev_csv": str(dev_path),
        "output": str(output_path),
        "source_examples": int(train_accepted + dev_accepted),
        "rows": int(len(rows)),
        "rows_per_source_example": 10,
        "split_source_counts": split_source_counts,
        "split_row_counts": split_row_counts,
        "failure_count": int(len(failures)),
        "failures": failures,
        "generic_question": str(args.generic_question),
        "actor_contrasts": {
            "ask_vs_guess": "gold clarification question vs gold-resolved plan under the ambiguous prompt",
            "targeted_vs_generic": "gold clarification question vs fixed generic question under the ambiguous prompt",
        },
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    problems: list[str] = []
    if train_accepted != int(args.expected_train):
        problems.append(
            f"accepted train examples={train_accepted}, expected={args.expected_train}"
        )
    if dev_accepted != int(args.expected_dev):
        problems.append(
            f"accepted dev examples={dev_accepted}, expected={args.expected_dev}"
        )
    if failures:
        problems.append(f"failed source rows={len(failures)}")

    print(json.dumps(metadata, indent=2))
    if problems:
        raise RuntimeError(
            "AmbiK actor corpus failed its integrity gate: "
            + "; ".join(problems)
            + f". Inspect {metadata_path}."
        )

    completion_path = output_path.with_suffix(".complete")
    completion_path.write_text(
        json.dumps(
            {
                "output": str(output_path),
                "metadata": str(metadata_path),
                "source_examples": int(train_accepted + dev_accepted),
                "rows": int(len(rows)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {completion_path}")


if __name__ == "__main__":
    main()
