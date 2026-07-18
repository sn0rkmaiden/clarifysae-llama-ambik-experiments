from __future__ import annotations

"""Build factorized concept-vector rows directly from paired AmbiK examples."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from clarifysae_llama.config import load_yaml
from clarifysae_llama.runners.generate_synthetic_corpus import _transcript
from clarifysae_llama.utils.io import ensure_dir


REQUIRED = {
    "id", "environment_full", "ambiguous_task", "unambiguous_direct",
    "question", "plan_for_clear_task",
}


def _split_for_id(value: str, train: float, dev: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    u = int.from_bytes(digest[:8], "big") / float(2**64)
    return "train" if u < train else "dev" if u < train + dev else "test"


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _rows_for_example(row: pd.Series, *, split: str, generic_question: str) -> list[dict[str, Any]]:
    sid = _clean(row["id"])
    context = _clean(row["environment_full"])
    ambiguous = _clean(row["ambiguous_task"])
    clear = _clean(row["unambiguous_direct"])
    question = _clean(row["question"])
    plan = _clean(row["plan_for_clear_task"])
    ambiguity_type = _clean(row.get("ambiguity_type", ""))
    if not all((sid, context, ambiguous, clear, question, plan)):
        raise ValueError(f"AmbiK row {sid!r} has an empty required value")

    metadata = {"source": "ambik", "ambiguity_type": ambiguity_type}
    output: list[dict[str, Any]] = []

    def add(concept: str, pair: str, label: int, variant: str, text: str, pooling: str):
        output.append({
            "id": f"ambik:{sid}:{concept}:{variant}",
            "pair_id": f"ambik:{sid}:{concept}:{pair}",
            "scenario_id": f"ambik:{sid}",
            "split": split,
            "concept": concept,
            "label": int(label),
            "variant": variant,
            "text": text,
            "recommended_pooling": pooling,
            "metadata": metadata,
        })

    add("ambiguity_state", "state", 1, "ambiguous_prompt",
        _transcript(context, ambiguous), "last_nonpad")
    add("ambiguity_state", "state", 0, "clear_prompt",
        _transcript(context, clear), "last_nonpad")

    # The annotated plan is placed after the ambiguous prompt as a benchmark
    # hard negative: it commits to the gold-resolved behavior instead of asking.
    add("ask_trajectory", "policy", 1, "gold_question",
        _transcript(context, ambiguous, question), "assistant_mean")
    add("ask_trajectory", "policy", 0, "gold_plan_without_question",
        _transcript(context, ambiguous, plan), "assistant_mean")

    add("targeted_question", "quality", 1, "gold_question",
        _transcript(context, ambiguous, question), "assistant_mean")
    add("targeted_question", "quality", 0, "generic_question",
        _transcript(context, ambiguous, generic_question), "assistant_mean")

    add("restraint_on_clear", "restraint", 1, "gold_plan",
        _transcript(context, clear, plan), "assistant_mean")
    add("restraint_on_clear", "restraint", 0, "unnecessary_question",
        _transcript(context, clear, "Do you want to provide any more details?"), "assistant_mean")

    add("neutral_prompt", "neutral_prompt", 0, "clear_prompt",
        _transcript(context, clear), "last_nonpad")
    add("neutral_response", "neutral_response", 0, "gold_plan",
        _transcript(context, clear, plan), "assistant_mean")
    return output


def run(config: dict[str, Any]) -> None:
    cfg = config["ambik_concept_corpus"]
    input_path = Path(cfg["input_path"])
    output_path = Path(cfg.get("output_path", "outputs/ambik/ambik_concept_corpus.jsonl"))
    ensure_dir(output_path.parent)
    df = pd.read_csv(input_path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"AmbiK input is missing columns: {sorted(missing)}")
    limit = cfg.get("limit")
    if limit is not None:
        df = df.head(int(limit)).copy()

    split_cfg = cfg.get("splits", {})
    train = float(split_cfg.get("train", 0.70))
    dev = float(split_cfg.get("dev", 0.15))
    if train <= 0 or dev < 0 or train + dev >= 1:
        raise ValueError("split fractions must satisfy train>0, dev>=0, train+dev<1")
    seed = int(config.get("seed", 42))
    generic_question = str(cfg.get("generic_question", "Could you provide more details?"))

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for _, source_row in df.iterrows():
        sid = _clean(source_row["id"])
        try:
            rows.extend(_rows_for_example(
                source_row,
                split=_split_for_id(sid, train, dev, seed),
                generic_question=generic_question,
            ))
        except Exception as exc:
            failures.append({"id": sid, "error": repr(exc)})

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "input_path": str(input_path),
        "rows": len(rows),
        "source_examples": len(rows) // 10,
        "failures": failures,
        "split_counts": {
            split: sum(1 for row in rows if row["split"] == split)
            for split in ("train", "dev", "test")
        },
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(load_yaml(args.config))
