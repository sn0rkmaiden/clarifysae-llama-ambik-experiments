from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_layer(hookpoint: str) -> int:
    match = re.search(r"layers\.(\d+)", hookpoint)
    if not match:
        raise ValueError(f"Cannot parse layer number from {hookpoint!r}")
    return int(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--ranking-output",
        default=None,
        help="Defaults to the output JSON path with .csv suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostics_path = Path(args.diagnostics)
    df = pd.read_csv(diagnostics_path)
    required = {"concept", "method", "hookpoint", "vector_key", "eval_auroc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{diagnostics_path} lacks required diagnostic columns: {sorted(missing)}"
        )

    ask = df[
        (df["concept"] == "ask_trajectory")
        & (df["method"] == "paired_difference")
    ][["hookpoint", "vector_key", "eval_auroc", "eval_balanced_accuracy"]].rename(
        columns={
            "vector_key": "ask_vector_key",
            "eval_auroc": "ask_eval_auroc",
            "eval_balanced_accuracy": "ask_eval_balanced_accuracy",
        }
    )
    gate = df[
        (df["concept"] == "ambiguity_state")
        & (df["method"] == "ridge_probe")
    ][["hookpoint", "vector_key", "eval_auroc", "eval_balanced_accuracy"]].rename(
        columns={
            "vector_key": "gate_vector_key",
            "eval_auroc": "gate_eval_auroc",
            "eval_balanced_accuracy": "gate_eval_balanced_accuracy",
        }
    )

    merged = ask.merge(gate, on="hookpoint", how="inner")
    if merged.empty:
        raise ValueError(
            "No layer contains both ask_trajectory/paired_difference and "
            "ambiguity_state/ridge_probe diagnostics."
        )
    merged = merged.dropna(subset=["ask_eval_auroc", "gate_eval_auroc"]).copy()
    if merged.empty:
        raise ValueError("All joint layer diagnostics have missing eval AUROC values.")

    merged["joint_min_auroc"] = merged[
        ["ask_eval_auroc", "gate_eval_auroc"]
    ].min(axis=1)
    merged["joint_mean_auroc"] = merged[
        ["ask_eval_auroc", "gate_eval_auroc"]
    ].mean(axis=1)
    merged["layer"] = merged["hookpoint"].map(parse_layer)
    merged = merged.sort_values(
        ["joint_min_auroc", "joint_mean_auroc", "layer"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    selected = merged.iloc[0].to_dict()
    selected.update(
        {
            "selection_rule": (
                "maximize the weaker held-out AUROC of the ambiguity probe and "
                "ask trajectory; break ties by mean AUROC"
            ),
            "diagnostics_path": str(diagnostics_path),
            "module_path": selected["hookpoint"],
        }
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    ranking_output = (
        Path(args.ranking_output)
        if args.ranking_output
        else output.with_suffix(".csv")
    )
    merged.to_csv(ranking_output, index=False)

    print(json.dumps(selected, indent=2))
    print(f"Layer ranking: {ranking_output}")


if __name__ == "__main__":
    main()
