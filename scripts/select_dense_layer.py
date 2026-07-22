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
    parser.add_argument(
        "--diagnostics",
        default=None,
        help="Backward-compatible single diagnostics CSV.",
    )
    parser.add_argument("--actor-diagnostics", default=None)
    parser.add_argument("--gate-diagnostics", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--ranking-output",
        default=None,
        help="Defaults to the output JSON path with .csv suffix.",
    )
    return parser.parse_args()


def _load(path: str | None, label: str) -> tuple[Path, pd.DataFrame]:
    if not path:
        raise ValueError(f"Missing --{label}-diagnostics")
    resolved = Path(path)
    frame = pd.read_csv(resolved)
    required = {"concept", "method", "hookpoint", "vector_key", "eval_auroc"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{resolved} lacks required diagnostic columns: {sorted(missing)}"
        )
    return resolved, frame


def main() -> None:
    args = parse_args()
    actor_path = args.actor_diagnostics or args.diagnostics
    gate_path = args.gate_diagnostics or args.diagnostics
    actor_diagnostics_path, actor_df = _load(actor_path, "actor")
    gate_diagnostics_path, gate_df = _load(gate_path, "gate")

    actor = actor_df[
        (actor_df["concept"] == "targeted_clarification")
        & (actor_df["method"] == "composite_paired_difference")
    ].copy()
    if actor.empty:
        raise ValueError(
            "Actor diagnostics contain no targeted_clarification/"
            "composite_paired_difference rows."
        )
    required_actor = {
        "eval_auroc_ask_vs_guess",
        "eval_auroc_targeted_vs_generic",
    }
    missing_actor = required_actor - set(actor.columns)
    if missing_actor:
        raise ValueError(
            f"{actor_diagnostics_path} lacks composite actor diagnostics: "
            f"{sorted(missing_actor)}"
        )
    actor = actor[
        [
            "hookpoint",
            "vector_key",
            "eval_auroc",
            "eval_auroc_ask_vs_guess",
            "eval_auroc_targeted_vs_generic",
        ]
    ].rename(
        columns={
            "vector_key": "ask_vector_key",
            "eval_auroc": "actor_min_eval_auroc",
            "eval_auroc_ask_vs_guess": "actor_ask_vs_guess_eval_auroc",
            "eval_auroc_targeted_vs_generic": (
                "actor_targeted_vs_generic_eval_auroc"
            ),
        }
    )

    gate = gate_df[
        (gate_df["concept"] == "ambik_ambiguity_state")
        & (gate_df["method"] == "ridge_probe")
    ][
        ["hookpoint", "vector_key", "eval_auroc", "eval_balanced_accuracy"]
    ].rename(
        columns={
            "vector_key": "gate_vector_key",
            "eval_auroc": "gate_eval_auroc",
            "eval_balanced_accuracy": "gate_eval_balanced_accuracy",
        }
    )

    merged = actor.merge(gate, on="hookpoint", how="inner")
    merged = merged.dropna(
        subset=[
            "actor_ask_vs_guess_eval_auroc",
            "actor_targeted_vs_generic_eval_auroc",
            "gate_eval_auroc",
        ]
    ).copy()
    if merged.empty:
        raise ValueError(
            "No layer has complete actor and AmbiK gate held-out diagnostics."
        )

    merged["joint_min_auroc"] = merged[
        [
            "actor_ask_vs_guess_eval_auroc",
            "actor_targeted_vs_generic_eval_auroc",
            "gate_eval_auroc",
        ]
    ].min(axis=1)
    merged["joint_mean_auroc"] = merged[
        [
            "actor_ask_vs_guess_eval_auroc",
            "actor_targeted_vs_generic_eval_auroc",
            "gate_eval_auroc",
        ]
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
                "maximize the weakest held-out AUROC among targeted-vs-guess, "
                "targeted-vs-generic, and real-AmbiK ambiguity detection; "
                "break ties by their mean"
            ),
            "actor_diagnostics_path": str(actor_diagnostics_path),
            "gate_diagnostics_path": str(gate_diagnostics_path),
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
