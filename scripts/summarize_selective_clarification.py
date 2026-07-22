from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def summarize_metrics(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    required = {
        "gold_ambiguous",
        "asked_question",
        "resolved_proxy_any",
        "num_questions",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")

    gold_ambiguous = _to_bool(df["gold_ambiguous"])
    asked = _to_bool(df["asked_question"])
    resolved = _to_bool(df["resolved_proxy_any"])
    ambiguous = df[gold_ambiguous].copy()
    clear = df[~gold_ambiguous].copy()

    result: dict[str, Any] = {
        "example_metrics_path": str(path),
        "n_total": int(len(df)),
        "n_ambiguous": int(len(ambiguous)),
        "n_clear": int(len(clear)),
        "asking_rate_ambiguous": float(asked[gold_ambiguous].mean())
        if len(ambiguous)
        else None,
        "overasking_rate_clear": float(asked[~gold_ambiguous].mean())
        if len(clear)
        else None,
        "resolved_proxy_any_ambiguous": float(resolved[gold_ambiguous].mean())
        if len(ambiguous)
        else None,
        "mean_questions_ambiguous": float(
            pd.to_numeric(ambiguous["num_questions"], errors="coerce").mean()
        )
        if len(ambiguous)
        else None,
        "mean_questions_clear": float(
            pd.to_numeric(clear["num_questions"], errors="coerce").mean()
        )
        if len(clear)
        else None,
    }
    if "json_schema_valid" in df.columns:
        result["json_schema_valid_rate"] = float(_to_bool(df["json_schema_valid"]).mean())
    else:
        result["json_schema_valid_rate"] = 1.0
    if "json_protocol_valid" in df.columns:
        result["json_protocol_valid_rate"] = float(_to_bool(df["json_protocol_valid"]).mean())
    else:
        result["json_protocol_valid_rate"] = result["json_schema_valid_rate"]
    return result


def resolve_path(raw: Any, manifest_path: Path) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, manifest_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--example-metrics", required=True)
    run.add_argument("--output", required=True)

    sweep = subparsers.add_parser("sweep")
    sweep.add_argument("--manifest", required=True)
    sweep.add_argument("--output-csv", required=True)
    sweep.add_argument("--selection-json", required=True)
    sweep.add_argument("--overask-penalty", type=float, default=0.50)
    sweep.add_argument("--invalid-json-penalty", type=float, default=0.25)
    sweep.add_argument(
        "--exclude-zero-parameter",
        action="store_true",
        help="Do not select the zero-valued sweep point as the causal intervention.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        summary = summarize_metrics(Path(args.example_metrics))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(manifest_path)
    if "example_metrics_path" not in manifest.columns:
        raise ValueError(f"{manifest_path} has no example_metrics_path column")

    rows: list[dict[str, Any]] = []
    for _, manifest_row in manifest.iterrows():
        metrics_path = resolve_path(manifest_row["example_metrics_path"], manifest_path)
        summary = summarize_metrics(metrics_path)
        row = manifest_row.to_dict()
        row.update(summary)
        resolved = float(summary.get("resolved_proxy_any_ambiguous") or 0.0)
        overask = float(summary.get("overasking_rate_clear") or 0.0)
        valid = float(summary.get("json_protocol_valid_rate") or 0.0)
        row["selection_utility"] = (
            resolved
            - float(args.overask_penalty) * overask
            - float(args.invalid_json_penalty) * (1.0 - valid)
        )
        rows.append(row)

    table = pd.DataFrame(rows).sort_values(
        ["selection_utility", "resolved_proxy_any_ambiguous", "overasking_rate_clear"],
        ascending=[False, False, True],
    )
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)

    selection_table = table
    if args.exclude_zero_parameter:
        if "value" not in table.columns:
            raise ValueError(
                "--exclude-zero-parameter requires a sweep manifest with a value column"
            )
        selection_table = table[pd.to_numeric(table["value"], errors="coerce").abs() > 1e-12]
        if selection_table.empty:
            raise ValueError("No non-zero sweep points are available for selection")
    selected = selection_table.iloc[0].to_dict()
    selected.update(
        {
            "selection_rule": (
                "resolved_proxy_any_ambiguous - overask_penalty * "
                "overasking_rate_clear - invalid_json_penalty * "
                "(1 - json_protocol_valid_rate)"
            ),
            "overask_penalty": float(args.overask_penalty),
            "invalid_json_penalty": float(args.invalid_json_penalty),
            "manifest": str(manifest_path),
        }
    )
    selection_json = Path(args.selection_json)
    selection_json.parent.mkdir(parents=True, exist_ok=True)
    selection_json.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    print(table.head(10).to_string(index=False))
    print("\nSelected configuration:")
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
