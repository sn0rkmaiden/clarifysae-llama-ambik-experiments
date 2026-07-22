from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


KEY_METRICS = [
    "asking_rate_ambiguous",
    "overasking_rate_clear",
    "selective_asking_gap",
    "resolved_proxy_any_rate_ambiguous",
    "resolved_proxy_any_given_asked_ambiguous",
    "mean_best_similarity_asked_ambiguous",
    "avg_num_questions_ambiguous",
    "avg_num_questions_clear",
    "ambiguity_decision_accuracy",
    "json_schema_valid_rate",
    "json_protocol_valid_rate",
    "mean_gate_weight",
    "steering_applied_rate",
    "mean_steering_delta_norm",
]


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().eq("true")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = numeric(series).mean()
    return None if pd.isna(value) else float(value)


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "gold_ambiguous",
        "asked_question",
        "resolved_proxy_any",
        "num_questions",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Example metrics lack columns: {sorted(missing)}")

    gold = to_bool(frame["gold_ambiguous"])
    asked = to_bool(frame["asked_question"])
    resolved = to_bool(frame["resolved_proxy_any"])
    ambiguous = frame[gold]
    clear = frame[~gold]
    asked_ambiguous = gold & asked

    ask_amb = float(asked[gold].mean()) if bool(gold.any()) else None
    overask = float(asked[~gold].mean()) if bool((~gold).any()) else None
    result: dict[str, Any] = {
        "n_total": int(len(frame)),
        "n_pairs": int(frame["source_id"].astype(str).nunique())
        if "source_id" in frame.columns
        else int(len(frame) // 2),
        "n_ambiguous": int(gold.sum()),
        "n_clear": int((~gold).sum()),
        "asking_rate_ambiguous": ask_amb,
        "overasking_rate_clear": overask,
        "selective_asking_gap": None
        if ask_amb is None or overask is None
        else float(ask_amb - overask),
        "resolved_proxy_any_rate_ambiguous": float(resolved[gold].mean())
        if bool(gold.any())
        else None,
        "resolved_proxy_any_given_asked_ambiguous": float(resolved[asked_ambiguous].mean())
        if bool(asked_ambiguous.any())
        else None,
        "mean_best_similarity_asked_ambiguous": safe_mean(
            frame.loc[asked_ambiguous, "model_question_best_similarity"]
        )
        if "model_question_best_similarity" in frame.columns
        else None,
        "avg_num_questions_ambiguous": safe_mean(ambiguous["num_questions"]),
        "avg_num_questions_clear": safe_mean(clear["num_questions"]),
    }

    if "ambiguity_decision_correct" in frame.columns:
        valid = frame["ambiguity_decision_correct"].notna()
        result["ambiguity_decision_accuracy"] = (
            float(to_bool(frame.loc[valid, "ambiguity_decision_correct"]).mean())
            if bool(valid.any())
            else None
        )
    else:
        result["ambiguity_decision_accuracy"] = None

    for column, name in (
        ("json_exact_valid", "json_exact_valid_rate"),
        ("json_schema_valid", "json_schema_valid_rate"),
        ("json_protocol_valid", "json_protocol_valid_rate"),
        ("json_recoverable_parse", "json_recoverable_parse_rate"),
    ):
        result[name] = float(to_bool(frame[column]).mean()) if column in frame.columns else None

    result["mean_gate_raw_score"] = (
        safe_mean(frame["gate_raw_score"])
        if "gate_raw_score" in frame.columns else None
    )
    result["mean_gate_standardized_score"] = (
        safe_mean(frame["gate_standardized_score"])
        if "gate_standardized_score" in frame.columns else None
    )
    result["mean_gate_weight"] = (
        safe_mean(frame["gate_weight"])
        if "gate_weight" in frame.columns else None
    )
    result["steering_applied_rate"] = (
        float(to_bool(frame["steering_applied"]).mean())
        if "steering_applied" in frame.columns else None
    )
    result["mean_steering_delta_norm"] = (
        safe_mean(frame["steering_delta_norm"])
        if "steering_delta_norm" in frame.columns else None
    )
    return result


def category_summaries(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, group in frame.groupby("ambiguity_type", dropna=False):
        asked = to_bool(group["asked_question"])
        resolved = to_bool(group["resolved_proxy_any"])
        row = {
            "ambiguity_type": str(category),
            "n_examples": int(len(group)),
            "asked_rate": float(asked.mean()),
            "resolved_proxy_any_rate": float(resolved.mean()),
            "avg_num_questions": safe_mean(group["num_questions"]),
            "mean_best_similarity_asked": safe_mean(
                group.loc[asked, "model_question_best_similarity"]
            )
            if "model_question_best_similarity" in group.columns
            else None,
            "json_schema_valid_rate": float(to_bool(group["json_schema_valid"]).mean())
            if "json_schema_valid" in group.columns
            else None,
            "json_protocol_valid_rate": float(to_bool(group["json_protocol_valid"]).mean())
            if "json_protocol_valid" in group.columns
            else None,
            "mean_gate_weight": safe_mean(group["gate_weight"])
            if "gate_weight" in group.columns
            else None,
            "steering_applied_rate": float(to_bool(group["steering_applied"]).mean())
            if "steering_applied" in group.columns
            else None,
            "mean_steering_delta_norm": safe_mean(group["steering_delta_norm"])
            if "steering_delta_norm" in group.columns
            else None,
        }
        rows.append(row)
    return rows


def paired_bootstrap(
    frame: pd.DataFrame,
    *,
    samples: int,
    seed: int,
    summary_fn: Callable[[pd.DataFrame], dict[str, Any]],
) -> dict[str, tuple[float, float]]:
    if samples <= 0:
        return {}
    if "source_id" not in frame.columns:
        raise ValueError(
            "source_id is required for paired bootstrap. Re-run evaluation after applying this patch."
        )
    source_ids = frame["source_id"].astype(str).drop_duplicates().tolist()
    if not source_ids:
        return {}
    groups = {
        source_id: frame[frame["source_id"].astype(str) == source_id]
        for source_id in source_ids
    }
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {key: [] for key in KEY_METRICS}
    for _ in range(samples):
        sampled = rng.choice(source_ids, size=len(source_ids), replace=True)
        boot = pd.concat([groups[str(source_id)] for source_id in sampled], ignore_index=True)
        metrics = summary_fn(boot)
        for key in KEY_METRICS:
            value = metrics.get(key)
            if value is not None and not pd.isna(value):
                values[key].append(float(value))
    return {
        key: (float(np.quantile(items, 0.025)), float(np.quantile(items, 0.975)))
        for key, items in values.items()
        if items
    }


def resolve(path_value: Any, manifest_path: Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, manifest_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(manifest_path)
    if "arm" not in manifest.columns or "example_metrics_path" not in manifest.columns:
        raise ValueError("Manifest must contain arm and example_metrics_path columns")

    overall_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    for arm_index, (_, manifest_row) in enumerate(manifest.iterrows()):
        metrics_path = resolve(manifest_row["example_metrics_path"], manifest_path)
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        frame = pd.read_csv(metrics_path)
        point = summarize(frame)
        cis = paired_bootstrap(
            frame,
            samples=args.bootstrap_samples,
            seed=args.seed + arm_index,
            summary_fn=summarize,
        )
        row = {
            "arm": str(manifest_row["arm"]),
            "example_metrics_path": str(metrics_path),
            **point,
        }
        for key, (low, high) in cis.items():
            row[f"{key}__ci_low"] = low
            row[f"{key}__ci_high"] = high
        overall_rows.append(row)

        for category_row in category_summaries(frame):
            category_rows.append({"arm": str(manifest_row["arm"]), **category_row})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = pd.DataFrame(overall_rows)
    categories = pd.DataFrame(category_rows)
    overall.to_csv(output_dir / "summary.csv", index=False)
    categories.to_csv(output_dir / "category_summary.csv", index=False)
    payload = {
        "manifest": str(manifest_path),
        "bootstrap_samples": int(args.bootstrap_samples),
        "seed": int(args.seed),
        "overall": overall_rows,
        "categories": category_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    display = [
        column
        for column in (
            "arm",
            "n_pairs",
            "asking_rate_ambiguous",
            "overasking_rate_clear",
            "selective_asking_gap",
            "resolved_proxy_any_rate_ambiguous",
            "resolved_proxy_any_given_asked_ambiguous",
            "avg_num_questions_ambiguous",
            "avg_num_questions_clear",
            "ambiguity_decision_accuracy",
            "json_protocol_valid_rate",
            "mean_gate_weight",
            "steering_applied_rate",
            "mean_steering_delta_norm",
        )
        if column in overall.columns
    ]
    print(overall[display].to_string(index=False))
    print(f"Summary: {output_dir / 'summary.csv'}")
    print(f"Category summary: {output_dir / 'category_summary.csv'}")


if __name__ == "__main__":
    main()
