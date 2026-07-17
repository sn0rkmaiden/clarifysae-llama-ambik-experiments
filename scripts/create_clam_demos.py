#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ("environment_full", "question", "answer")
REQUIRED_COLUMNS = {
    "environment_full",
    "ambiguous_task",
    "unambiguous_direct",
    "ambiguity_type",
    "question",
    "answer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create balanced CLAM few-shot demonstrations and an optional "
            "held-out threshold-calibration split from AmbiK without overlap "
            "with the current evaluation or reserved test examples."
        )
    )
    parser.add_argument("--calib", required=True, type=Path)
    parser.add_argument("--test400", required=True, type=Path)
    parser.add_argument("--test900", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--pairs-per-category",
        type=int,
        default=1,
        help="Number of ambiguous/clear demo pairs per ambiguity type (default: 1).",
    )
    parser.add_argument(
        "--threshold-output",
        type=Path,
        default=None,
        help="Optional CSV path for a disjoint threshold-calibration split.",
    )
    parser.add_argument(
        "--threshold-pairs",
        type=int,
        default=50,
        help="Number of source pairs in the threshold-calibration split (default: 50).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def check_columns(df: pd.DataFrame, name: str) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def normalize(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def add_key(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    parts = [normalize(result[col]) for col in KEY_COLUMNS]
    result["_example_key"] = parts[0] + "\u241f" + parts[1] + "\u241f" + parts[2]
    return result


def source_label(row: pd.Series) -> str:
    for column in ("id", "Unnamed: 0", "Unnamed: 0.1"):
        if column in row.index and pd.notna(row[column]):
            return str(row[column])
    return str(row.name)


def balanced_sample_by_category(
    pool: pd.DataFrame,
    *,
    n: int,
    seed: int,
) -> pd.DataFrame:
    if n < 1:
        raise ValueError("Sample size must be at least 1")
    if len(pool) < n:
        raise ValueError(f"Need {n} rows but only {len(pool)} are available")

    categories = sorted(pool["ambiguity_type"].dropna().astype(str).unique())
    if not categories:
        raise ValueError("No ambiguity categories are available")

    base, remainder = divmod(n, len(categories))
    selected_parts: list[pd.DataFrame] = []
    used_indices: set[int] = set()

    for category_index, category in enumerate(categories):
        target = base + (1 if category_index < remainder else 0)
        if target == 0:
            continue
        group = pool[pool["ambiguity_type"].astype(str) == category]
        if len(group) < target:
            raise ValueError(
                f"Not enough rows for category {category!r}: need {target}, found {len(group)}"
            )
        sampled = group.sample(
            n=target,
            random_state=seed + category_index,
            replace=False,
        )
        selected_parts.append(sampled)
        used_indices.update(int(index) for index in sampled.index)

    selected = pd.concat(selected_parts, axis=0)
    if len(selected) < n:
        remainder_pool = pool[~pool.index.isin(used_indices)]
        extra = remainder_pool.sample(
            n=n - len(selected),
            random_state=seed + 10_000,
            replace=False,
        )
        selected = pd.concat([selected, extra], axis=0)

    return selected.sample(frac=1.0, random_state=seed).copy()


def main() -> None:
    args = parse_args()
    if args.pairs_per_category < 1:
        raise ValueError("--pairs-per-category must be at least 1")
    if args.threshold_pairs < 1:
        raise ValueError("--threshold-pairs must be at least 1")

    calib = pd.read_csv(args.calib)
    test400 = pd.read_csv(args.test400)
    test900 = pd.read_csv(args.test900)

    for name, df in (
        ("calib", calib),
        ("test400", test400),
        ("test900", test900),
    ):
        check_columns(df, name)

    calib = add_key(calib)
    test400 = add_key(test400)
    test900 = add_key(test900)

    calib_keys = set(calib["_example_key"])
    test400_keys = set(test400["_example_key"])
    test900_keys = set(test900["_example_key"])

    calib_test400_overlap = calib_keys & test400_keys
    calib_test900_overlap = calib_keys & test900_keys
    missing_from_test900 = test400_keys - test900_keys

    print(f"calib rows: {len(calib)}")
    print(f"test400 rows: {len(test400)}")
    print(f"test900 rows: {len(test900)}")
    print(f"calib ∩ test400: {len(calib_test400_overlap)}")
    print(f"calib ∩ test900: {len(calib_test900_overlap)}")
    print(f"test400 missing from test900: {len(missing_from_test900)}")

    if calib_test400_overlap or calib_test900_overlap:
        raise ValueError("Calibration examples overlap another split.")
    if missing_from_test900:
        raise ValueError("test400 is not a subset of test900 under the stable key.")

    excluded_keys = calib_keys | test400_keys
    pool = test900[~test900["_example_key"].isin(excluded_keys)].copy()

    # A source row can only form a useful pair when the ambiguous and clear
    # instructions are both present and actually differ.
    for column in REQUIRED_COLUMNS:
        pool = pool[normalize(pool[column]) != ""]

    pool = pool[
        normalize(pool["ambiguous_task"]).str.casefold()
        != normalize(pool["unambiguous_direct"]).str.casefold()
    ].copy()

    print(f"Eligible source pool: {len(pool)} rows")
    print("Eligible rows by ambiguity type:")
    print(pool["ambiguity_type"].value_counts().to_string())

    selected_parts = []
    for category in sorted(pool["ambiguity_type"].dropna().astype(str).unique()):
        group = pool[pool["ambiguity_type"].astype(str) == category]
        if len(group) < args.pairs_per_category:
            raise ValueError(
                f"Not enough eligible rows for category {category!r}: "
                f"need {args.pairs_per_category}, found {len(group)}"
            )
        selected_parts.append(
            group.sample(
                n=args.pairs_per_category,
                random_state=args.seed,
                replace=False,
            )
        )

    selected = pd.concat(selected_parts, ignore_index=False)
    demonstrations: list[dict[str, str]] = []

    for _, row in selected.iterrows():
        environment = str(row["environment_full"]).strip()
        ambiguous_task = str(row["ambiguous_task"]).strip()
        clear_task = str(row["unambiguous_direct"]).strip()
        question = str(row["question"]).strip()
        demonstrations.append(
            {
                "environment": environment,
                "task": ambiguous_task,
                "label": "AMBIGUOUS",
                "question": question,
            }
        )
        demonstrations.append(
            {
                "environment": environment,
                "task": clear_task,
                "label": "CLEAR",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(demonstrations, file, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(demonstrations)} demonstrations to: {args.output}")
    print("Selected demonstration source rows:")
    for _, row in selected.iterrows():
        print(
            f"  source={source_label(row)}, "
            f"type={row['ambiguity_type']}, "
            f"question={row['question']}"
        )

    if args.threshold_output is not None:
        threshold_pool = pool[~pool.index.isin(selected.index)].copy()
        threshold_rows = balanced_sample_by_category(
            threshold_pool,
            n=args.threshold_pairs,
            seed=args.seed + 1_000,
        )
        threshold_rows = threshold_rows.drop(columns=['_example_key'], errors='ignore')
        args.threshold_output.parent.mkdir(parents=True, exist_ok=True)
        threshold_rows.to_csv(args.threshold_output, index=False)
        print(
            f"\nSaved {len(threshold_rows)} disjoint threshold-calibration "
            f"source pairs to: {args.threshold_output}"
        )
        print("Threshold rows by ambiguity type:")
        print(threshold_rows['ambiguity_type'].value_counts().to_string())


if __name__ == "__main__":
    main()
