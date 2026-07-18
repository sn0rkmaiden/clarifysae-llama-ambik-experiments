from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "environment_full",
    "ambiguity_type",
    "ambiguous_task",
    "unambiguous_direct",
    "question",
    "answer",
}


def split_for_id(value: str, *, train: float, dev: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    u = int.from_bytes(digest[:8], "big") / float(2**64)
    if u < train:
        return "train"
    if u < train + dev:
        return "dev"
    return "test"


def validate(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    result = df.copy()
    if "id" not in result.columns:
        result.insert(0, "id", range(len(result)))
    if "plan_for_clear_task" not in result.columns:
        result["plan_for_clear_task"] = ""
    return result


def paired_rows(df: pd.DataFrame, *, seed: int, add_internal_split: bool) -> pd.DataFrame:
    rows: list[dict] = []
    for position, (_, source_row) in enumerate(df.iterrows()):
        source = source_row.to_dict()
        source_id = str(source.get("id", position))
        source_split = (
            split_for_id(source_id, train=0.70, dev=0.15, seed=seed)
            if add_internal_split
            else "external"
        )

        ambiguous = dict(source)
        ambiguous.update(
            {
                "id": 2 * position,
                "source_id": source_id,
                "source_split": source_split,
                "pair_variant": "ambiguous",
            }
        )
        rows.append(ambiguous)

        clear = dict(source)
        clear.update(
            {
                "id": 2 * position + 1,
                "source_id": source_id,
                "source_split": source_split,
                "pair_variant": "clear",
                "ambiguous_task": source["unambiguous_direct"],
                "ambiguity_type": "unambiguous_direct",
                "question": "",
            }
        )
        rows.append(clear)

    return pd.DataFrame(rows)


def write_pair_files(
    source_path: Path,
    destination_path: Path,
    *,
    seed: int,
    add_internal_split: bool,
) -> dict:
    source = validate(pd.read_csv(source_path), source_path)
    paired = paired_rows(source, seed=seed, add_internal_split=add_internal_split)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(destination_path, index=False)

    metadata = {
        "source": str(source_path),
        "destination": str(destination_path),
        "source_examples": int(len(source)),
        "paired_rows": int(len(paired)),
        "ambiguous_rows": int((paired["pair_variant"] == "ambiguous").sum()),
        "clear_rows": int((paired["pair_variant"] == "clear").sum()),
    }

    if add_internal_split:
        split_counts = paired.groupby("source_split").size().to_dict()
        metadata["paired_rows_by_internal_split"] = {
            str(key): int(value) for key, value in split_counts.items()
        }
        internal_test = paired[paired["source_split"] == "test"].copy()
        internal_test_path = destination_path.with_name(
            destination_path.stem.replace("_paired", "_internal_test_paired")
            + destination_path.suffix
        )
        internal_test.to_csv(internal_test_path, index=False)
        metadata["internal_test_destination"] = str(internal_test_path)
        metadata["internal_test_rows"] = int(len(internal_test))

    destination_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calib",
        default="data/raw/ambik/ambik_calib_100.csv",
    )
    parser.add_argument(
        "--test",
        default="data/raw/ambik/ambik_test_400.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calib = Path(args.calib)
    test = Path(args.test)
    if not calib.exists():
        raise FileNotFoundError(
            f"Missing {calib}. Copy the AmbiK calibration CSV to this location."
        )
    if not test.exists():
        raise FileNotFoundError(
            f"Missing {test}. Copy the AmbiK test CSV to this location."
        )

    reports = [
        write_pair_files(
            calib,
            calib.with_name("ambik_calib_100_paired.csv"),
            seed=args.seed,
            add_internal_split=False,
        ),
        write_pair_files(
            test,
            test.with_name("ambik_test_400_paired.csv"),
            seed=args.seed,
            add_internal_split=True,
        ),
    ]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
