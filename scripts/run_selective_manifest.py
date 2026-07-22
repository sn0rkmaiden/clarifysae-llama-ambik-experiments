from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(manifest_path)
    required = {"arm", "config_path", "example_metrics_path", "aggregate_metrics_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{manifest_path} lacks columns: {sorted(missing)}")

    for _, row in manifest.iterrows():
        config_path = Path(str(row["config_path"]))
        example_path = Path(str(row["example_metrics_path"]))
        aggregate_path = Path(str(row["aggregate_metrics_path"]))
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        if not args.no_resume and example_path.exists() and aggregate_path.exists():
            print(f"REUSE {row['arm']}: {example_path}")
            continue
        print(f"RUN   {row['arm']}: {config_path}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "clarifysae_llama.runners.run_eval",
                "--config",
                str(config_path),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
