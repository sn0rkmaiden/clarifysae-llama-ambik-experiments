from __future__ import annotations

from pathlib import Path

import pandas as pd

from clarifysae_llama.utils.io import write_csv


def save_metric_tables(
    example_metrics: pd.DataFrame,
    aggregate_metrics: pd.DataFrame,
    category_metrics: pd.DataFrame,
    output_root: Path,
) -> None:
    write_csv(output_root / 'metrics' / 'example_metrics.csv', example_metrics)
    write_csv(output_root / 'tables' / 'aggregate_metrics.csv', aggregate_metrics)
    write_csv(output_root / 'tables' / 'category_metrics.csv', category_metrics)
