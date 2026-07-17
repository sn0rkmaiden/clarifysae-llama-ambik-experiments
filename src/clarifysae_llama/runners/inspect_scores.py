from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from clarifysae_llama.config import load_yaml
from clarifysae_llama.utils.io import ensure_dir


def _load_feature_score_csv(path: str | Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {col: f'{prefix}_{col}' for col in df.columns if col != 'feature_idx'}
    return df.rename(columns=rename_map)


def _load_output_score_csv(path: str | Path, prefix: str = 'output') -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {col: f'{prefix}_{col}' for col in df.columns if col != 'feature_idx'}
    return df.rename(columns=rename_map)


def _optional_top_tokens(row: pd.Series) -> str:
    for col in row.index:
        if col.endswith('top_tokens'):
            return str(row[col])
    return ''


def run_inspection(config: dict[str, Any]) -> None:
    inspect_cfg = config['inspection']
    dataframes: list[pd.DataFrame] = []

    for item in inspect_cfg.get('feature_score_tables', []):
        dataframes.append(_load_feature_score_csv(item['path'], item['name']))

    if inspect_cfg.get('output_score_table'):
        out_cfg = inspect_cfg['output_score_table']
        dataframes.append(_load_output_score_csv(out_cfg['path'], out_cfg.get('name', 'output')))

    if not dataframes:
        raise ValueError('No score tables were provided for inspection.')

    merged = dataframes[0]
    for df in dataframes[1:]:
        merged = merged.merge(df, on='feature_idx', how='outer')

    sort_by = inspect_cfg.get('sort_by', 'feature_idx')
    ascending = bool(inspect_cfg.get('ascending', False))
    merged = merged.sort_values(sort_by, ascending=ascending)

    if inspect_cfg.get('feature_indices'):
        requested = {int(x) for x in inspect_cfg['feature_indices']}
        merged = merged[merged['feature_idx'].isin(requested)]

    if inspect_cfg.get('min_values'):
        for col, threshold in inspect_cfg['min_values'].items():
            merged = merged[merged[col] >= threshold]

    limit = inspect_cfg.get('top_n')
    if limit is not None:
        merged = merged.head(int(limit))

    output_dir = ensure_dir(inspect_cfg.get('output_dir', 'outputs/discovery/inspection'))
    merged.to_csv(output_dir / 'merged_scores.csv', index=False)
    (output_dir / 'inspection_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    printable = merged.copy()
    printable['top_tokens_preview'] = printable.apply(_optional_top_tokens, axis=1)
    with pd.option_context('display.max_columns', None, 'display.width', 200):
        print(printable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_inspection(load_yaml(args.config))
