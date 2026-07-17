#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


STABLE_COLUMNS = ('environment_full', 'question', 'answer')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Validate that CLAM demonstrations are disjoint from AmbiK evaluation and threshold splits.'
    )
    parser.add_argument('--calib', required=True, type=Path)
    parser.add_argument('--test400', required=True, type=Path)
    parser.add_argument('--test900', required=True, type=Path)
    parser.add_argument('--threshold', required=True, type=Path)
    parser.add_argument('--classification-demos', required=True, type=Path)
    parser.add_argument('--question-demos', required=True, type=Path)
    return parser.parse_args()


def normalize(value: Any) -> str:
    return re.sub(r'\s+', ' ', str(value or '').strip()).casefold()


def stable_key(row: pd.Series) -> tuple[str, str, str]:
    return tuple(normalize(row[column]) for column in STABLE_COLUMNS)  # type: ignore[return-value]


def prompt_key(environment: Any, task: Any, question: Any) -> tuple[str, str, str]:
    return normalize(environment), normalize(task), normalize(question)


def load_json_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, list):
        raise ValueError(f'{path} must contain a JSON list')
    return [dict(item) for item in payload]


def resolve_demo_rows(
    demos: list[dict[str, Any]],
    test900: pd.DataFrame,
    *,
    name: str,
) -> pd.DataFrame:
    index: dict[tuple[str, str, str], list[int]] = {}
    for row_index, row in test900.iterrows():
        key = prompt_key(row['environment_full'], row['ambiguous_task'], row['question'])
        index.setdefault(key, []).append(int(row_index))

    resolved_indices: list[int] = []
    for demo_index, demo in enumerate(demos):
        if str(demo.get('label', '')).strip().upper() != 'AMBIGUOUS':
            continue
        key = prompt_key(demo.get('environment'), demo.get('task'), demo.get('question'))
        matches = index.get(key, [])
        if len(matches) != 1:
            raise ValueError(
                f'{name} demonstration {demo_index} resolves to {len(matches)} test900 rows; expected exactly one.'
            )
        resolved_indices.append(matches[0])

    if not resolved_indices:
        raise ValueError(f'{name} has no ambiguous source demonstrations')
    if len(set(resolved_indices)) != len(resolved_indices):
        raise ValueError(f'{name} contains duplicate ambiguous source rows')
    return test900.loc[resolved_indices].copy()


def main() -> None:
    args = parse_args()
    calib = pd.read_csv(args.calib)
    test400 = pd.read_csv(args.test400)
    test900 = pd.read_csv(args.test900)
    threshold = pd.read_csv(args.threshold)
    classification_demos = load_json_list(args.classification_demos)
    question_demos = load_json_list(args.question_demos)

    for name, frame in (
        ('calib', calib),
        ('test400', test400),
        ('test900', test900),
        ('threshold', threshold),
    ):
        missing = set(STABLE_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f'{name} is missing columns: {sorted(missing)}')

    classification_rows = resolve_demo_rows(
        classification_demos,
        test900,
        name='classification demonstrations',
    )
    question_rows = resolve_demo_rows(
        question_demos,
        test900,
        name='question demonstrations',
    )

    forbidden = {
        'calib': {stable_key(row) for _, row in calib.iterrows()},
        'test400': {stable_key(row) for _, row in test400.iterrows()},
        'threshold': {stable_key(row) for _, row in threshold.iterrows()},
    }
    classification_keys = {stable_key(row) for _, row in classification_rows.iterrows()}
    question_keys = {stable_key(row) for _, row in question_rows.iterrows()}

    for demo_name, keys in (
        ('classification', classification_keys),
        ('question', question_keys),
    ):
        for split_name, split_keys in forbidden.items():
            overlap = keys & split_keys
            if overlap:
                raise ValueError(
                    f'{demo_name} demonstrations overlap {split_name}: {len(overlap)} source rows'
                )

    shared = classification_keys & question_keys
    if shared:
        raise ValueError(
            'Classification and question demonstrations overlap: '
            f'{len(shared)} source rows'
        )

    print('CLAM demonstration split validation passed.')
    print(f'classification source rows: {len(classification_rows)}')
    print(classification_rows['ambiguity_type'].value_counts().to_string())
    print(f'question source rows: {len(question_rows)}')
    print(question_rows['ambiguity_type'].value_counts().to_string())
    print('overlap with calib/test400/threshold: 0')


if __name__ == '__main__':
    main()
