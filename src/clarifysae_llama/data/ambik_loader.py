from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    'id',
    'environment_full',
    'ambiguity_type',
    'ambiguous_task',
    'question',
    'answer',
}

OPTIONAL_COLUMNS = {
    'plan_for_clear_task',
}

PASSTHROUGH_COLUMNS = [
    'source_id',
    'source_split',
    'pair_variant',
]


def _ensure_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'id' in df.columns:
        return df
    if 'Unnamed: 0' in df.columns:
        return df.rename(columns={'Unnamed: 0': 'id'})
    df = df.copy()
    df.insert(0, 'id', range(len(df)))
    return df


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ''
    return str(value).strip()


def _normalize_task_text(value: Any) -> str:
    return re.sub(r'\s+', ' ', _clean_text(value)).casefold()


def load_ambik_clarification_dataset(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _ensure_id_column(df)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in dataset: {sorted(missing)}')

    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = ''

    keep_cols = [
        'id',
        'environment_full',
        'ambiguity_type',
        'ambiguous_task',
        'question',
        'answer',
        'plan_for_clear_task',
    ]
    keep_cols.extend(column for column in PASSTHROUGH_COLUMNS if column in df.columns)
    result = df[keep_cols].copy().reset_index(drop=True)

    if limit is not None:
        result = result.head(limit).copy()

    return result


# Backward-compatible alias used by existing configs / imports.
load_ambik_no_help_dataset = load_ambik_clarification_dataset


def load_ambik_selective_dataset(
    path: str | Path,
    limit_pairs: int | None = None,
    *,
    include_unambiguous_pairs: bool = True,
) -> pd.DataFrame:
    """Load AmbiK as a mixed ambiguous/clear selective-clarification set.

    Each source row contributes its ``ambiguous_task``. When
    ``include_unambiguous_pairs`` is true, the paired ``unambiguous_direct``
    instruction is also included.

    Some AmbiK rows contain textually identical ambiguous and clear variants.
    Both variants are retained for oracle-gated question-generation analysis,
    but they are marked ``classification_eligible=False`` so that a
    deterministic classifier is not evaluated against contradictory labels for
    the same input.
    """
    df = pd.read_csv(path)
    df = _ensure_id_column(df)

    required = set(REQUIRED_COLUMNS)
    if include_unambiguous_pairs:
        required.add('unambiguous_direct')
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            'Missing columns required for CLAM selective evaluation: '
            f'{sorted(missing)}. Use the paired AmbiK CSV, or set '
            'dataset.include_unambiguous_pairs=false for an ambiguous-only '
            'diagnostic run.'
        )

    if limit_pairs is not None:
        df = df.head(limit_pairs).copy()

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        source_id = _clean_text(row['id'])
        ambiguous_task = _clean_text(row['ambiguous_task'])
        clear_task = _clean_text(row.get('unambiguous_direct', ''))
        pair_texts_identical = bool(
            include_unambiguous_pairs
            and _normalize_task_text(ambiguous_task) == _normalize_task_text(clear_task)
        )
        classification_eligible = bool(
            include_unambiguous_pairs and not pair_texts_identical
        )
        source_ambiguity_type = _clean_text(row['ambiguity_type'])

        common = {
            'source_id': source_id,
            'environment_full': _clean_text(row['environment_full']),
            'source_ambiguity_type': source_ambiguity_type,
            'gold_answer': _clean_text(row.get('answer', '')),
            'gold_plan_for_clear': _clean_text(row.get('plan_for_clear_task', '')),
            'pair_texts_identical': pair_texts_identical,
            'classification_eligible': classification_eligible,
        }
        rows.append({
            **common,
            'id': f'{source_id}:ambiguous',
            'variant': 'ambiguous',
            'task': ambiguous_task,
            'ambiguity_type': source_ambiguity_type,
            'gold_ambiguous': True,
            'gold_question': _clean_text(row.get('question', '')),
        })
        if include_unambiguous_pairs:
            rows.append({
                **common,
                'id': f'{source_id}:clear',
                'variant': 'clear',
                'task': clear_task,
                'ambiguity_type': 'unambiguous_direct',
                'gold_ambiguous': False,
                'gold_question': '',
            })

    return pd.DataFrame(rows)
