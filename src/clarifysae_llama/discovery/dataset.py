from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
import torch
from datasets import Dataset, load_dataset


SUPPORTED_SOURCES = {'huggingface', 'csv', 'json', 'text', 'parquet'}


def _load_text_dataset(dataset_cfg: dict[str, Any]) -> Dataset:
    source = dataset_cfg.get('source', 'huggingface')
    if source not in SUPPORTED_SOURCES:
        raise ValueError(f'Unsupported dataset source: {source}. Expected one of {sorted(SUPPORTED_SOURCES)}')

    if source == 'huggingface':
        name = dataset_cfg['name']
        subset = dataset_cfg.get('subset')
        split = dataset_cfg.get('split', 'train')
        streaming = bool(dataset_cfg.get('streaming', False))
        if subset is not None:
            dataset = load_dataset(name, subset, split=split, streaming=streaming)
        else:
            dataset = load_dataset(name, split=split, streaming=streaming)
        if streaming:
            n_texts = dataset_cfg.get('n_texts')
            if n_texts is None:
                raise ValueError('For streaming datasets, dataset.n_texts must be set to a finite integer.')
            rows = []
            for idx, row in enumerate(dataset):
                if idx >= int(n_texts):
                    break
                rows.append(row)
            return Dataset.from_list(rows)
        return dataset

    path = str(dataset_cfg['path'])

    if source == 'csv':
        loaded = load_dataset('csv', data_files=path)
        split = dataset_cfg.get('split', 'train')
        return loaded[split]

    elif source == 'json':
        loaded = load_dataset('json', data_files=path)
        split = dataset_cfg.get('split', 'train')
        return loaded[split]

    elif source == 'parquet':
        df = pd.read_parquet(path)
        return Dataset.from_pandas(df, preserve_index=False)

    else:
        loaded = load_dataset('text', data_files=path)
        split = dataset_cfg.get('split', 'train')
        return loaded[split]


def _normalize_token_ids(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.flatten().tolist()]
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for x in value:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return out
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith('[') and stripped.endswith(']'):
            pieces = [piece.strip() for piece in stripped[1:-1].split(',')]
            out: list[int] = []
            for piece in pieces:
                if not piece:
                    continue
                try:
                    out.append(int(piece))
                except ValueError:
                    continue
            return out
    return []


def load_token_chunks(dataset_cfg: dict[str, Any], tokenizer, tokenization_cfg: dict[str, Any]) -> list[torch.Tensor]:
    dataset = _load_text_dataset(dataset_cfg)
    text_column = dataset_cfg.get('text_column', 'text')
    max_texts = dataset_cfg.get('n_texts')
    max_length = int(tokenization_cfg.get('max_length', 256))
    stride = int(tokenization_cfg.get('stride', max_length))
    add_special_tokens = bool(tokenization_cfg.get('add_special_tokens', False))
    kind = str(tokenization_cfg.get('kind', 'text')).lower()

    if max_length <= 0:
        raise ValueError('tokenization.max_length must be positive.')
    if stride <= 0:
        raise ValueError('tokenization.stride must be positive.')

    chunks: list[torch.Tensor] = []
    for idx, row in enumerate(dataset):
        if max_texts is not None and idx >= int(max_texts):
            break

        if text_column not in row:
            raise KeyError(f"Column '{text_column}' not found in dataset row. Available keys: {sorted(row.keys())}")

        if kind == 'pretokenized':
            token_ids = _normalize_token_ids(row[text_column])
        else:
            text_value = row[text_column]
            if not isinstance(text_value, str):
                text_value = str(text_value)
            text_value = text_value.strip()
            if not text_value:
                continue
            token_ids = tokenizer.encode(text_value, add_special_tokens=add_special_tokens)

        if len(token_ids) == 0:
            continue

        start = 0
        while start < len(token_ids):
            piece = token_ids[start:start + max_length]
            if len(piece) == 0:
                break
            chunks.append(torch.tensor(piece, dtype=torch.long))
            if len(piece) < max_length:
                break
            start += stride

    if len(chunks) == 0:
        raise ValueError('No token chunks were produced from the configured dataset.')

    return chunks
