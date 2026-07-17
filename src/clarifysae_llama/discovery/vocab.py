from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch


class VocabFormatError(ValueError):
    pass


def _normalize_vocab_string(value: str) -> str:
    return value.lstrip().lower()


def _coerce_vocab_payload(payload) -> dict[str, list[str]]:
    if isinstance(payload, list):
        if not all(isinstance(item, str) for item in payload):
            raise VocabFormatError('List vocabulary files must contain only strings.')
        return {item: [item] for item in payload}

    if isinstance(payload, dict):
        normalized: dict[str, list[str]] = {}
        for key, values in payload.items():
            if isinstance(values, str):
                normalized[str(key)] = [values]
            elif isinstance(values, list) and all(isinstance(item, str) for item in values):
                normalized[str(key)] = values
            else:
                raise VocabFormatError(
                    'Dictionary vocabulary files must map strings to either a string or a list of strings.'
                )
        return normalized

    raise VocabFormatError('Vocabulary file must be either a JSON list or a JSON dictionary.')


def load_vocab_groups(path: str | Path, tokenizer) -> list[list[torch.Tensor]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    raw_groups = _coerce_vocab_payload(payload)

    grouped_tokens: dict[str, list[torch.Tensor]] = defaultdict(list)
    for _, variants in raw_groups.items():
        for variant in variants:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(token_ids) == 0:
                continue
            grouped_tokens[_normalize_vocab_string(variant)].append(torch.tensor(token_ids, dtype=torch.long))

    if len(grouped_tokens) == 0:
        raise VocabFormatError(f'No non-empty token sequences were produced from vocabulary file: {path}')

    return list(grouped_tokens.values())
