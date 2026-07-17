#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml

from clarifysae_llama.discovery.sae_utils import get_decoder_matrix
from clarifysae_llama.steering.sparsify_steerer import load_sae, move_sae_to_device_dtype


def _parse_features(raw: str) -> list[int]:
    path = Path(raw)
    if path.exists():
        if path.suffix.lower() == '.json':
            data = json.loads(path.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                if 'features' in data:
                    data = data['features']
                else:
                    data = list(data.keys())
            return [int(x) for x in data]
        text = path.read_text(encoding='utf-8')
    else:
        text = raw
    return [int(m.group(0)) for m in re.finditer(r'-?\d+', text)]


def _dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {'float32', 'fp32'}:
        return torch.float32
    if name in {'float16', 'fp16'}:
        return torch.float16
    if name in {'bfloat16', 'bf16'}:
        return torch.bfloat16
    raise ValueError(f'Unsupported dtype: {name}')


def _connected_components(sim: torch.Tensor, features: list[int], threshold: float) -> list[list[int]]:
    n = len(features)
    seen = [False] * n
    clusters: list[list[int]] = []
    adjacency = sim >= float(threshold)
    adjacency.fill_diagonal_(True)

    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(features[current])
            neighbors = adjacency[current].nonzero(as_tuple=False).flatten().tolist()
            for nb in neighbors:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        clusters.append(sorted(component))
    return sorted(clusters, key=lambda xs: (-len(xs), xs[0]))


def _write_yaml_snippet(path: Path, clusters: list[list[int]], label_prefix: str) -> None:
    feature_sets = [
        {'label': f'{label_prefix}{idx}', 'features': cluster}
        for idx, cluster in enumerate(clusters)
    ]
    payload = {'feature_sets': feature_sets}
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build ClarQ multi-feature sweep feature_sets by clustering SAE decoder directions.'
    )
    parser.add_argument('--features', required=True, help='Comma/space-separated feature ids or a txt/json file.')
    parser.add_argument('--threshold', type=float, default=0.60, help='Cosine similarity threshold for connected components.')
    parser.add_argument('--min_size', type=int, default=2, help='Drop clusters smaller than this size.')
    parser.add_argument('--label_prefix', default='cluster')
    parser.add_argument('--out_yaml', required=True)
    parser.add_argument('--out_json', default=None)

    parser.add_argument('--loader', default='saelens', choices=['sparsify', 'dictionary_learning', 'saelens'])
    parser.add_argument('--sae_repo', required=True)
    parser.add_argument('--hookpoint', required=True)
    parser.add_argument('--sae_file', default=None)
    parser.add_argument('--sae_id', default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--dtype', default='float32')
    args = parser.parse_args()

    features = sorted(set(_parse_features(args.features)))
    if not features:
        raise ValueError('No features were parsed from --features.')

    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    sae = load_sae(
        loader=args.loader,
        sae_repo=args.sae_repo,
        hookpoint=args.hookpoint,
        sae_file=args.sae_file,
        sae_id=args.sae_id,
        device=device,
        dtype=dtype,
    )
    sae = move_sae_to_device_dtype(sae, device=device, dtype=dtype)
    decoder = get_decoder_matrix(sae).to(device=device, dtype=torch.float32)

    if max(features) >= decoder.shape[0] or min(features) < 0:
        raise ValueError(f'Feature id out of bounds for decoder with {decoder.shape[0]} rows.')

    vectors = decoder[torch.tensor(features, device=device, dtype=torch.long)]
    vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sim = vectors @ vectors.T

    clusters = _connected_components(sim.detach().cpu(), features, args.threshold)
    clusters = [cluster for cluster in clusters if len(cluster) >= int(args.min_size)]
    if not clusters:
        raise ValueError('No clusters passed min_size. Lower --threshold or --min_size.')

    out_yaml = Path(args.out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    _write_yaml_snippet(out_yaml, clusters, args.label_prefix)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            'threshold': args.threshold,
            'min_size': args.min_size,
            'features': features,
            'clusters': clusters,
            'similarity': sim.detach().cpu().tolist(),
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print(f'Wrote {len(clusters)} feature sets to {out_yaml}')


if __name__ == '__main__':
    main()
