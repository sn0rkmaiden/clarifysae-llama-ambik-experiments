from __future__ import annotations

"""Approximate a dense concept vector with a small signed set of SAE features."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from clarifysae_llama.config import load_yaml
from clarifysae_llama.discovery.concept_vectors import cosine_similarity, l2_normalize
from clarifysae_llama.discovery.sae_utils import get_decoder_matrix
from clarifysae_llama.steering.sparsify_steerer import load_sae
from clarifysae_llama.utils.io import ensure_dir


def _load_vector(path: str, key: str) -> tuple[torch.Tensor, dict[str, Any]]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    records = bundle.get("vectors", bundle)
    if key not in records:
        raise KeyError(f"Vector key {key!r} not found; available={sorted(records.keys())}")
    record = records[key]
    if isinstance(record, torch.Tensor):
        return record.float().flatten(), {}
    return record["vector"].float().flatten(), dict(record)


def run(config: dict[str, Any]) -> None:
    cfg = config["sae_projection"]
    vector, vector_record = _load_vector(cfg["vector_path"], cfg["vector_key"])
    device_name = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_name)
    dtype_name = str(cfg.get("dtype", "float32"))
    dtype = getattr(torch, dtype_name)

    sae = load_sae(
        loader=cfg.get("loader", "sparsify"),
        sae_repo=cfg["sae_repo"],
        hookpoint=cfg["hookpoint"],
        sae_file=cfg.get("sae_file"),
        sae_id=cfg.get("sae_id"),
        device=device,
        dtype=dtype,
    )
    decoder = get_decoder_matrix(sae).detach().float().cpu()
    if decoder.shape[1] != vector.numel():
        raise ValueError(
            f"SAE decoder dimension {decoder.shape[1]} does not match vector dimension {vector.numel()}"
        )

    vector_unit = l2_normalize(vector)
    decoder_norms = decoder.norm(dim=1).clamp_min(1e-8)
    cosines = (decoder @ vector_unit) / decoder_norms
    candidate_k = min(int(cfg.get("candidate_k", 256)), decoder.shape[0])
    selected_k = min(int(cfg.get("selected_k", 16)), candidate_k)
    candidate_indices = torch.topk(cosines.abs(), k=candidate_k).indices
    candidate_decoder = decoder[candidate_indices]

    # Solve D^T c ~= v over a moderately sized candidate pool, then retain the
    # largest signed coefficients and refit on that sparse support.
    coeffs = torch.linalg.lstsq(candidate_decoder.T, vector).solution
    keep_local = torch.topk(coeffs.abs(), k=selected_k).indices
    feature_indices = candidate_indices[keep_local]
    sparse_decoder = decoder[feature_indices]
    sparse_coeffs = torch.linalg.lstsq(sparse_decoder.T, vector).solution
    reconstruction = sparse_decoder.T @ sparse_coeffs

    normalize_weights = bool(cfg.get("normalize_weights", True))
    weights = sparse_coeffs.clone()
    strength = 1.0
    if normalize_weights:
        scale = float(weights.abs().max().clamp_min(1e-8))
        weights = weights / scale
        strength = scale

    rows = []
    for idx, weight in zip(feature_indices.tolist(), weights.tolist()):
        rows.append({
            "feature_idx": int(idx),
            "weight": float(weight),
            "decoder_cosine": float(cosines[int(idx)]),
            "abs_decoder_cosine": abs(float(cosines[int(idx)])),
        })
    rows.sort(key=lambda row: abs(row["weight"]), reverse=True)

    output_dir = ensure_dir(Path(cfg.get("output_dir", "outputs/sae_projection")))
    csv_path = output_dir / "sae_feature_projection.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    steering_snippet = {
        "steering": {
            "enabled": True,
            "method": "sae",
            "loader": cfg.get("loader", "sparsify"),
            "sae_repo": cfg["sae_repo"],
            "sae_file": cfg.get("sae_file"),
            "sae_id": cfg.get("sae_id"),
            "hookpoint": cfg["hookpoint"],
            "module_path": cfg.get("module_path"),
            "feature_indices": [int(row["feature_idx"]) for row in rows],
            "feature_weights": [float(row["weight"]) for row in rows],
            "strength": float(strength),
            "mode": "decoder_vector",
            "max_act": 1.0,
            "normalize_each": False,
            "apply_to": "last_position",
            "steer_generated_tokens_only": True,
        }
    }
    yaml_path = output_dir / "steering_snippet.yaml"
    yaml_path.write_text(yaml.safe_dump(steering_snippet, sort_keys=False), encoding="utf-8")

    metrics = {
        "vector_key": cfg["vector_key"],
        "hookpoint": cfg["hookpoint"],
        "candidate_k": candidate_k,
        "selected_k": selected_k,
        "reconstruction_cosine": cosine_similarity(vector, reconstruction),
        "relative_l2_error": float((vector - reconstruction).norm() / vector.norm().clamp_min(1e-8)),
        "dense_vector_metadata": {k: v for k, v in vector_record.items() if k != "vector" and not isinstance(v, torch.Tensor)},
        "csv_path": str(csv_path),
        "yaml_path": str(yaml_path),
    }
    (output_dir / "projection_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(load_yaml(args.config))
