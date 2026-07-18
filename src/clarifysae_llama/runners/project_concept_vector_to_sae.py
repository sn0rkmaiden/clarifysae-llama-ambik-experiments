from __future__ import annotations

"""Approximate a dense concept vector with nested signed SAE feature sets."""

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


def _projection_for_k(
    *,
    vector: torch.Tensor,
    decoder: torch.Tensor,
    cosines: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_coeffs: torch.Tensor,
    k: int,
    normalize_weights: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    keep_local = torch.topk(candidate_coeffs.abs(), k=k).indices
    feature_indices = candidate_indices[keep_local]
    sparse_decoder = decoder[feature_indices]
    sparse_coeffs = torch.linalg.lstsq(sparse_decoder.T, vector).solution
    reconstruction = sparse_decoder.T @ sparse_coeffs

    weights = sparse_coeffs.clone()
    strength = 1.0
    if normalize_weights:
        strength = float(weights.abs().max().clamp_min(1e-8))
        weights = weights / strength

    rows = [
        {
            "feature_idx": int(idx),
            "weight": float(weight),
            "raw_coefficient": float(raw_coefficient),
            "decoder_cosine": float(cosines[int(idx)]),
            "abs_decoder_cosine": abs(float(cosines[int(idx)])),
        }
        for idx, weight, raw_coefficient in zip(
            feature_indices.tolist(), weights.tolist(), sparse_coeffs.tolist()
        )
    ]
    rows.sort(key=lambda row: abs(row["raw_coefficient"]), reverse=True)

    metrics = {
        "selected_k": int(k),
        "reconstruction_cosine": cosine_similarity(vector, reconstruction),
        "relative_l2_error": float(
            (vector - reconstruction).norm() / vector.norm().clamp_min(1e-8)
        ),
        "coefficient_l1": float(sparse_coeffs.abs().sum()),
        "coefficient_l2": float(sparse_coeffs.norm()),
        "steering_strength": float(strength),
    }
    return rows, metrics, {"indices": feature_indices, "coefficients": sparse_coeffs}


def _steering_snippet(cfg: dict[str, Any], rows: list[dict[str, Any]], strength: float) -> dict[str, Any]:
    return {
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
    candidate_indices = torch.topk(cosines.abs(), k=candidate_k).indices
    candidate_decoder = decoder[candidate_indices]

    # Fit once over a broad candidate pool, rank by coefficient magnitude, then
    # refit nested sparse supports. This exposes whether one feature is enough or
    # whether the behavior is better represented by a signed feature subspace.
    candidate_coeffs = torch.linalg.lstsq(candidate_decoder.T, vector).solution
    requested_ks = cfg.get("selected_ks")
    if requested_ks is None:
        requested_ks = [1, 2, 4, 8, int(cfg.get("selected_k", 16)), 32]
    ks = sorted({min(max(int(k), 1), candidate_k) for k in requested_ks})
    primary_k = min(max(int(cfg.get("selected_k", 16)), 1), candidate_k)
    if primary_k not in ks:
        ks.append(primary_k)
        ks.sort()

    normalize_weights = bool(cfg.get("normalize_weights", True))
    output_dir = ensure_dir(Path(cfg.get("output_dir", "outputs/sae_projection")))
    curve: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] | None = None
    primary_metrics: dict[str, Any] | None = None

    for k in ks:
        rows, metrics, _raw = _projection_for_k(
            vector=vector,
            decoder=decoder,
            cosines=cosines,
            candidate_indices=candidate_indices,
            candidate_coeffs=candidate_coeffs,
            k=k,
            normalize_weights=normalize_weights,
        )
        csv_path = output_dir / f"sae_feature_projection_k{k}.csv"
        yaml_path = output_dir / f"steering_snippet_k{k}.yaml"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        yaml_path.write_text(
            yaml.safe_dump(
                _steering_snippet(cfg, rows, float(metrics["steering_strength"])),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        metrics.update({"csv_path": str(csv_path), "yaml_path": str(yaml_path)})
        curve.append(metrics)
        if k == primary_k:
            primary_rows = rows
            primary_metrics = metrics

    assert primary_rows is not None and primary_metrics is not None
    # Backwards-compatible aliases for the primary k.
    pd.DataFrame(primary_rows).to_csv(output_dir / "sae_feature_projection.csv", index=False)
    (output_dir / "steering_snippet.yaml").write_text(
        yaml.safe_dump(
            _steering_snippet(cfg, primary_rows, float(primary_metrics["steering_strength"])),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(curve).to_csv(output_dir / "projection_curve.csv", index=False)

    metrics = {
        "vector_key": cfg["vector_key"],
        "hookpoint": cfg["hookpoint"],
        "candidate_k": candidate_k,
        "primary_selected_k": primary_k,
        "projection_curve": curve,
        "dense_vector_metadata": {
            k: v
            for k, v in vector_record.items()
            if k not in {"vector", "raw_vector"} and not isinstance(v, torch.Tensor)
        },
    }
    (output_dir / "projection_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(load_yaml(args.config))
