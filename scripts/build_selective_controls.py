from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} does not contain a YAML mapping")
    return value


def save_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def selected_value(payload: dict[str, Any], fallback: float) -> float:
    for key in ("value", "selected_strength", "strength", "threshold"):
        if key in payload and payload[key] is not None:
            return float(payload[key])
    return float(fallback)


def load_vector_record(bundle: dict[str, Any], key: str) -> dict[str, Any]:
    records = bundle.get("vectors", bundle)
    if key not in records:
        raise KeyError(f"Vector key {key!r} is absent; available={sorted(records)}")
    record = records[key]
    if isinstance(record, torch.Tensor):
        return {"vector": record}
    if not isinstance(record, dict) or "vector" not in record:
        raise TypeError(f"Vector record {key!r} must contain a tensor under 'vector'")
    return record


def make_orthogonal_random_bundle(
    *,
    vector_path: Path,
    gate_vector_path: Path,
    ask_key: str,
    gate_key: str,
    output_path: Path,
    seed: int,
) -> str:
    bundle = torch.load(vector_path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError(f"{vector_path} is not a dense-vector bundle")
    ask_record = load_vector_record(bundle, ask_key)
    gate_bundle = torch.load(
        gate_vector_path, map_location="cpu", weights_only=False
    )
    if not isinstance(gate_bundle, dict):
        raise TypeError(f"{gate_vector_path} is not a dense-vector bundle")
    gate_record = load_vector_record(gate_bundle, gate_key)
    ask = ask_record["vector"].detach().float().flatten()
    gate = gate_record["vector"].detach().float().flatten()
    if ask.shape != gate.shape:
        raise ValueError(
            f"Ask and gate dimensions differ: {tuple(ask.shape)} vs {tuple(gate.shape)}"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random_vector = torch.randn(ask.shape, generator=generator, dtype=torch.float32)
    basis_matrix = torch.stack([ask, gate], dim=1)
    u, singular_values, _vh = torch.linalg.svd(basis_matrix, full_matrices=False)
    rank = int((singular_values > 1e-6).sum().item())
    if rank > 0:
        orthonormal_basis = u[:, :rank]
        random_vector = random_vector - orthonormal_basis @ (
            orthonormal_basis.T @ random_vector
        )
    if float(random_vector.norm()) < 1e-6:
        raise RuntimeError("Random direction collapsed during orthogonalization")
    random_vector = random_vector / random_vector.norm()

    key = "random_orthogonal_control"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "clarifysae_random_direction_control_v1",
            "source_vector_path": str(vector_path),
            "source_gate_vector_path": str(gate_vector_path),
            "seed": int(seed),
            "vectors": {
                key: {
                    "vector": random_vector,
                    "concept": "random_direction_control",
                    "method": "orthogonal_gaussian",
                    "hookpoint": ask_record.get("hookpoint"),
                    "average_residual_norm": float(
                        ask_record.get("average_residual_norm", 0.0)
                    ),
                    "orthogonal_to": [ask_key, gate_key],
                    "cosine_with_ask": float(
                        torch.dot(random_vector, ask / ask.norm().clamp_min(1e-8))
                    ),
                    "cosine_with_gate": float(
                        torch.dot(random_vector, gate / gate.norm().clamp_min(1e-8))
                    ),
                }
            },
        },
        output_path,
    )
    return key


def dense_config(
    template: dict[str, Any],
    *,
    experiment_name: str,
    dataset_path: str,
    output_root: str,
    vector_path: str,
    vector_key: str,
    gate_vector_path: str,
    gate_vector_key: str,
    hookpoint: str,
    strength: float,
    threshold: float,
    arm: str,
    gate_enabled: bool = True,
) -> dict[str, Any]:
    config = copy.deepcopy(template)
    config["experiment_name"] = experiment_name
    config["dataset"]["path"] = dataset_path
    config["dataset"]["limit"] = None
    config["output"]["root_dir"] = output_root
    steering = config["steering"]
    steering["vector_path"] = vector_path
    steering["vector_key"] = vector_key
    steering["hookpoint"] = hookpoint
    steering["module_path"] = hookpoint
    steering["strength"] = float(strength)
    steering["gate"]["enabled"] = bool(gate_enabled)
    steering["gate"]["vector_path"] = gate_vector_path
    steering["gate"]["vector_key"] = gate_vector_key
    steering["gate"]["threshold"] = float(threshold)
    config["run_metadata"] = {
        "arm": arm,
        "strength": float(strength),
        "gate_threshold": float(threshold),
        "gate_enabled": bool(gate_enabled),
    }
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--vector-path", required=True)
    parser.add_argument("--gate-vector-path", default=None)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--strength-selection", default=None)
    parser.add_argument("--gate-selection", default=None)
    parser.add_argument("--strength", type=float, default=0.02)
    parser.add_argument("--gate-threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--baseline-template",
        default="configs/baselines/baseline_llama32_1b_instruct_100.yaml",
    )
    parser.add_argument(
        "--dense-template",
        default="configs/steering/dense_vector_probe_gated.yaml",
    )
    parser.add_argument("--sae-feature", type=int, default=6230)
    parser.add_argument("--sae-layer", type=int, default=12)
    parser.add_argument("--sae-strength", type=float, default=-5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    vector_path = Path(args.vector_path)
    gate_vector_path = Path(args.gate_vector_path or args.vector_path)
    selection_path = Path(args.selection)
    for path in (dataset, vector_path, gate_vector_path, selection_path):
        if not path.exists():
            raise FileNotFoundError(path)

    selection = load_json(str(selection_path))
    strength = selected_value(load_json(args.strength_selection), args.strength)
    threshold = selected_value(load_json(args.gate_selection), args.gate_threshold)
    if strength == 0.0:
        raise ValueError("Selected strength is zero; a causal comparison requires a nonzero value")

    hookpoint = str(selection["hookpoint"])
    ask_key = str(selection["ask_vector_key"])
    gate_key = str(selection["gate_vector_key"])
    output_root = Path(args.output_root)
    config_dir = output_root / "configs"
    random_path = output_root / "artifacts/random_direction.pt"
    random_key = make_orthogonal_random_bundle(
        vector_path=vector_path,
        gate_vector_path=gate_vector_path,
        ask_key=ask_key,
        gate_key=gate_key,
        output_path=random_path,
        seed=args.seed,
    )

    baseline_template = load_yaml(Path(args.baseline_template))
    dense_template = load_yaml(Path(args.dense_template))
    records: list[dict[str, Any]] = []

    def register(arm: str, config: dict[str, Any]) -> None:
        config_path = config_dir / f"{config['experiment_name']}.yaml"
        save_yaml(config_path, config)
        run_dir = output_root / config["experiment_name"]
        records.append(
            {
                "arm": arm,
                "experiment_name": config["experiment_name"],
                "config_path": str(config_path),
                "example_metrics_path": str(run_dir / "metrics/example_metrics.csv"),
                "aggregate_metrics_path": str(run_dir / "tables/aggregate_metrics.csv"),
                "category_metrics_path": str(run_dir / "tables/category_metrics.csv"),
                "strength": config.get("steering", {}).get("strength", 0.0),
                "gate_threshold": config.get("steering", {}).get("gate", {}).get(
                    "threshold", None
                ),
                "gate_enabled": config.get("steering", {}).get("gate", {}).get(
                    "enabled", False
                ),
            }
        )

    baseline = copy.deepcopy(baseline_template)
    baseline["experiment_name"] = f"{args.prefix}_baseline"
    baseline["dataset"]["path"] = str(dataset)
    baseline["dataset"]["limit"] = None
    baseline["output"]["root_dir"] = str(output_root)
    baseline["steering"] = {"enabled": False}
    baseline["run_metadata"] = {"arm": "baseline"}
    register("baseline", baseline)

    ungated = dense_config(
        dense_template,
        experiment_name=f"{args.prefix}_dense_ungated",
        dataset_path=str(dataset),
        output_root=str(output_root),
        vector_path=str(vector_path),
        vector_key=ask_key,
        gate_vector_path=str(gate_vector_path),
        gate_vector_key=gate_key,
        hookpoint=hookpoint,
        strength=strength,
        threshold=threshold,
        arm="dense_ungated",
        gate_enabled=False,
    )
    register("dense_ungated", ungated)

    selected = dense_config(
        dense_template,
        experiment_name=f"{args.prefix}_dense_selected",
        dataset_path=str(dataset),
        output_root=str(output_root),
        vector_path=str(vector_path),
        vector_key=ask_key,
        gate_vector_path=str(gate_vector_path),
        gate_vector_key=gate_key,
        hookpoint=hookpoint,
        strength=strength,
        threshold=threshold,
        arm="dense_selected",
    )
    register("dense_selected", selected)

    sign_flip = dense_config(
        dense_template,
        experiment_name=f"{args.prefix}_dense_sign_flip",
        dataset_path=str(dataset),
        output_root=str(output_root),
        vector_path=str(vector_path),
        vector_key=ask_key,
        gate_vector_path=str(gate_vector_path),
        gate_vector_key=gate_key,
        hookpoint=hookpoint,
        strength=-strength,
        threshold=threshold,
        arm="dense_sign_flip",
    )
    register("dense_sign_flip", sign_flip)

    random_control = dense_config(
        dense_template,
        experiment_name=f"{args.prefix}_random_direction",
        dataset_path=str(dataset),
        output_root=str(output_root),
        vector_path=str(random_path),
        vector_key=random_key,
        gate_vector_path=str(gate_vector_path),
        gate_vector_key=gate_key,
        hookpoint=hookpoint,
        strength=strength,
        threshold=threshold,
        arm="random_direction",
    )
    register("random_direction", random_control)

    sae = copy.deepcopy(baseline_template)
    sae["experiment_name"] = f"{args.prefix}_sae_baseline"
    sae["dataset"]["path"] = str(dataset)
    sae["dataset"]["limit"] = None
    sae["output"]["root_dir"] = str(output_root)
    sae["steering"] = {
        "enabled": True,
        "method": "sae",
        "loader": "sparsify",
        "sae_repo": "apart/llama3.2_1b_instruct_saes_vader",
        "hookpoint": f"model.layers.{args.sae_layer}",
        "module_path": f"model.layers.{args.sae_layer}",
        "feature_indices": [int(args.sae_feature)],
        "strength": float(args.sae_strength),
        "mode": "additive",
        "apply_to": "all_positions",
        "steer_generated_tokens_only": True,
        "runtime": {
            "normalize_reconstruction": True,
            "preserve_unsteered_residual": True,
            "clamp_latents": None,
            "log_feature_acts": False,
        },
    }
    sae["run_metadata"] = {
        "arm": "sae_baseline",
        "feature": int(args.sae_feature),
        "layer": int(args.sae_layer),
        "strength": float(args.sae_strength),
    }
    register("sae_baseline", sae)

    manifest = pd.DataFrame(records)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    metadata = {
        "prefix": args.prefix,
        "dataset": str(dataset),
        "vector_path": str(vector_path),
        "gate_vector_path": str(gate_vector_path),
        "selection_path": str(selection_path),
        "hookpoint": hookpoint,
        "ask_vector_key": ask_key,
        "gate_vector_key": gate_key,
        "selected_strength": strength,
        "gate_threshold": threshold,
        "random_bundle": str(random_path),
        "seed": int(args.seed),
        "arms": records,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(manifest.to_string(index=False))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
