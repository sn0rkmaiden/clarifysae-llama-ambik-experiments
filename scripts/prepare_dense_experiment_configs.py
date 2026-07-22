from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} does not contain a YAML mapping")
    return value


def save_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)
    print(f"Wrote {path}")


def load_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_static(root: Path) -> None:
    local = root / "configs/local"

    baseline = load_yaml(
        root / "configs/baselines/baseline_llama32_1b_instruct_100.yaml"
    )
    baseline["experiment_name"] = "baseline_llama32_1b_paired_calib"
    baseline["dataset"]["path"] = "data/raw/ambik/ambik_calib_100_paired.csv"
    baseline["dataset"]["limit"] = None
    save_yaml(local / "baseline_llama32_1b_paired_calib.yaml", baseline)

    baseline_test = json.loads(json.dumps(baseline))
    baseline_test["experiment_name"] = "baseline_llama32_1b_paired_test"
    baseline_test["dataset"]["path"] = "data/raw/ambik/ambik_test_400_paired.csv"
    save_yaml(local / "baseline_llama32_1b_paired_test.yaml", baseline_test)

    baseline_internal = json.loads(json.dumps(baseline))
    baseline_internal["experiment_name"] = "baseline_llama32_1b_internal_test"
    baseline_internal["dataset"]["path"] = (
        "data/raw/ambik/ambik_test_400_internal_test_paired.csv"
    )
    save_yaml(local / "baseline_llama32_1b_internal_test.yaml", baseline_internal)

    synthetic = load_yaml(
        root / "configs/synthetic/generate_clarification_counterfactuals.yaml"
    )
    synthetic["experiment_name"] = "synthetic_clarification_counterfactuals_smoke"
    synthetic["generation"]["max_new_tokens"] = 384
    synthetic["synthetic_corpus"]["output_path"] = (
        "outputs/synthetic/clarification_counterfactuals_smoke.jsonl"
    )
    synthetic["synthetic_corpus"]["scenarios_per_topic"] = 1
    save_yaml(local / "generate_clarification_counterfactuals_smoke.yaml", synthetic)

    extraction = load_yaml(
        root / "configs/concept_vectors/extract_clarification_vectors.yaml"
    )
    extraction["experiment_name"] = "extract_clarification_vectors_smoke"
    extraction["concept_discovery"]["dataset_path"] = (
        "outputs/synthetic/clarification_counterfactuals_smoke.jsonl"
    )
    extraction["concept_discovery"]["output_path"] = (
        "outputs/concept_vectors/llama32_1b_clarification_vectors_smoke.pt"
    )
    extraction["concept_discovery"]["batch_size"] = 2
    save_yaml(local / "extract_clarification_vectors_smoke.yaml", extraction)


def write_dense(
    root: Path,
    *,
    source: str,
    vector_path: str,
    selection_path: Path,
    strength_selection_path: str | None,
    gate_selection_path: str | None,
    calib_dataset: str,
    test_dataset: str,
    strength_values: list[float] | None = None,
    gate_values: list[float] | None = None,
) -> None:
    selection = load_json(str(selection_path))
    strength_selection = load_json(strength_selection_path)
    gate_selection = load_json(gate_selection_path)

    hookpoint = str(selection["hookpoint"])
    ask_key = str(selection["ask_vector_key"])
    gate_key = str(selection["gate_vector_key"])
    strength = float(strength_selection.get("value", 0.02))
    threshold = float(gate_selection.get("value", 0.0))
    local = root / "configs/local"

    base = load_yaml(root / "configs/steering/dense_vector_probe_gated.yaml")
    base["steering"]["vector_path"] = vector_path
    base["steering"]["vector_key"] = ask_key
    base["steering"]["hookpoint"] = hookpoint
    base["steering"]["module_path"] = hookpoint
    base["steering"]["strength"] = strength
    base["steering"]["gate"]["vector_path"] = vector_path
    base["steering"]["gate"]["vector_key"] = gate_key
    base["steering"]["gate"]["threshold"] = threshold
    base["dataset"]["path"] = calib_dataset
    base["dataset"]["limit"] = None

    unconditional = json.loads(json.dumps(base))
    unconditional["experiment_name"] = f"{source}_dense_unconditional_calib"
    unconditional["steering"]["gate"]["enabled"] = False
    save_yaml(local / f"{source}_dense_unconditional.yaml", unconditional)

    strength_sweep = load_yaml(
        root / "configs/steering/sweep_dense_vector_strengths.yaml"
    )
    strength_sweep["experiment_name"] = f"{source}_dense_unconditional_strength_sweep"
    strength_sweep["base_config"] = (
        f"configs/local/{source}_dense_unconditional.yaml"
    )
    if strength_values:
        strength_sweep["sweep"]["values"] = [float(value) for value in strength_values]
    save_yaml(local / f"{source}_sweep_dense_strengths.yaml", strength_sweep)

    gated = json.loads(json.dumps(base))
    gated["experiment_name"] = f"{source}_dense_gated_calib"
    gated["steering"]["gate"]["enabled"] = True
    save_yaml(local / f"{source}_dense_gated.yaml", gated)

    gate_sweep = load_yaml(
        root / "configs/steering/sweep_dense_gate_thresholds.yaml"
    )
    gate_sweep["experiment_name"] = f"{source}_dense_gate_threshold_sweep"
    gate_sweep["base_config"] = f"configs/local/{source}_dense_gated.yaml"
    if gate_values:
        gate_sweep["sweep"]["values"] = [float(value) for value in gate_values]
    save_yaml(local / f"{source}_sweep_gate_thresholds.yaml", gate_sweep)

    final_unconditional = json.loads(json.dumps(unconditional))
    final_unconditional["experiment_name"] = f"{source}_dense_unconditional_test"
    final_unconditional["dataset"]["path"] = test_dataset
    final_unconditional["steering"]["strength"] = strength
    save_yaml(local / f"{source}_dense_unconditional_test.yaml", final_unconditional)

    final_gated = json.loads(json.dumps(gated))
    final_gated["experiment_name"] = f"{source}_dense_gated_test"
    final_gated["dataset"]["path"] = test_dataset
    final_gated["steering"]["strength"] = strength
    final_gated["steering"]["gate"]["threshold"] = threshold
    save_yaml(local / f"{source}_dense_gated_test.yaml", final_gated)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    static = subparsers.add_parser("static")
    static.add_argument("--root", default=".")

    dense = subparsers.add_parser("dense")
    dense.add_argument("--root", default=".")
    dense.add_argument("--source", required=True)
    dense.add_argument("--vector-path", required=True)
    dense.add_argument("--selection", required=True)
    dense.add_argument("--strength-selection", default=None)
    dense.add_argument("--gate-selection", default=None)
    dense.add_argument(
        "--calib-dataset",
        default="data/raw/ambik/ambik_calib_100_paired.csv",
    )
    dense.add_argument(
        "--test-dataset",
        default="data/raw/ambik/ambik_test_400_paired.csv",
    )
    dense.add_argument("--strength-values", nargs="*", type=float, default=None)
    dense.add_argument("--gate-values", nargs="*", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if args.command == "static":
        write_static(root)
        return
    write_dense(
        root,
        source=args.source,
        vector_path=args.vector_path,
        selection_path=Path(args.selection),
        strength_selection_path=args.strength_selection,
        gate_selection_path=args.gate_selection,
        calib_dataset=args.calib_dataset,
        test_dataset=args.test_dataset,
        strength_values=args.strength_values,
        gate_values=args.gate_values,
    )


if __name__ == "__main__":
    main()
