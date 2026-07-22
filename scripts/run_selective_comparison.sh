#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-smoke}"
PYTHON="${PYTHON:-python}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

require_file() {
  [[ -f "$1" ]] || { echo "ERROR: missing required file: $1" >&2; exit 1; }
}

run_config_manifest() {
  local manifest="$1"
  "$PYTHON" scripts/run_selective_manifest.py --manifest "$manifest"
}

package_results() {
  local root="$1"
  local mode="$2"
  local list="$root/results_to_return_files.txt"
  local archive="$root/results_to_return.tar.gz"
  find "$root" -type f \( \
    -name 'manifest.csv' -o \
    -name 'manifest.json' -o \
    -name 'summary.csv' -o \
    -name 'summary.json' -o \
    -name 'category_summary.csv' -o \
    -name 'example_metrics.csv' -o \
    -name 'aggregate_metrics.csv' -o \
    -name 'category_metrics.csv' -o \
    -name 'predictions_full.jsonl' -o \
    -name 'random_direction.pt' \
  \) | sort > "$list"

  local extra=()
  if [[ "$mode" == "smoke" ]]; then
    extra+=(
      outputs/synthetic/clarification_counterfactuals_smoke.metadata.json
      outputs/synthetic/clarification_counterfactuals_smoke.failures.json
      outputs/concept_vectors/llama32_1b_clarification_vectors_smoke.csv
      outputs/concept_vectors/llama32_1b_clarification_vectors_smoke.config.json
      outputs/selection/smoke_layer_selection.json
      outputs/selection/smoke_layer_selection.csv
    )
  else
    extra+=(
      outputs/synthetic/clarification_counterfactuals.metadata.json
      outputs/synthetic/clarification_counterfactuals.failures.json
      outputs/concept_vectors/llama32_1b_clarification_vectors.csv
      outputs/concept_vectors/llama32_1b_clarification_vectors.config.json
      outputs/selection/synthetic_layer_selection.json
      outputs/selection/synthetic_layer_selection.csv
      outputs/selection/selective_strength_selection.json
      outputs/selection/selective_strength_ranking.csv
      outputs/selection/selective_gate_selection.json
      outputs/selection/selective_gate_ranking.csv
    )
  fi
  for path in "${extra[@]}"; do
    [[ -f "$path" ]] && printf '%s\n' "$path" >> "$list"
  done
  sort -u -o "$list" "$list"
  tar -czf "$archive" -T "$list"
  echo "Return archive: $archive"
  sha256sum "$archive"
}

prepare_common() {
  require_file data/raw/ambik/ambik_calib_100.csv
  require_file data/raw/ambik/ambik_test_400.csv
  "$PYTHON" scripts/prepare_dense_experiment_data.py \
    --smoke-pairs 4 \
    --selection-pairs 50 \
    --heldout-pairs 100
  "$PYTHON" scripts/prepare_dense_experiment_configs.py static --root .
}

run_smoke() {
  prepare_common

  if [[ ! -f outputs/synthetic/clarification_counterfactuals_smoke.jsonl ]]; then
    "$PYTHON" -m clarifysae_llama.runners.generate_synthetic_corpus \
      --config configs/local/generate_clarification_counterfactuals_smoke.yaml
  fi
  if [[ ! -f outputs/concept_vectors/llama32_1b_clarification_vectors_smoke.pt ]]; then
    "$PYTHON" -m clarifysae_llama.runners.extract_concept_vectors \
      --config configs/local/extract_clarification_vectors_smoke.yaml
  fi
  "$PYTHON" scripts/select_dense_layer.py \
    --diagnostics outputs/concept_vectors/llama32_1b_clarification_vectors_smoke.csv \
    --output outputs/selection/smoke_layer_selection.json

  local out=outputs/selective_smoke_comparison
  "$PYTHON" scripts/build_selective_controls.py \
    --prefix smoke \
    --dataset data/raw/ambik/ambik_calib_smoke_paired.csv \
    --vector-path outputs/concept_vectors/llama32_1b_clarification_vectors_smoke.pt \
    --selection outputs/selection/smoke_layer_selection.json \
    --strength 0.02 \
    --gate-threshold 0.0 \
    --output-root "$out"
  run_config_manifest "$out/manifest.csv"
  "$PYTHON" scripts/compare_selective_arms.py \
    --manifest "$out/manifest.csv" \
    --output-dir "$out/comparison" \
    --bootstrap-samples 200 \
    --seed 42
  package_results "$out" smoke
}

run_main() {
  prepare_common

  if [[ ! -f outputs/synthetic/clarification_counterfactuals.jsonl ]]; then
    "$PYTHON" -m clarifysae_llama.runners.generate_synthetic_corpus \
      --config configs/synthetic/generate_clarification_counterfactuals.yaml
  fi
  if [[ ! -f outputs/concept_vectors/llama32_1b_clarification_vectors.pt ]]; then
    "$PYTHON" -m clarifysae_llama.runners.extract_concept_vectors \
      --config configs/concept_vectors/extract_clarification_vectors.yaml
  fi
  "$PYTHON" scripts/select_dense_layer.py \
    --diagnostics outputs/concept_vectors/llama32_1b_clarification_vectors.csv \
    --output outputs/selection/synthetic_layer_selection.json

  "$PYTHON" scripts/prepare_dense_experiment_configs.py dense \
    --root . \
    --source selective \
    --vector-path outputs/concept_vectors/llama32_1b_clarification_vectors.pt \
    --selection outputs/selection/synthetic_layer_selection.json \
    --calib-dataset data/raw/ambik/ambik_calib_select50_paired.csv \
    --test-dataset data/raw/ambik/ambik_test_eval100_paired.csv \
    --strength-values -0.05 -0.02 0.0 0.02 0.05 \
    --gate-values -1.0 0.0 1.0

  if [[ ! -f outputs/selection/selective_strength_selection.json ]]; then
    "$PYTHON" -m clarifysae_llama.runners.sweep \
      --config configs/local/selective_sweep_dense_strengths.yaml
    "$PYTHON" scripts/summarize_selective_clarification.py sweep \
      --manifest outputs/sweeps/selective_dense_unconditional_strength_sweep/manifest.csv \
      --output-csv outputs/selection/selective_strength_ranking.csv \
      --selection-json outputs/selection/selective_strength_selection.json \
      --overask-penalty 0.50 \
      --invalid-json-penalty 0.25 \
      --exclude-zero-parameter
  fi

  "$PYTHON" scripts/prepare_dense_experiment_configs.py dense \
    --root . \
    --source selective \
    --vector-path outputs/concept_vectors/llama32_1b_clarification_vectors.pt \
    --selection outputs/selection/synthetic_layer_selection.json \
    --strength-selection outputs/selection/selective_strength_selection.json \
    --calib-dataset data/raw/ambik/ambik_calib_select50_paired.csv \
    --test-dataset data/raw/ambik/ambik_test_eval100_paired.csv \
    --strength-values -0.05 -0.02 0.0 0.02 0.05 \
    --gate-values -1.0 0.0 1.0

  if [[ ! -f outputs/selection/selective_gate_selection.json ]]; then
    "$PYTHON" -m clarifysae_llama.runners.sweep \
      --config configs/local/selective_sweep_gate_thresholds.yaml
    "$PYTHON" scripts/summarize_selective_clarification.py sweep \
      --manifest outputs/sweeps/selective_dense_gate_threshold_sweep/manifest.csv \
      --output-csv outputs/selection/selective_gate_ranking.csv \
      --selection-json outputs/selection/selective_gate_selection.json \
      --overask-penalty 0.50 \
      --invalid-json-penalty 0.25
  fi

  local out=outputs/selective_main_comparison
  "$PYTHON" scripts/build_selective_controls.py \
    --prefix main \
    --dataset data/raw/ambik/ambik_test_eval100_paired.csv \
    --vector-path outputs/concept_vectors/llama32_1b_clarification_vectors.pt \
    --selection outputs/selection/synthetic_layer_selection.json \
    --strength-selection outputs/selection/selective_strength_selection.json \
    --gate-selection outputs/selection/selective_gate_selection.json \
    --output-root "$out"
  run_config_manifest "$out/manifest.csv"
  "$PYTHON" scripts/compare_selective_arms.py \
    --manifest "$out/manifest.csv" \
    --output-dir "$out/comparison" \
    --bootstrap-samples 1000 \
    --seed 42
  package_results "$out" main
}

case "$MODE" in
  smoke) run_smoke ;;
  main) run_main ;;
  *)
    echo "Usage: $0 {smoke|main}" >&2
    exit 2
    ;;
esac
