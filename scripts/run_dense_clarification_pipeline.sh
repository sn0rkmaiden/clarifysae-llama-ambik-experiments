#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
OVERASK_PENALTY="${OVERASK_PENALTY:-0.50}"
INVALID_JSON_PENALTY="${INVALID_JSON_PENALTY:-0.25}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

usage() {
  cat <<'EOF'
ClarifySAE dense clarification experiment runner

Run from the repository root:

  bash scripts/run_dense_clarification_pipeline.sh <command>

Commands:
  setup              Create .venv, install the package, and run unit tests.
  preflight          Validate configs, dependencies, datasets, and CUDA.
  prepare            Build paired AmbiK files and all static local configs.
  baseline-calib     Run the paired calibration baseline.
  smoke              Generate a small corpus and extract smoke-test vectors.
  full-vectors       Generate the full synthetic corpus and extract vectors.
  dense-sweep        Select a joint layer and sweep unconditional strengths.
  gate-sweep         Fix the selected strength and sweep gate thresholds.
  final-test         Run baseline, unconditional, and gated systems on held-out test data.
  ambik-vectors      Build benchmark-derived vectors and run their calibration sweeps.
  ambik-final-test   Run benchmark-derived baseline/unconditional/gated internal-test evaluations.
  project-sae        Project the selected synthetic dense direction to 1-32 SAE features.
  summarize          Print paths to all selection and summary tables.
  phase1             setup + prepare + preflight + baseline-calib + smoke.
  phase2             full-vectors + dense-sweep + gate-sweep.
  all                phase1 + phase2 + final-test + ambik-vectors + ambik-final-test + project-sae.
  status             Show which required and generated files currently exist.
  help               Show this message.

Environment variables:
  VENV_DIR               Virtual environment path; default: .venv
  OVERASK_PENALTY        Selection penalty for clear-input over-asking; default: 0.50
  INVALID_JSON_PENALTY   Selection penalty for invalid output JSON; default: 0.25
  HF_TOKEN               Hugging Face token when a model/checkpoint requires it

The script never modifies the bundled experiment templates. It writes generated
configs under configs/local/ and all results under outputs/.
EOF
}

say() {
  printf '\n\033[1;34m==> %s\033[0m\n' "$*"
}

fail() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

require_python() {
  [[ -x "$PY" ]] || fail "Virtual environment is missing. Run: bash scripts/run_dense_clarification_pipeline.sh setup"
}

require_file() {
  [[ -f "$1" ]] || fail "Missing required file: $1"
}

run_setup() {
  say "Creating virtual environment"
  python3 -m venv "$VENV_DIR"
  "$PY" -m pip install --upgrade pip
  "$PIP" install -e .
  say "Running unit tests"
  "$PY" -m pytest -q
}

run_preflight() {
  require_python
  mkdir -p outputs
  "$PY" scripts/preflight_experiments.py --output outputs/preflight_report.json
}

run_prepare() {
  require_python
  require_file data/raw/ambik/ambik_calib_100.csv
  require_file data/raw/ambik/ambik_test_400.csv
  say "Creating paired ambiguous/clear AmbiK files"
  "$PY" scripts/prepare_dense_experiment_data.py
  say "Creating local experiment configs"
  "$PY" scripts/prepare_dense_experiment_configs.py static --root .
}

run_baseline_calib() {
  require_python
  require_file data/raw/ambik/ambik_calib_100_paired.csv
  require_file configs/local/baseline_llama32_1b_paired_calib.yaml
  "$PY" -m clarifysae_llama.runners.run_eval \
    --config configs/local/baseline_llama32_1b_paired_calib.yaml
  "$PY" scripts/summarize_selective_clarification.py run \
    --example-metrics outputs/baseline_llama32_1b_paired_calib/metrics/example_metrics.csv \
    --output outputs/selection/baseline_calib_summary.json
}

run_smoke() {
  require_python
  require_file configs/local/generate_clarification_counterfactuals_smoke.yaml
  say "Generating 1 synthetic scenario per topic"
  "$PY" -m clarifysae_llama.runners.generate_synthetic_corpus \
    --config configs/local/generate_clarification_counterfactuals_smoke.yaml
  say "Extracting smoke-test vectors"
  "$PY" -m clarifysae_llama.runners.extract_concept_vectors \
    --config configs/local/extract_clarification_vectors_smoke.yaml
  say "Ranking smoke-test layers"
  "$PY" scripts/select_dense_layer.py \
    --diagnostics outputs/concept_vectors/llama32_1b_clarification_vectors_smoke.csv \
    --output outputs/selection/smoke_layer_selection.json
}

run_full_vectors() {
  require_python
  say "Generating the full synthetic matched corpus"
  "$PY" -m clarifysae_llama.runners.generate_synthetic_corpus \
    --config configs/synthetic/generate_clarification_counterfactuals.yaml
  say "Extracting full synthetic dense vectors and probes"
  "$PY" -m clarifysae_llama.runners.extract_concept_vectors \
    --config configs/concept_vectors/extract_clarification_vectors.yaml
}

prepare_dense_configs() {
  local source="$1"
  local vector_path="$2"
  local selection="$3"
  local test_dataset="$4"
  local strength_selection="${5:-}"
  local gate_selection="${6:-}"

  local args=(
    scripts/prepare_dense_experiment_configs.py dense
    --root .
    --source "$source"
    --vector-path "$vector_path"
    --selection "$selection"
    --calib-dataset data/raw/ambik/ambik_calib_100_paired.csv
    --test-dataset "$test_dataset"
  )
  if [[ -n "$strength_selection" ]]; then
    args+=(--strength-selection "$strength_selection")
  fi
  if [[ -n "$gate_selection" ]]; then
    args+=(--gate-selection "$gate_selection")
  fi
  "$PY" "${args[@]}"
}

run_dense_sweep_for_source() {
  local source="$1"
  local vector_path="$2"
  local diagnostics="$3"
  local selection="$4"
  local test_dataset="$5"

  require_file "$vector_path"
  require_file "$diagnostics"
  say "Selecting one joint layer for ambiguity detection and ask steering ($source)"
  "$PY" scripts/select_dense_layer.py \
    --diagnostics "$diagnostics" \
    --output "$selection"

  prepare_dense_configs "$source" "$vector_path" "$selection" "$test_dataset"

  say "Sweeping unconditional dense-vector strengths ($source)"
  "$PY" -m clarifysae_llama.runners.sweep \
    --config "configs/local/${source}_sweep_dense_strengths.yaml"

  local sweep_dir="outputs/sweeps/${source}_dense_unconditional_strength_sweep"
  "$PY" scripts/summarize_selective_clarification.py sweep \
    --manifest "$sweep_dir/manifest.csv" \
    --output-csv "outputs/selection/${source}_strength_ranking.csv" \
    --selection-json "outputs/selection/${source}_strength_selection.json" \
    --overask-penalty "$OVERASK_PENALTY" \
    --invalid-json-penalty "$INVALID_JSON_PENALTY"

  prepare_dense_configs \
    "$source" "$vector_path" "$selection" "$test_dataset" \
    "outputs/selection/${source}_strength_selection.json"
}

run_gate_sweep_for_source() {
  local source="$1"
  local vector_path="$2"
  local selection="$3"
  local test_dataset="$4"
  require_file "outputs/selection/${source}_strength_selection.json"

  prepare_dense_configs \
    "$source" "$vector_path" "$selection" "$test_dataset" \
    "outputs/selection/${source}_strength_selection.json"

  say "Sweeping ambiguity-gate thresholds ($source)"
  "$PY" -m clarifysae_llama.runners.sweep \
    --config "configs/local/${source}_sweep_gate_thresholds.yaml"

  local sweep_dir="outputs/sweeps/${source}_dense_gate_threshold_sweep"
  "$PY" scripts/summarize_selective_clarification.py sweep \
    --manifest "$sweep_dir/manifest.csv" \
    --output-csv "outputs/selection/${source}_gate_ranking.csv" \
    --selection-json "outputs/selection/${source}_gate_selection.json" \
    --overask-penalty "$OVERASK_PENALTY" \
    --invalid-json-penalty "$INVALID_JSON_PENALTY"

  prepare_dense_configs \
    "$source" "$vector_path" "$selection" "$test_dataset" \
    "outputs/selection/${source}_strength_selection.json" \
    "outputs/selection/${source}_gate_selection.json"
}

run_dense_sweep() {
  require_python
  run_dense_sweep_for_source \
    synthetic \
    outputs/concept_vectors/llama32_1b_clarification_vectors.pt \
    outputs/concept_vectors/llama32_1b_clarification_vectors.csv \
    outputs/selection/synthetic_layer_selection.json \
    data/raw/ambik/ambik_test_400_paired.csv
}

run_gate_sweep() {
  require_python
  run_gate_sweep_for_source \
    synthetic \
    outputs/concept_vectors/llama32_1b_clarification_vectors.pt \
    outputs/selection/synthetic_layer_selection.json \
    data/raw/ambik/ambik_test_400_paired.csv
}

run_final_test() {
  require_python
  require_file outputs/selection/synthetic_gate_selection.json
  say "Running held-out baseline"
  "$PY" -m clarifysae_llama.runners.run_eval \
    --config configs/local/baseline_llama32_1b_paired_test.yaml
  say "Running held-out unconditional dense steering"
  "$PY" -m clarifysae_llama.runners.run_eval \
    --config configs/local/synthetic_dense_unconditional_test.yaml
  say "Running held-out probe-gated dense steering"
  "$PY" -m clarifysae_llama.runners.run_eval \
    --config configs/local/synthetic_dense_gated_test.yaml

  for run in \
    baseline_llama32_1b_paired_test \
    synthetic_dense_unconditional_test \
    synthetic_dense_gated_test; do
    "$PY" scripts/summarize_selective_clarification.py run \
      --example-metrics "outputs/$run/metrics/example_metrics.csv" \
      --output "outputs/selection/${run}_summary.json"
  done
}

run_ambik_vectors() {
  require_python
  require_file data/raw/ambik/ambik_test_400.csv
  say "Building benchmark-derived factorized concept corpus"
  "$PY" -m clarifysae_llama.runners.build_ambik_concept_corpus \
    --config configs/concept_vectors/build_ambik_concept_corpus.yaml
  say "Extracting benchmark-derived vectors"
  "$PY" -m clarifysae_llama.runners.extract_concept_vectors \
    --config configs/concept_vectors/extract_ambik_vectors.yaml

  run_dense_sweep_for_source \
    ambik \
    outputs/concept_vectors/llama32_1b_ambik_vectors.pt \
    outputs/concept_vectors/llama32_1b_ambik_vectors.csv \
    outputs/selection/ambik_layer_selection.json \
    data/raw/ambik/ambik_test_400_internal_test_paired.csv

  run_gate_sweep_for_source \
    ambik \
    outputs/concept_vectors/llama32_1b_ambik_vectors.pt \
    outputs/selection/ambik_layer_selection.json \
    data/raw/ambik/ambik_test_400_internal_test_paired.csv
}

run_ambik_final_test() {
  require_python
  require_file outputs/selection/ambik_gate_selection.json
  "$PY" -m clarifysae_llama.runners.run_eval \
    --config configs/local/baseline_llama32_1b_internal_test.yaml
  "$PY" -m clarifysae_llama.runners.run_eval \
    --config configs/local/ambik_dense_unconditional_test.yaml
  "$PY" -m clarifysae_llama.runners.run_eval \
    --config configs/local/ambik_dense_gated_test.yaml

  for run in \
    baseline_llama32_1b_internal_test \
    ambik_dense_unconditional_test \
    ambik_dense_gated_test; do
    "$PY" scripts/summarize_selective_clarification.py run \
      --example-metrics "outputs/$run/metrics/example_metrics.csv" \
      --output "outputs/selection/${run}_summary.json"
  done
}

run_project_sae() {
  require_python
  require_file outputs/selection/synthetic_layer_selection.json
  say "Creating a projection config for the automatically selected layer"
  "$PY" - <<'PY'
import json
from pathlib import Path
import yaml

selection = json.loads(Path("outputs/selection/synthetic_layer_selection.json").read_text())
source = Path("configs/sae_projection/project_dense_vector_to_sae.yaml")
config = yaml.safe_load(source.read_text())
hookpoint = selection["hookpoint"]
layer = int(selection["layer"])
config["experiment_name"] = f"project_dense_clarification_vector_to_sae_layer{layer}"
projection = config["sae_projection"]
projection["vector_key"] = selection["ask_vector_key"]
projection["hookpoint"] = hookpoint
projection["module_path"] = hookpoint
projection["output_dir"] = f"outputs/sae_projection/ask_trajectory_layer{layer}"
output = Path("configs/local/project_synthetic_dense_vector_to_sae.yaml")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(f"Wrote {output}")
PY
  "$PY" -m clarifysae_llama.runners.project_concept_vector_to_sae \
    --config configs/local/project_synthetic_dense_vector_to_sae.yaml
}

run_summarize() {
  echo "Layer selections:"
  find outputs/selection -maxdepth 1 -name '*layer_selection.json' -print 2>/dev/null | sort || true
  echo "Strength selections:"
  find outputs/selection -maxdepth 1 -name '*strength_selection.json' -print 2>/dev/null | sort || true
  echo "Gate selections:"
  find outputs/selection -maxdepth 1 -name '*gate_selection.json' -print 2>/dev/null | sort || true
  echo "Final summaries:"
  find outputs/selection -maxdepth 1 -name '*_summary.json' -print 2>/dev/null | sort || true
  echo "Sweep rankings:"
  find outputs/selection -maxdepth 1 -name '*_ranking.csv' -print 2>/dev/null | sort || true
}

run_status() {
  local files=(
    data/raw/ambik/ambik_calib_100.csv
    data/raw/ambik/ambik_test_400.csv
    data/raw/ambik/ambik_calib_100_paired.csv
    data/raw/ambik/ambik_test_400_paired.csv
    outputs/synthetic/clarification_counterfactuals_smoke.jsonl
    outputs/concept_vectors/llama32_1b_clarification_vectors_smoke.pt
    outputs/synthetic/clarification_counterfactuals.jsonl
    outputs/concept_vectors/llama32_1b_clarification_vectors.pt
    outputs/selection/synthetic_layer_selection.json
    outputs/selection/synthetic_strength_selection.json
    outputs/selection/synthetic_gate_selection.json
    outputs/concept_vectors/llama32_1b_ambik_vectors.pt
    outputs/selection/ambik_gate_selection.json
  )
  for path in "${files[@]}"; do
    if [[ -e "$path" ]]; then
      printf 'OK      %s\n' "$path"
    else
      printf 'MISSING %s\n' "$path"
    fi
  done
}

run_phase1() {
  run_setup
  run_prepare
  run_preflight
  run_baseline_calib
  run_smoke
}

run_phase2() {
  run_full_vectors
  run_dense_sweep
  run_gate_sweep
}

command="${1:-help}"
case "$command" in
  setup) run_setup ;;
  preflight) run_preflight ;;
  prepare) run_prepare ;;
  baseline-calib) run_baseline_calib ;;
  smoke) run_smoke ;;
  full-vectors) run_full_vectors ;;
  dense-sweep) run_dense_sweep ;;
  gate-sweep) run_gate_sweep ;;
  final-test) run_final_test ;;
  ambik-vectors) run_ambik_vectors ;;
  ambik-final-test) run_ambik_final_test ;;
  project-sae) run_project_sae ;;
  summarize) run_summarize ;;
  status) run_status ;;
  phase1) run_phase1 ;;
  phase2) run_phase2 ;;
  all)
    run_phase1
    run_phase2
    run_final_test
    run_ambik_vectors
    run_ambik_final_test
    run_project_sae
    ;;
  help|-h|--help) usage ;;
  *) usage; fail "Unknown command: $command" ;;
esac
