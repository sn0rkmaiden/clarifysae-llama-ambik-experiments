#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-pilot}"
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
    -name 'pilot_assessment.json' -o \
    -name 'category_summary.csv' -o \
    -name 'example_metrics.csv' -o \
    -name 'aggregate_metrics.csv' -o \
    -name 'category_metrics.csv' -o \
    -name 'predictions_full.jsonl' -o \
    -name 'random_direction.pt' \
  \) | sort > "$list"

  local extra=(
    outputs/probe_corpus/ambik_probe_train60_dev20.jsonl
    outputs/probe_corpus/ambik_probe_train60_dev20.metadata.json
    outputs/concept_vectors/llama32_1b_ambik_clarification_actor_vectors_pilot.csv
    outputs/concept_vectors/llama32_1b_ambik_clarification_actor_vectors_pilot.config.json
    outputs/concept_vectors/llama32_1b_ambik_ambiguity_probe_pilot.csv
    outputs/concept_vectors/llama32_1b_ambik_ambiguity_probe_pilot.config.json
    outputs/selection/corrective_pilot_layer_selection.json
    outputs/selection/corrective_pilot_layer_selection.csv
    outputs/selection/corrective_pilot_strength_selection.json
    outputs/selection/corrective_pilot_strength_ranking.csv
    outputs/selection/corrective_pilot_gate_selection.json
    outputs/selection/corrective_pilot_gate_ranking.csv
  )
  if [[ "$mode" == "pilot" || "$mode" == "smoke" ]]; then
    extra+=(
      outputs/actor_corpus/ambik_actor_train60_dev20.jsonl
      outputs/actor_corpus/ambik_actor_train60_dev20.metadata.json
      outputs/actor_corpus/ambik_actor_train60_dev20.complete
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
    --probe-train-pairs 60 \
    --pilot-select-pairs 20 \
    --pilot-eval-pairs 20 \
    --heldout-pairs 100
  "$PYTHON" scripts/prepare_dense_experiment_configs.py static --root .
}

prepare_corrective_representations() {
  local actor_corpus=outputs/actor_corpus/ambik_actor_train60_dev20.jsonl
  local actor_complete=outputs/actor_corpus/ambik_actor_train60_dev20.complete
  local actor=outputs/concept_vectors/llama32_1b_ambik_clarification_actor_vectors_pilot.pt
  local probe_corpus=outputs/probe_corpus/ambik_probe_train60_dev20.jsonl
  local gate=outputs/concept_vectors/llama32_1b_ambik_ambiguity_probe_pilot.pt
  local selection=outputs/selection/corrective_pilot_layer_selection.json

  if [[ ! -f "$actor_complete" ]]; then
    rm -f \
      "$actor_corpus" \
      outputs/actor_corpus/ambik_actor_train60_dev20.metadata.json \
      "$actor_complete"
    "$PYTHON" scripts/build_ambik_actor_corpus.py \
      --train data/raw/ambik/ambik_calib_probe_train60.csv \
      --dev data/raw/ambik/ambik_calib_pilot_select20.csv \
      --output "$actor_corpus" \
      --expected-train 60 \
      --expected-dev 20
  fi
  require_file "$actor_corpus"
  require_file "$actor_complete"

  if [[ ! -f "$probe_corpus" ]]; then
    "$PYTHON" scripts/build_ambik_probe_corpus.py \
      --train data/raw/ambik/ambik_calib_probe_train60_paired.csv \
      --dev data/raw/ambik/ambik_calib_pilot_select20_paired.csv \
      --output "$probe_corpus"
  fi

  if [[ ! -f "$actor" ]]; then
    "$PYTHON" -m clarifysae_llama.runners.extract_concept_vectors \
      --config configs/local/extract_clarification_actor_vectors_pilot.yaml
  fi
  if [[ ! -f "$gate" ]]; then
    "$PYTHON" -m clarifysae_llama.runners.extract_concept_vectors \
      --config configs/local/extract_ambik_ambiguity_probe_pilot.yaml
  fi

  "$PYTHON" scripts/select_dense_layer.py \
    --actor-diagnostics outputs/concept_vectors/llama32_1b_ambik_clarification_actor_vectors_pilot.csv \
    --gate-diagnostics outputs/concept_vectors/llama32_1b_ambik_ambiguity_probe_pilot.csv \
    --output "$selection"
}

prepare_pilot_selections() {
  local actor=outputs/concept_vectors/llama32_1b_ambik_clarification_actor_vectors_pilot.pt
  local gate=outputs/concept_vectors/llama32_1b_ambik_ambiguity_probe_pilot.pt
  local selection=outputs/selection/corrective_pilot_layer_selection.json

  "$PYTHON" scripts/prepare_dense_experiment_configs.py dense \
    --root . \
    --source corrective_pilot \
    --vector-path "$actor" \
    --gate-vector-path "$gate" \
    --selection "$selection" \
    --calib-dataset data/raw/ambik/ambik_calib_pilot_select20_paired.csv \
    --test-dataset data/raw/ambik/ambik_calib_pilot_eval20_paired.csv \
    --strength-values -0.20 -0.10 -0.05 0.05 0.10 0.20 \
    --gate-values -1.0 0.0 1.0

  if [[ ! -f outputs/selection/corrective_pilot_strength_selection.json ]]; then
    "$PYTHON" -m clarifysae_llama.runners.sweep \
      --config configs/local/corrective_pilot_sweep_dense_strengths.yaml
    "$PYTHON" scripts/summarize_selective_clarification.py sweep \
      --manifest outputs/sweeps/corrective_pilot_dense_unconditional_strength_sweep/manifest.csv \
      --output-csv outputs/selection/corrective_pilot_strength_ranking.csv \
      --selection-json outputs/selection/corrective_pilot_strength_selection.json \
      --overask-penalty 0.50 \
      --invalid-json-penalty 0.50 \
      --exclude-zero-parameter
  fi

  "$PYTHON" scripts/prepare_dense_experiment_configs.py dense \
    --root . \
    --source corrective_pilot \
    --vector-path "$actor" \
    --gate-vector-path "$gate" \
    --selection "$selection" \
    --strength-selection outputs/selection/corrective_pilot_strength_selection.json \
    --calib-dataset data/raw/ambik/ambik_calib_pilot_select20_paired.csv \
    --test-dataset data/raw/ambik/ambik_calib_pilot_eval20_paired.csv \
    --strength-values -0.20 -0.10 -0.05 0.05 0.10 0.20 \
    --gate-values -1.0 0.0 1.0

  if [[ ! -f outputs/selection/corrective_pilot_gate_selection.json ]]; then
    "$PYTHON" -m clarifysae_llama.runners.sweep \
      --config configs/local/corrective_pilot_sweep_gate_thresholds.yaml
    "$PYTHON" scripts/summarize_selective_clarification.py sweep \
      --manifest outputs/sweeps/corrective_pilot_dense_gate_threshold_sweep/manifest.csv \
      --output-csv outputs/selection/corrective_pilot_gate_ranking.csv \
      --selection-json outputs/selection/corrective_pilot_gate_selection.json \
      --overask-penalty 0.50 \
      --invalid-json-penalty 0.50
  fi
}

run_pilot() {
  prepare_common
  prepare_corrective_representations
  prepare_pilot_selections

  local out=outputs/selective_corrective_pilot
  "$PYTHON" scripts/build_selective_controls.py \
    --prefix pilot \
    --dataset data/raw/ambik/ambik_calib_pilot_eval20_paired.csv \
    --vector-path outputs/concept_vectors/llama32_1b_ambik_clarification_actor_vectors_pilot.pt \
    --gate-vector-path outputs/concept_vectors/llama32_1b_ambik_ambiguity_probe_pilot.pt \
    --selection outputs/selection/corrective_pilot_layer_selection.json \
    --strength-selection outputs/selection/corrective_pilot_strength_selection.json \
    --gate-selection outputs/selection/corrective_pilot_gate_selection.json \
    --output-root "$out"
  run_config_manifest "$out/manifest.csv"
  "$PYTHON" scripts/compare_selective_arms.py \
    --manifest "$out/manifest.csv" \
    --output-dir "$out/comparison" \
    --bootstrap-samples 500 \
    --seed 42
  "$PYTHON" scripts/assess_corrective_pilot.py
  package_results "$out" pilot
}

run_smoke() {
  prepare_common
  prepare_corrective_representations
  local out=outputs/selective_corrective_smoke
  "$PYTHON" scripts/build_selective_controls.py \
    --prefix smoke_v2 \
    --dataset data/raw/ambik/ambik_calib_smoke_paired.csv \
    --vector-path outputs/concept_vectors/llama32_1b_ambik_clarification_actor_vectors_pilot.pt \
    --gate-vector-path outputs/concept_vectors/llama32_1b_ambik_ambiguity_probe_pilot.pt \
    --selection outputs/selection/corrective_pilot_layer_selection.json \
    --strength 0.10 \
    --gate-threshold 0.0 \
    --output-root "$out"
  run_config_manifest "$out/manifest.csv"
  "$PYTHON" scripts/compare_selective_arms.py \
    --manifest "$out/manifest.csv" \
    --output-dir "$out/comparison" \
    --bootstrap-samples 100 \
    --seed 42
  package_results "$out" smoke
}

run_main() {
  prepare_common
  require_file outputs/concept_vectors/llama32_1b_ambik_clarification_actor_vectors_pilot.pt
  require_file outputs/concept_vectors/llama32_1b_ambik_ambiguity_probe_pilot.pt
  require_file outputs/selection/corrective_pilot_layer_selection.json
  require_file outputs/selection/corrective_pilot_strength_selection.json
  require_file outputs/selection/corrective_pilot_gate_selection.json
  require_file outputs/selective_corrective_pilot/comparison/pilot_assessment.json
  if [[ "${FORCE_MAIN:-0}" != "1" ]]; then
    "$PYTHON" scripts/assess_corrective_pilot.py --require-go
  else
    echo "WARNING: FORCE_MAIN=1 bypasses the corrective-pilot go/no-go check." >&2
  fi

  local out=outputs/selective_main_comparison
  "$PYTHON" scripts/build_selective_controls.py \
    --prefix main \
    --dataset data/raw/ambik/ambik_test_eval100_paired.csv \
    --vector-path outputs/concept_vectors/llama32_1b_ambik_clarification_actor_vectors_pilot.pt \
    --gate-vector-path outputs/concept_vectors/llama32_1b_ambik_ambiguity_probe_pilot.pt \
    --selection outputs/selection/corrective_pilot_layer_selection.json \
    --strength-selection outputs/selection/corrective_pilot_strength_selection.json \
    --gate-selection outputs/selection/corrective_pilot_gate_selection.json \
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
  pilot) run_pilot ;;
  main) run_main ;;
  *)
    echo "Usage: $0 {smoke|pilot|main}" >&2
    exit 2
    ;;
esac
