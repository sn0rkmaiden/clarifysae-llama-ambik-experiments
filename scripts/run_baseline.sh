#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/baselines/baseline_llama32_1b_instruct_100.yaml}"
python -m clarifysae_llama.runners.run_eval --config "$CONFIG"
