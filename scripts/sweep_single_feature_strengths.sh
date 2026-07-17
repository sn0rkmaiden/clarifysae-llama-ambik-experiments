#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/steering/sweep_single_feature_strengths_1b_chosen_layers10_13.yaml}"
python -m clarifysae_llama.runners.sweep --config "$CONFIG"
