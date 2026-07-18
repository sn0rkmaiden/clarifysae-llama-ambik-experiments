# Run the dense clarification experiments without editing YAML

All generated configurations are created automatically under `configs/local/`.
You only need to place the two AmbiK CSV files in the expected directory and run
shell commands.

## Required data

```text
data/raw/ambik/ambik_calib_100.csv
data/raw/ambik/ambik_test_400.csv
```

The CSVs must contain `environment_full`, `ambiguity_type`, `ambiguous_task`,
`unambiguous_direct`, `question`, and `answer`. The benchmark-derived vector arm
also requires `plan_for_clear_task`.

## Recommended command sequence

Run these one at a time. Each command can be restarted after a completed stage.

```bash
bash scripts/run_dense_clarification_pipeline.sh setup
bash scripts/run_dense_clarification_pipeline.sh prepare
bash scripts/run_dense_clarification_pipeline.sh preflight
bash scripts/run_dense_clarification_pipeline.sh baseline-calib
bash scripts/run_dense_clarification_pipeline.sh smoke
bash scripts/run_dense_clarification_pipeline.sh full-vectors
bash scripts/run_dense_clarification_pipeline.sh dense-sweep
bash scripts/run_dense_clarification_pipeline.sh gate-sweep
bash scripts/run_dense_clarification_pipeline.sh final-test
bash scripts/run_dense_clarification_pipeline.sh ambik-vectors
bash scripts/run_dense_clarification_pipeline.sh ambik-final-test
bash scripts/run_dense_clarification_pipeline.sh project-sae
bash scripts/run_dense_clarification_pipeline.sh summarize
```

The first five stages can be started as one command:

```bash
bash scripts/run_dense_clarification_pipeline.sh phase1
```

The full synthetic vector and calibration sweeps can be started as:

```bash
bash scripts/run_dense_clarification_pipeline.sh phase2
```

`all` runs every stage, including final held-out evaluations. It can take a long
time and should be used only when the smoke run has been inspected:

```bash
bash scripts/run_dense_clarification_pipeline.sh all
```

## What each stage tests

| Command | Scientific question |
|---|---|
| `baseline-calib` | How often does the unsteered model ask on ambiguous and paired clear prompts? |
| `smoke` | Does corpus generation and vector extraction work, and are the concepts decodable on held-out topics? |
| `full-vectors` | Can matched synthetic counterfactuals produce stable ambiguity, ask-trajectory, targeting, and restraint representations? |
| `dense-sweep` | Does the response-side `ask_trajectory` direction causally change question asking, and which sign/strength is useful? |
| `gate-sweep` | Does the prompt-side ambiguity probe reduce unnecessary questions while preserving useful clarification? |
| `final-test` | Do the selected synthetic-vector settings transfer to the held-out AmbiK test set without selecting on it? |
| `ambik-vectors` | Do benchmark-derived vectors reproduce the synthetic-corpus result without synthetic-prose dependence? |
| `ambik-final-test` | Do benchmark-derived vectors transfer to the source examples held out from their internal train/dev split? |
| `project-sae` | Can one SAE feature reproduce the successful dense direction, or are 2-32 signed features required? |

## Automatic model selection

The script makes three development-only selections and records every ranking:

1. **Layer:** maximize the weaker of held-out AUROC for
   `ambiguity_state/ridge_probe` and `ask_trajectory/paired_difference`.
2. **Steering strength:** maximize
   `resolved_proxy_any_ambiguous - 0.50 * overasking_rate_clear - 0.25 * invalid_json_rate`.
3. **Gate threshold:** use the same utility after fixing the selected strength.

Change the penalties without editing files:

```bash
OVERASK_PENALTY=1.0 INVALID_JSON_PENALTY=0.5 \
  bash scripts/run_dense_clarification_pipeline.sh dense-sweep

OVERASK_PENALTY=1.0 INVALID_JSON_PENALTY=0.5 \
  bash scripts/run_dense_clarification_pipeline.sh gate-sweep
```

Selection artifacts are stored in `outputs/selection/`. Final test configurations
are generated only from those development selections.

## Main files to inspect

```text
outputs/selection/synthetic_layer_selection.csv
outputs/selection/synthetic_strength_ranking.csv
outputs/selection/synthetic_gate_ranking.csv
outputs/selection/*_summary.json
outputs/sweeps/*/aggregate_summary.csv
outputs/sweeps/*/category_summary.csv
outputs/sweeps/*/predictions/
```

Use the status command at any time:

```bash
bash scripts/run_dense_clarification_pipeline.sh status
```
