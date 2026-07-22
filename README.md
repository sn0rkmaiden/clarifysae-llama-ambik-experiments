# ClarifySAE experiment repository

This is a **curated** experiment repository derived from `clarifysae-llama-ambik`;
it is not a byte-for-byte copy of the uploaded repository. It keeps the code and
configuration templates needed to run future AmbiK,
ClarQ-LLM, CLAM, SAE-feature, dense concept-vector, conditional-steering, and
dense-to-SAE experiments. Historical predictions, tables, visualization data,
packaging artifacts, and redundant configurations were intentionally omitted.

The original uploaded repository was not modified. A source-complete overlay can
be produced by copying the original tree and applying this repository on top.

## Included experiment families

1. **Unsteered Hugging Face baselines** on AmbiK and ClarQ-LLM.
2. **CLAM baseline** on AmbiK.
3. **Original ClarifySAE pipeline**: ClarifyScore/ReasonScore-style discovery,
   OutputScore filtering, single- and multi-feature SAE steering, and sweeps.
4. **Anthropic-inspired dense concept vectors** from structured matched
   clarification counterfactuals.
5. **Probe-gated dense steering**, separating when-to-ask from how-to-ask.
6. **Dense-to-SAE projection**, testing whether a dense direction is recoverable
   with one or several signed SAE features.

See `docs/ANTHROPIC_ADAPTATION.md` and `docs/RESEARCH_PLAN.md` for the research
motivation and proposed comparisons.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Some model checkpoints require Hugging Face authentication. GPU memory needs
depend on the selected model, dtype, and SAE checkpoint.

## Data layout

Raw datasets are deliberately not bundled. Place them under `data/raw/` or edit
the relevant YAML paths.

Typical locations:

```text
data/raw/ambik/ambik_calib_100.csv
data/raw/ambik/ambik_test_400.csv
data/raw/clarq/English/
data/raw/clarq/Chinese/
```

Generated corpora, vectors, predictions, and metrics go under `outputs/`, which
is git-ignored.

## Main workflows

### A. Original ClarifySAE feature discovery

Representative 1B configuration:

```bash
python -m clarifysae_llama.runners.discover_features \
  --config configs/discovery/1b/discover_llama32_1b_instruct_vocab_layer10resid.yaml
```

Inspect feature scores and compute OutputScore using the corresponding configs
in `configs/discovery/1b/`. Equivalent 8B templates are in
`configs/discovery/8b/`.

### B. SAE steering sweeps

```bash
python -m clarifysae_llama.runners.sweep \
  --config configs/steering/sweep_single_feature_strengths_1b_chosen_layers10_13.yaml
```

The 8B template is:

```bash
python -m clarifysae_llama.runners.sweep \
  --config configs/steering/sweep_single_feature_strengths_8b_instruct_chosen.yaml
```

### C. Generate structured matched clarification data

```bash
python -m clarifysae_llama.runners.generate_synthetic_corpus \
  --config configs/synthetic/generate_clarification_counterfactuals.yaml
```

The bundled YAML uses the target model itself, which is the closest analogue to
Anthropic's self-generated corpus. Also generate a second, independently written
corpus and reuse it across target models as a cross-model control.

Build the same factorized rows directly from paired AmbiK examples:

```bash
python -m clarifysae_llama.runners.build_ambik_concept_corpus \
  --config configs/concept_vectors/build_ambik_concept_corpus.yaml

python -m clarifysae_llama.runners.extract_concept_vectors \
  --config configs/concept_vectors/extract_ambik_vectors.yaml
```

### D. Extract dense clarification vectors

```bash
python -m clarifysae_llama.runners.extract_concept_vectors \
  --config configs/concept_vectors/extract_clarification_vectors.yaml
```

The runner supports matched-pair differences, difference in means, ridge probes,
several pooling strategies, multiple layers, and optional neutral-PC removal.

### E. Evaluate probe-gated dense steering

```bash
python -m clarifysae_llama.runners.run_eval \
  --config configs/steering/dense_vector_probe_gated.yaml
```

This uses an ambiguity-state probe to gate an `ask_trajectory` direction. The
name is deliberate: positive and negative examples share the exact same prompt,
so their representations become different only on assistant response tokens.

### F. Project a dense vector into SAE space

```bash
python -m clarifysae_llama.runners.project_concept_vector_to_sae \
  --config configs/sae_projection/project_dense_vector_to_sae.yaml
```

This fits nested signed sparse combinations of 1, 2, 4, 8, 16, and 32 SAE
decoder directions and reports the full reconstruction curve.


### Preflight before a GPU run

```bash
python scripts/preflight_experiments.py --output outputs/preflight_report.json
```

This validates every YAML file, reports absent raw/generated prerequisites,
checks optional Python packages, and reports CUDA availability.

### Dense strength and gate sweeps

```bash
python -m clarifysae_llama.runners.sweep \
  --config configs/steering/sweep_dense_vector_strengths.yaml

python -m clarifysae_llama.runners.sweep \
  --config configs/steering/sweep_dense_gate_thresholds.yaml
```

For a closer multi-layer analogue to Anthropic's intervention, start from
`configs/steering/dense_vector_multilayer.yaml`.

### G. ClarQ-LLM and CLAM

ClarQ examples:

```bash
python -m clarifysae_llama.runners.run_clarq_eval \
  --config configs/clarq/baseline_clarq_llama32_1b.yaml

python -m clarifysae_llama.runners.run_clarq_eval \
  --config configs/clarq/steer_clarq_llama32_1b.yaml
```

CLAM example:

```bash
python -m clarifysae_llama.runners.run_clam_eval \
  --config configs/baselines/clam_ambik_llama_1b.yaml
```

## Tests

```bash
pytest -q
```

The included unit tests cover concept-vector extraction and recalibration,
neutral-PC projection, synthetic corpus expansion, dense steering, recorded-norm
scaling, and gate caching. They do not replace GPU-scale end-to-end validation.

## What was intentionally left out

See `docs/REPOSITORY_SCOPE.md` for the complete curation policy. In brief, the
copy excludes old generated results, visualization archives, duplicate configs,
notebooks, patch files, and `*.egg-info` build artifacts.

## One-command staged runner

To run the dense clarification study without manually editing YAML files, follow
`docs/RUN_DENSE_CLARIFICATION_EXPERIMENTS.md` or start with:

```bash
bash scripts/run_dense_clarification_pipeline.sh help
```

## Recommended corrective pilot: dense clarification versus SAE steering

The current staged comparison first runs a **corrective pilot**, not the final
100-pair test. It separates the two functions needed for selective
clarification:

- a synthetic **targeted-clarification actor direction**, combining
  targeted-question versus silent-guess and targeted-question versus generic-
  question contrasts;
- a real-AmbiK **ambiguity probe**, trained on 60 matched ambiguous/clear pairs
  and validated on 20 disjoint calibration pairs.

The pilot evaluates 20 further disjoint calibration pairs under six arms:
unsteered baseline, ungated dense actor, probe-gated dense actor, sign-flipped
actor, orthogonal random direction with the same gate, and the frozen Llama-1B
SAE baseline (layer 12, feature 6230, strength -5). It also logs per-example
gate scores, gate weights, applied steering norms, and strict JSON-protocol
validity.

```bash
bash scripts/run_selective_comparison.sh pilot
```

The runner generates 60 synthetic scenario slots, requires at least 80%
acceptance and coverage of all 20 topics, selects a layer using held-out actor
and gate AUROCs, sweeps a small diagnostic strength/gate grid, and writes an
automatic go/no-go report to:

```text
outputs/selective_corrective_pilot/comparison/pilot_assessment.json
```

Only after reviewing a `GO_TO_MAIN` pilot should the held-out 100-pair test be
run:

```bash
bash scripts/run_selective_comparison.sh main
```

The main command refuses to run after a failed pilot unless deliberately
overridden with `FORCE_MAIN=1`.
