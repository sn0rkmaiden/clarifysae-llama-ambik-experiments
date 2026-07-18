# Adapting Anthropic-style concept vectors to clarification

## What is being copied—and what is not

Anthropic's emotion work did **not** search an SAE for a single emotion feature.
It generated concept-rich text, ran that text through the model, averaged
residual-stream activations, formed one dense direction per concept and layer,
removed a neutral nuisance subspace, and tested the direction causally by
activation addition.

The closest clarification analogue in this repository is therefore dense
residual-vector discovery. The SAE experiment is a later projection question:
can that dense direction be represented by one feature, or does it require a
signed set of features?

## Why “clarification” must be factorized

An emotion can be treated approximately as broad semantic content. Clarification
is a conditional decision policy involving at least:

1. recognizing that a required variable is not specified;
2. judging whether the variable matters enough to ask about;
3. choosing to ask rather than guess;
4. selecting the particular missing slot;
5. producing a concise, answerable question;
6. refraining from asking when the prompt is sufficiently specified;
7. using the user's answer correctly afterward.

A corpus of texts that merely contain questions will mostly recover question
marks, wh-words, politeness, formatting, and turn boundaries.

## Corpus emitted by the generator

Every accepted scenario creates matched rows for:

- `ambiguity_state`: ambiguous prompt versus minimally edited clear prompt,
  pooled at the final prompt token;
- `ask_trajectory`: targeted question versus a silent guess under the exact same
  ambiguous prompt, pooled over assistant response tokens;
- `targeted_question`: a slot-specific question versus a generic interrogative;
- `restraint_on_clear`: direct response versus an unnecessary question on the
  clear prompt;
- `neutral_prompt` and `neutral_response`: nuisance corpora for pooling-specific
  PCA removal.

`ask_trajectory` is deliberately not called a pre-response policy state. Because
its positive and negative prompts are identical, their final-prompt hidden state
is identical. The distinction only appears during the assistant continuation.
Applying that response-derived direction during generation is a causal
experiment, but it must not be described as discovering a pre-existing prompt
state without further evidence.

The generator also:

- assigns entire topics to train/dev/test splits;
- enforces minimal-pair similarity and basic hard-negative checks;
- rejects explicit meta-labels such as “ambiguity” and “clarification”;
- emits failure logs for manual audit.

## Extraction

```bash
python -m clarifysae_llama.runners.generate_synthetic_corpus \
  --config configs/synthetic/generate_clarification_counterfactuals.yaml

python -m clarifysae_llama.runners.extract_concept_vectors \
  --config configs/concept_vectors/extract_clarification_vectors.yaml
```

A benchmark-authored control is also implemented:

```bash
python -m clarifysae_llama.runners.build_ambik_concept_corpus \
  --config configs/concept_vectors/build_ambik_concept_corpus.yaml
python -m clarifysae_llama.runners.extract_concept_vectors \
  --config configs/concept_vectors/extract_ambik_vectors.yaml
```

The AmbiK builder uses the annotated question as the positive continuation and
the annotated resolved plan under the ambiguous prompt as the ask-versus-commit
hard negative. This is useful but not perfectly matched stylistically, so report
it as a benchmark control rather than the uniquely correct definition.

Three estimators are implemented at each layer:

- class difference in means;
- matched-pair difference in means;
- ridge-probe direction.

Neutral principal components are estimated separately for `last_nonpad` and
`assistant_mean` representations. After projecting a ridge direction away from
the neutral subspace, its bias and score scale are **recalibrated**; retaining
the pre-projection bias would produce an invalid gate.

The vector bundle records:

- raw and PCA-cleaned vectors;
- average residual norm at each extraction layer;
- calibrated score bias and temperature;
- train and held-out AUROC, balanced accuracy, means, and margins.

## Steering

### Single-layer gated steering

```bash
python -m clarifysae_llama.runners.run_eval \
  --config configs/steering/dense_vector_probe_gated.yaml
```

The ambiguity probe is evaluated once on the prefill decision state and cached.
This matters for no-cache decoding: recomputing the gate on every full-sequence
forward pass would silently turn a prompt decision into a changing token-level
decision.

`recorded_residual_norm_fraction` scales the intervention using the average
residual norm measured during extraction, closer to Anthropic's convention than
using each current token's norm.

### Multi-layer dense steering

```bash
python -m clarifysae_llama.runners.run_eval \
  --config configs/steering/dense_vector_multilayer.yaml
```

This injects the layer-specific version of the same concept direction over a
middle-layer band. Begin with unconditional multi-layer steering; independently
gating every layer can create inconsistent per-layer decisions and should be a
separate ablation.

### Strength and gate sweeps

```bash
python -m clarifysae_llama.runners.sweep \
  --config configs/steering/sweep_dense_vector_strengths.yaml

python -m clarifysae_llama.runners.sweep \
  --config configs/steering/sweep_dense_gate_thresholds.yaml
```

Select sign, layer, strength, and threshold only on a development split. Report
complete curves and clear-input false-positive rates.

## One feature or many?

```bash
python -m clarifysae_llama.runners.project_concept_vector_to_sae \
  --config configs/sae_projection/project_dense_vector_to_sae.yaml
```

The projection runner fits a broad signed SAE decoder basis, then refits nested
supports of 1, 2, 4, 8, 16, and 32 features. It emits a reconstruction curve and
one steering YAML per feature count. Negative coefficients are allowed: a dense
concept direction need not lie in the positive cone of SAE decoder features.

The scientific comparison is:

1. best individual SAE feature;
2. existing decoder-similarity cluster;
3. signed dense-to-SAE reconstruction at each `k`;
4. the original dense direction.

A better multi-feature reconstruction supports a distributed representation; it
does not establish that the individual SAE features are independently
monosemantic or causal.

## Required controls

- **Target-self-generated corpus:** closest Anthropic analogue.
- **Independent writer corpus:** detects circularity and model-specific prose.
- **Benchmark-authored pairs:** detects synthetic-generator artifacts.
- **Hard negatives:** guessing, generic questions, needless questions, and
  question-form-matched controls.
- **Held-out topics/categories:** prevent topic memorization.
- **Both signs and all curves:** avoid cherry-picking.
- **Prompt-side versus response-side positions:** distinguish recognition,
  decision, and realization.
- **Clear-input controls:** measure over-asking directly.

## Interpretation boundary

A useful ambiguity probe does not imply that its normal vector is the best
steering direction. A steerable question trajectory does not imply that the
model has a single “clarification concept.” Treat detection, causal control,
question targeting, and downstream task use as separate hypotheses.
