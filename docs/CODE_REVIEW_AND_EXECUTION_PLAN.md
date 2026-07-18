# ClarifySAE: paper strengthening, method adaptation, code review, and execution plan

## Executive verdict

The most promising next paper is **not** simply “ClarifyScore with a better
vocabulary.” The current results and rebuttal already show that static feature
steering acts mainly as a **clarification prior**: it can increase questions on
ambiguous examples, but also increases them on paired clear examples. That is a
useful negative result because it identifies the missing mechanism: selective
clarification requires at least a prompt-side decision about whether information
is needed and a separate response-side mechanism for producing the right
question.

The recommended project is therefore:

> **Factorized Representation Discovery and Conditional Control of
> Clarification-Seeking in Language Models**

Run dense, sparse, and learned interventions in parallel, but make the central
scientific comparison about four separable functions:

1. **ambiguity / missing-variable recognition**;
2. **ask-versus-commit behavior**;
3. **selection of the missing slot and question content**;
4. **restraint on clear inputs and use of the answer afterward**.

The highest-priority experiment is a **prompt-side ambiguity probe used as a
gate for a response-side dense question trajectory**, compared with the current
best single SAE feature, a signed multi-feature SAE reconstruction, prompting,
and CLAM.

## What Anthropic did

Anthropic's emotion study used a dense representation-engineering procedure,
not an SAE feature search:

1. Define 171 emotion concepts.
2. Ask the model under study to write 12 stories for each of 100 topics and each
   emotion, while forbidding the emotion word and direct synonyms.
3. Feed the stories back through the model.
4. At every layer, average residual-stream activations over story positions from
   token 50 onward.
5. For each emotion, subtract the average activation over other emotions.
6. Compute principal components on emotionally neutral transcripts and project
   out enough components to explain 50% of neutral variance.
7. Validate the direction on implicit and controlled prompts.
8. Add the direction to residual activations and measure causal changes in text,
   preferences, and alignment evaluations.

Most headline plots use a mid-late layer around two-thirds through the model.
Some preference experiments apply a layer-specific direction across the same
middle-layer band, with strength expressed as a fraction of average residual
norm.

### One feature or many?

Anthropic generally intervenes with **one dense emotion direction at a time** in
an experiment, but that direction:

- has one coordinate for every hidden dimension;
- is estimated separately at each layer;
- can be applied over multiple layers;
- is not one neuron and is not one SAE latent.

They construct many different emotion vectors and compare them, but they are not
jointly turning on a set of SAE features to represent one emotion. The correct
analogue for your question is to first learn a dense clarification direction,
then test whether an SAE can approximate it with `k = 1, 2, 4, 8, 16, 32`
signed decoder directions.

## Why clarification is not an emotion-like scalar concept

“Anger” can often be detected from broad semantic content. Clarification is a
relation among the user's request, the environment, the agent's action space,
and the cost of a wrong interpretation. It is also conditional: the same phrase
can be clear in one context and unclear in another.

A fixed “question” direction is therefore likely to mix:

- interrogative syntax and punctuation;
- dialogue-turn structure;
- epistemic uncertainty;
- politeness and requests for detail;
- recognition of multiple interpretations;
- risk aversion;
- a general tendency to defer or refuse;
- the actual target slot that must be queried.

The model may linearly encode recognition without automatically acting on it.
Recent work explicitly reports a recognition–behavior gap: models can classify a
question as ambiguous when asked, yet still answer directly in normal QA. This
makes a detector-plus-policy decomposition especially well motivated.

## Factorized concepts and datasets

### 1. `ambiguity_state`

Positive: an instruction lacks exactly one consequential variable. Negative:
the minimally edited instruction fills it. Pool at the final prompt token.

This is the best candidate for a detector or gate. It should be evaluated as a
classifier before any steering claim is made.

### 2. `ask_trajectory`

Positive and negative examples use the exact same ambiguous prompt. The positive
assistant continuation asks the targeted question; the negative continuation
silently commits to a plausible interpretation. Pool over assistant tokens.

This is intentionally called a **trajectory**, not a prompt-side policy state.
Before the response begins, both examples have identical activations. Applying
this direction during generation tests whether a response-derived trajectory is
causally useful; it does not prove that it was already represented at the final
prompt token.

### 3. `targeted_question`

Positive: question about the annotated missing slot. Negative: grammatical but
generic question that does not identify the slot. This controls for the mere
presence of interrogative syntax.

### 4. `restraint_on_clear`

Positive: direct useful response on a clear prompt. Negative: needless question
on that same prompt. This can be used as a separate direction or a negative
regularizer.

### 5. Answer integration

After the user supplies the missing information, evaluate whether the model
updates the plan correctly. This should eventually have a separate representation
study: “uses provided disambiguating evidence” versus “continues with the prior
assumption.” A model that asks but cannot exploit the answer does not solve the
end-to-end task.

## Corpora to build

Use three sources, not one:

1. **Target-self-generated matched corpus.** Closest Anthropic analogue; reveals
   what the target model itself associates with each concept.
2. **Independent-writer matched corpus.** Same validated scenarios for all target
   models; tests whether the direction depends on model-specific prose.
3. **Benchmark-authored corpus.** AmbiK paired prompts, annotated questions, and
   plans; tests synthetic-data circularity.

Hold out entire topics and ambiguity categories. Do not randomly split near-
duplicate paraphrases across train and test.

## Discovery estimators

Run all at the same layers and on the same examples.

### Dense centroid difference

`mean(positive) - mean(negative)`. This is the closest Anthropic analogue.

### Matched-pair difference

Average `h_positive - h_negative` within each scenario. This cancels much of the
topic and wording variation and is the default recommendation.

### Linear probe

Fit ridge/logistic classification on prompt-side `ambiguity_state`. Use its
score for detection/gating. Separately test the normal vector for steering, but
do not infer causality from high AUROC.

### Neutral and nuisance subspace removal

Estimate neutral PCs separately for prompt-final and assistant-span pooling.
Also test projection away from explicitly constructed nuisance directions:

- question mark / wh-word form;
- dialogue boundary / `Assistant:` formatting;
- verbosity and politeness;
- generic uncertainty;
- refusal and deferral.

Neutral projection should be an ablation, not a mandatory truth: a high-variance
component may contain real task state.

### SAE-based discovery

Keep the current ClarifyScore + OutputScore arm. Add:

- outcome correlation with ambiguity labels and slot-alignment outcomes;
- sparse supervised subspace selection (SAE-SSV style);
- signed dense-to-SAE reconstruction;
- side-effect screening using decoder geometry, co-activation, and direct-logit
  footprint when feasible.

## Intervention arms

### A. Current ClarifySAE single feature

Reproduce the selected feature and full strength curve. This is the anchor.

### B. Current clustered/multi-feature SAE steering

Use existing decoder-similarity clusters, but allow signed weights and report
all clusters. Avoid calling a cluster “the clarification mechanism” based only
on one selected maximum.

### C. Dense self-generated matched vector

Inject a layer-specific `ask_trajectory` or `targeted_question` vector. Sweep
both signs and small strengths scaled to recorded average residual norm.

### D. Dense benchmark vector

Repeat with AmbiK-derived pairs. Compare vector cosine and causal effect to the
self-generated direction.

### E. Probe-gated dense steering — highest priority

Compute `ambiguity_state` once at the final prompt token. Use hard or sigmoid
gating to apply the response-side vector only when the score is high. Tune the
threshold on development data and report a risk/coverage or asking/over-asking
frontier.

### F. Multi-layer dense steering

Use the layer-specific direction across a middle-layer band. This is closer to
some Anthropic experiments. Compare against the best single layer at equal total
intervention norm.

### G. Dense-to-SAE projection

Evaluate nested signed supports of 1–32 features. Compare reconstruction cosine
with causal effect. Good reconstruction does not guarantee good steering, so
both are required.

### H. SAE-SSV

Train a linear classifier in SAE latent space, select a small task-relevant
subspace, and learn a supervised direction within it. This tests whether SAEs
need supervised selection rather than lexical retrieval.

### I. ReFT-r1 / LoReFT

Learn a rank-one or low-rank hidden-state intervention. AxBench found simple
representation baselines and rank-one ReFT much stronger than standard SAE
steering in its tested settings. This should be a serious baseline, not an
appendix afterthought.

### J. Prompt, CLAM, and small LoRA

These establish practical performance. A mechanistic method need not win every
metric, but it must clarify its value: training-free control, monitoring,
interpretability, low online overhead, or transfer.

### K. Candidate generation and selection

A non-steering approach may ultimately be best. Generate a direct answer/plan
and a clarification candidate, then choose using:

- calibrated ambiguity probability;
- expected information gain or reduction in valid interpretations;
- predicted action risk/cost;
- slot alignment and answerability;
- a budget on interaction turns.

This treats clarification as decision-making rather than style control. It can
be implemented with a separate detector and reranker, and provides a strong
systems baseline even if the main paper remains mechanistic.

## Evaluation

### Detection / when to ask

- AUROC and AUPRC;
- balanced accuracy and calibration error;
- clarification rate on ambiguous prompts;
- false-positive question rate on paired clear prompts;
- risk–coverage and asking–over-asking curves;
- category results for preference, commonsense, and safety ambiguity.

### Question quality / what to ask

- gold-slot alignment;
- minimality: one question for one missing variable;
- relevance, answerability, specificity, and actionability;
- semantic information gain;
- repetition, unnatural wording, and role-play artifacts;
- blinded human comparison on a preregistered sample.

### End to end

- task success after receiving an answer;
- correct use of the answer;
- turns, tokens, and latency;
- safety cost of under-asking;
- usability cost of over-asking;
- general-capability and fluency degradation.

### Generalization

Use final held-out evaluation on:

- full paired AmbiK test set;
- ClarQ-LLM task types not used for discovery;
- QuestBench, especially Planning-Q and Logic-Q;
- AR-Bench interactive tasks;
- KnowNo or other embodied uncertainty tasks;
- at least one open-domain ambiguity benchmark;
- newer instruction-tuned models, with frontier-model behavior measured rather
  than assumed from older citations.

## Statistical protocol

1. Split discovery, hyperparameter selection, and final evaluation.
2. Select layer, direction, strength, threshold, and `k` only on development.
3. Report the whole sweep and Pareto frontier.
4. Use paired confidence intervals and exact paired tests on fixed examples.
5. Correct broad layer/feature/strength sweeps for multiple comparisons.
6. Use multiple decoding seeds when sampling.
7. Bootstrap at the source-example level, not the individual generated token or
   duplicated pair-row level.
8. Test stability across writer model, corpus seed, target model, layer, model
   size, and SAE checkpoint.
9. Distinguish confirmatory final-test results from exploratory sweeps.

## Paper changes

### Reframe the current result honestly

The rebuttal's unambiguous control should move into the main paper. State that
ClarifySAE is a controllable clarification prior, not an ambiguity detector. The
new work asks whether factorized representations and conditional control can
turn that prior into selective behavior.

### Strengthen the novelty claim

The current combination of ClarifyScore, unchanged OutputScore, and standard SAE
activation addition is vulnerable to the “incremental novelty” criticism. A
stronger contribution is a controlled study of:

- lexical SAE retrieval versus matched self-generated representation discovery;
- recognition versus action versus question targeting;
- unconditional versus gated interventions;
- one SAE feature versus signed sparse subspaces versus dense/learned directions.

### Expand feature analysis

For every selected direction/feature set, show:

- top activating examples on discovery and held-out corpora;
- activation by token position and ambiguity category;
- decoder/logit projections;
- nuisance correlations;
- causal effect by prompt class;
- collateral behavior changes;
- cross-model representational similarity, without assuming feature IDs align.

### Report cost carefully

The rebuttal estimates very large one-time ClarifyScore costs for Gemma: about
63.43 GPU-hours for 2B and 175.72 GPU-hours for 9B on a V100, versus tiny
OutputScore runs. Dense matched-vector extraction can be much cheaper and should
be benchmarked directly, including corpus-generation tokens, extraction GPU
hours, memory, and online latency.

### Claims that would be defensible if supported

A strong target claim is:

> Clarification-seeking is represented as a partially distributed and
> factorized policy. Separating ambiguity recognition from question-generation
> control reduces unnecessary questions, and matched contrastive directions
> provide a stronger causal intervention signal than lexical SAE retrieval
> alone.

Do not claim this before the held-out conditional experiments succeed.

## Code review of the uploaded starting repository

### Repository-copy inconsistency

The starting repository described itself as a compact curated copy and omitted
historical results, visualization code, old configurations, scripts, a patch,
and generated packaging metadata. It was therefore **not** a literal copy of the
previous repository plus new scripts. This is documented clearly now, and a
source-complete overlay archive is produced separately.

The `SOURCE.md` names a Git commit, but the uploaded archive contains no `.git`
history, so that commit cannot be independently verified from the archive alone.

### Critical issues found and fixed

1. **Neutral PCA could never run.** The extractor filtered to target concepts
   before locating neutral rows, and the generator emitted none. Fixed by
   retaining pooling-specific neutral concepts and generating `neutral_prompt`
   and `neutral_response` rows.
2. **Ridge gates were miscalibrated after PCA.** The direction was projected and
   renormalized while retaining the old bias and score scale. Fixed by
   recalibrating threshold, means, and temperature after projection.
3. **Gate caching was wrong for no-cache decoding.** It recomputed whenever
   sequence length exceeded one. Fixed to compute once per generation reset.
4. **Stored probe temperature was ignored.** A configuration default of `1.0`
   always overrode the stored scale. `null` now uses calibrated bundle metadata.
5. **Anthropic norm scaling was not represented.** Added scaling by recorded
   average residual norm.
6. **Only single-layer dense steering existed.** Added a multi-layer wrapper and
   example configuration using layer-specific vectors.
7. **`ask_policy` was semantically overstated.** Renamed to `ask_trajectory` and
   documented why identical prompts cannot produce different prompt-final
   states.
8. **Assistant-span boundary handling was fragile.** Added exact offset mapping
   for fast tokenizers, a safer fallback, and a hard error when the response is
   truncated.
9. **Synthetic validation was too weak.** Added forbidden-label checks,
   question-form checks, minimal-pair similarity, basic slot overlap, hard-
   negative leakage checks, topic-level splits, and failure logs.
10. **No held-out diagnostics.** Extraction now records train and development
    AUROC, balanced accuracy, score margins, and average residual norm.
11. **Dense-to-SAE tested only one `k`.** It now emits nested 1–32 feature
    reconstructions and steering snippets.
12. **No benchmark corpus builder.** Added a paired AmbiK concept-corpus runner.
13. **Direct `pytest` depended on editable installation.** Added pytest source
    path configuration.
14. **No systematic prerequisite check.** Added `scripts/preflight_experiments.py`.

### Verification performed

- all 44 YAML files parse;
- Python `compileall` succeeds for source, scripts, and tests;
- 10 unit tests pass;
- preflight finds no stale `ask_policy` configuration references and no YAML
  errors;
- the environment used for review has CPU PyTorch but no CUDA and lacks the
  heavy optional packages/checkpoints, so no model or SAE end-to-end experiment
  was run.

The preflight currently reports missing raw datasets and generated artifacts, as
expected for an archive that intentionally does not bundle AmbiK, ClarQ-LLM,
processed LMSYS activation data, generated synthetic text, vector bundles, or
model/SAE checkpoints.

## Step-by-step execution

### Step 0 — environment and data

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python scripts/preflight_experiments.py --output outputs/preflight_report.json
pytest -q
```

Place AmbiK and ClarQ-LLM at the paths shown in `README.md`. Build the LMSYS
processed parquet only if reproducing ClarifyScore.

### Step 1 — reproduce anchors

Run unsteered, prompt, CLAM, best current single feature, and best current
multi-feature configuration on the same development subset. Confirm the paper's
metrics and over-asking behavior before adding new methods.

### Step 2 — generate matched synthetic corpus

```bash
python -m clarifysae_llama.runners.generate_synthetic_corpus \
  --config configs/synthetic/generate_clarification_counterfactuals.yaml
```

Manually audit at least 100 scenarios. Reject examples with more than one
missing slot, non-consequential ambiguity, invalid generic negatives, or questions
whose answer is already in context. Save an immutable corpus version and hash.

Repeat with an independent writer model.

### Step 3 — build benchmark-authored corpus

```bash
python -m clarifysae_llama.runners.build_ambik_concept_corpus \
  --config configs/concept_vectors/build_ambik_concept_corpus.yaml
```

Inspect the benchmark hard negative: AmbiK's resolved plan is placed after the
ambiguous prompt as an ask-versus-commit continuation. It is useful but may have
length/style confounds; measure and optionally length-match it.

### Step 4 — extract vectors

```bash
python -m clarifysae_llama.runners.extract_concept_vectors \
  --config configs/concept_vectors/extract_clarification_vectors.yaml

python -m clarifysae_llama.runners.extract_concept_vectors \
  --config configs/concept_vectors/extract_ambik_vectors.yaml
```

Inspect the CSV diagnostics. Stop and redesign the corpus if held-out ambiguity
AUROC is weak, if the sign reverses unpredictably, or if top activating examples
are mostly formatting/question punctuation.

### Step 5 — unconditional dense causal test

Use the gated config with the gate disabled, or the multi-layer config. Sweep
both signs and strengths. The first question is whether the direction causally
changes slot-aligned clarification more than it changes generic question rate.

### Step 6 — gated selective steering

```bash
python -m clarifysae_llama.runners.sweep \
  --config configs/steering/sweep_dense_vector_strengths.yaml

python -m clarifysae_llama.runners.sweep \
  --config configs/steering/sweep_dense_gate_thresholds.yaml
```

Choose a development operating point based on a utility function that penalizes
clear-input questions and rewards correct slot acquisition, not only raw
clarification rate.

### Step 7 — one versus many SAE features

```bash
python -m clarifysae_llama.runners.project_concept_vector_to_sae \
  --config configs/sae_projection/project_dense_vector_to_sae.yaml
```

Run every emitted `k` snippet. Plot causal effect and collateral degradation
against reconstruction cosine and `k`. The best reconstruction may not be the
best intervention.

### Step 8 — learned representation baselines

Implement/run SAE-SSV and ReFT-r1/LoReFT on exactly the same train/dev splits.
Also include a small LoRA and explicit selective-clarification prompt. Match
training examples and report parameter count and compute.

### Step 9 — frozen final test

Freeze the complete selection procedure and evaluate once on held-out AmbiK
pairs, held-out ClarQ task types, and at least one external benchmark. Use paired
statistics and publish all selected hyperparameters.

## Go/no-go criteria

Continue toward the factorized-control paper if at least one conditional method:

- improves slot-aligned question rate on ambiguous examples;
- materially reduces over-asking versus unconditional steering;
- improves end-to-end task success after the answer;
- transfers to held-out topics or a second benchmark;
- does not create severe repetition, refusal, or fluency regressions.

A scientifically valuable negative result is also possible: ambiguity recognition
may be linearly decodable while fixed additive directions fail to implement the
conditional action. In that case, the paper should emphasize the recognition–
control gap and show why learned or decision-level interventions outperform SAE
feature activation.

## Primary references for the new study

- Sofroniew et al. (2026), *Emotion Concepts and their Function in a Large
  Language Model*.
- Wu et al. (2025), *AxBench: Steering LLMs? Even Simple Baselines Outperform
  Sparse Autoencoders*.
- Arad et al. (2025), *SAEs Are Good for Steering—If You Select the Right
  Features*.
- He et al. (2025), *SAE-SSV*.
- Panickssery et al. (2023), *Contrastive Activation Addition*.
- Wu et al. (2024), *ReFT*.
- Li et al. (2025), *QuestBench*.
- Zhou et al. (2025), *AR-Bench*.
- Su and Cardie (2026), *Knowing but Not Showing*.
- Kuhn et al. (2022), *CLAM*.
