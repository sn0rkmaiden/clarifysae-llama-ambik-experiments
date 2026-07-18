# ClarifySAE: research and implementation plan for self-generated concept vectors

## Executive recommendation

The self-generated-text hypothesis is worth testing, but the most informative version is **not** “generate texts that contain clarifying questions and search for features that activate on them.” That design will strongly confound the desired behavior with question marks, wh-words, modal question syntax, dialogue-turn boundaries, politeness, and generic uncertainty language.

The recommended design is a **factorized, matched-counterfactual representation study**:

1. extract a representation of **underspecification / missing required information** from minimally different ambiguous and clear prompts;
2. extract an **ask-versus-guess response trajectory** vector from the same ambiguous prompt paired with a targeted question or a silent guess;
3. extract a **target-slot question quality** vector from targeted versus generic questions;
4. extract a **restraint-on-clear-inputs** vector from direct answers versus unnecessary questions;
5. use the first representation as a detector/gate and the second or third as the steering intervention.

This directly addresses the paper's central weakness: static clarification steering behaves like a global prior and can increase questions on clear inputs.

## What Anthropic actually did

Anthropic's emotion work extracted **dense residual-stream vectors**, not SAE features. For each emotion, the researchers generated many short stories, averaged residual activations over story tokens and stories, subtracted a mean over comparison emotions, and projected out principal components estimated on neutral transcripts. They then intervened with the resulting dense direction at a selected layer.

Therefore:

- “one emotion vector” means one dense direction with thousands of coordinates;
- it is not equivalent to one SAE latent;
- the study generated a distinct vector per emotion and generally tested one target emotion direction at a time;
- a dense clarification direction may require a signed combination of many SAE decoder directions to approximate it.

## Why clarification is harder than emotion labels

Clarification-seeking combines several computations that can dissociate:

- recognizing that a task has multiple plausible interpretations;
- recognizing that the missing variable is decision-relevant;
- estimating the cost/risk of guessing;
- deciding to ask rather than act or answer;
- selecting the minimal missing slot;
- realizing a grammatical, actionable question;
- stopping after the required information is obtained;
- refraining from asking on sufficiently specified inputs.

A corpus labeled only by the presence of a question cannot isolate these factors.

## Corpus construction

For every topic and underlying scenario, generate this matched set:

1. **Ambiguous prompt** with exactly one consequential missing slot.
2. **Clear prompt**, minimally edited to fill that slot.
3. **Targeted clarification**, asking only for the missing slot.
4. **Silent guess**, answering the ambiguous prompt while selecting one plausible value.
5. **Generic question**, interrogative but not targeting the missing slot.
6. **Direct clear response**, correctly responding to the clear prompt.
7. **Unnecessary question**, asking after the clear prompt.
8. Paraphrases that vary surface form while preserving the same missing variable.

Important controls:

- prohibit the labels “ambiguous,” “clarify,” and “missing information” inside stimuli;
- balance question words and punctuation across targeted and generic-question pairs;
- keep topics, length, register, and instruction wording matched;
- use both self-generated and benchmark/human-authored corpora;
- generate a second corpus with an independent writer model;
- reserve whole topics and ambiguity categories for held-out testing.

## Vector estimators

Test at least three estimators at every candidate layer:

### 1. Anthropic-style difference in class centroids

`mean(positive activations) - mean(negative activations)`.

This is the closest conceptual analogue, although clarification should use carefully chosen comparison classes rather than “all other emotions.”

### 2. Matched-pair difference in means

For each scenario, calculate the positive-minus-negative activation and then average over scenarios. This is equivalent to the core idea behind contrastive activation addition and better cancels topic and wording confounds.

### 3. Regularized linear probe

Fit a ridge or logistic classifier to distinguish the desired state. Use its score for detection/gating; separately test its normal vector for steering. Probe separability alone is not causal evidence, so retain intervention experiments.

### Neutral-PC removal

Generate topic- and style-matched neutral/direct transcripts, find principal components explaining a chosen fraction of neutral variance, and project them out. Treat this as an ablation, not an unquestioned default: removing high-variance components may also remove genuine task state.

## Token positions and layers

Do not use one pooling strategy for every concept.

- **Ambiguity state:** final prompt token before the assistant turn.
- **Ask/guess trajectory:** final prompt token and first few assistant positions.
- **Question content:** mean over the assistant question span.
- **Story-style Anthropic replication:** mean over later tokens, e.g. after token 50, as a separate ablation.

Sweep several middle-to-late layers and choose on a development split. Report both detection and intervention curves by layer.

## Intervention approaches to compare

### Arm A — Existing ClarifySAE

ClarifyScore + OutputScore, single SAE features. This remains the reproduction baseline.

### Arm B — Existing multi-feature ClarifySAE

Clustered SAE features and signed/weighted variants. Report complete cluster and strength sweeps.

### Arm C — Dense self-generated concept vector

The direct Anthropic analogue. Test positive and negative signs and residual-norm-scaled strengths.

### Arm D — Dense benchmark-pair vector

Use AmbiK's paired ambiguous/unambiguous prompts and targeted/guessing responses. This tests whether model-written examples add value beyond benchmark counterfactuals.

### Arm E — Probe-gated dense steering

Compute an ambiguity-state score at the final prompt token. Apply ask-trajectory steering only when the score exceeds a development-calibrated threshold, or scale continuously with a sigmoid. This is the highest-priority intervention.

### Arm F — Dense-to-SAE projection

Rank SAE decoder directions by cosine to the dense vector, fit a signed sparse least-squares reconstruction, and intervene with 1, 2, 4, 8, 16, and 32 features. This answers one-versus-many empirically.

### Arm G — Supervised SAE subspace steering

Train a classifier in SAE latent space, select a small task-relevant subspace, and learn a steering vector restricted to it, in the style of SAE-SSV.

### Arm H — ReFT/LoReFT or rank-one representation finetuning

Learn a small low-rank intervention on hidden states. This is a strong representation-based baseline and may be more effective than fixed directions.

### Arm I — Prompting, CLAM, and small LoRA

These are important practical baselines. A mechanistic intervention does not need to beat all of them to be scientifically useful, but it must be positioned honestly.

### Arm J — Candidate generation and reranking

Generate both a direct response and a clarification candidate, then choose using ambiguity probability, expected information gain, slot alignment, and action risk. This may outperform unconditional activation steering because clarification is a structured decision rather than a style attribute.

## Evaluation

### When to ask

- ambiguity AUROC and AUPRC;
- true-positive clarification rate on ambiguous inputs;
- false-positive question rate on paired clear inputs;
- balanced accuracy and calibration;
- selective risk/coverage;
- performance by preference, commonsense, and safety ambiguity.

### What to ask

- gold-slot alignment;
- minimality and number of questions;
- relevance, answerability, specificity, and actionability;
- semantic information gain / reduction in valid interpretations;
- repetition and role-play artifacts;
- small blinded human evaluation.

### End-to-end value

- task success after a simulated or real user answers;
- number of turns and interaction cost;
- fluency and general-capability degradation;
- safety cost of under-asking and usability cost of over-asking.

### Statistical protocol

- predeclare development selection and final test set;
- report all strength curves and Pareto frontiers, not only best points;
- use multiple generation seeds where sampling is involved;
- paired bootstrap confidence intervals and exact paired tests;
- multiple-comparison correction for broad feature/layer/strength sweeps;
- stability across corpus generator, layer, model seed, SAE checkpoint, and model size.

## Recommended paper reframing

The strongest new paper is broader than “a new vocabulary for SAE feature selection.” A possible framing is:

**Factorized Representation Discovery and Conditional Control of Clarification-Seeking in Language Models**

ClarifySAE becomes one method within a controlled comparison of dense, sparse, and learned interventions.

Suggested research questions:

1. Does a model-self-generated matched corpus identify more causal clarification representations than lexical feature retrieval?
2. Are clarification state, ask trajectory, and question content linearly separable and causally dissociable?
3. Does conditional gating reduce unnecessary questions while preserving gains on ambiguous inputs?
4. Are dense directions better represented by one SAE feature or a signed multi-feature subspace?
5. Which intervention family transfers across datasets, ambiguity types, model families, and scales?

The main scientific claim should be conditional on results. A defensible target claim is not “SAEs solve clarification,” but:

> Clarification-seeking is represented as a partially distributed, factorized policy; separating ambiguity detection from question-generation control improves selectivity, and matched contrastive representations provide a stronger test of causal controllability than lexical feature retrieval alone.

## Implementation included in the modified repository

- self-generated matched-counterfactual corpus generator;
- dense difference-in-means, paired-difference, ridge-probe, and neutral-PCA utilities;
- multi-layer residual activation extractor with concept-specific pooling, held-out diagnostics, and pooling-specific neutral PCA;
- dense residual-vector steering with absolute, current-residual, or recorded-average-residual scaling;
- post-PCA-recalibrated ambiguity-probe hard/sigmoid gating cached once from the prefill decision state;
- dense-to-SAE nested signed sparse projection for 1–32 features;
- example YAML configurations;
- tests for vector extraction and gated steering;
- a detailed usage guide.

The implementation has been compiled and its ten unit tests pass. No GPU-scale corpus generation or benchmark run was performed in this environment, so empirical claims remain to be tested.

## Priority order under limited compute

1. Build 200–500 matched scenarios and extract dense paired vectors at 4–6 layers.
2. Evaluate ambiguity probe and unconditional/gated ask-trajectory steering on paired AmbiK.
3. Compare against existing best single SAE feature, best cluster, explicit prompt, and oracle CLAM.
4. Run dense-to-SAE projections at feature counts 1–32.
5. Transfer the selected development configuration to held-out AmbiK categories and QuestBench.
6. Only then invest in SAE-SSV, ReFT, LoRA, and larger models.

This ordering is much cheaper than rerunning full lexical ClarifyScore over every feature and gives a clear go/no-go signal for the central hypothesis.
