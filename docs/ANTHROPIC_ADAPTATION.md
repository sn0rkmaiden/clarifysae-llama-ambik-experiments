# Adapting self-generated concept vectors to clarification

This branch adds a dense concept-vector route alongside ClarifySAE. It is designed to test the user's hypothesis while avoiding the main confound in a naïve corpus of “texts containing questions”: such a corpus mostly identifies interrogative syntax, punctuation, and dialogue-turn structure.

## What Anthropic's emotion procedure corresponds to here

The closest analogue is **not an SAE feature search**. It is:

1. Ask the model under study to generate many labeled examples in which a concept is clearly expressed.
2. Run those examples back through the model and pool residual-stream activations.
3. Form a dense concept direction by subtracting a comparison-class mean.
4. Optionally project out high-variance components estimated from neutral text.
5. Add the dense vector to a residual stream and measure causal effects.

For clarification, a single label is insufficient because the behavior is a policy with several separable computations. The corpus generator therefore creates matched counterfactuals for four concepts:

- `ambiguity_state`: the prompt lacks a task-relevant variable versus the minimally edited clear prompt;
- `ask_policy`: ask the targeted question versus silently guess under the same ambiguous prompt;
- `targeted_question`: ask for the gold missing slot versus ask a generic question;
- `restraint_on_clear`: answer a clear instruction directly versus ask an unnecessary question.

Each example has a `pair_id`, so the recommended estimator is matched-pair difference-in-means rather than a broad unpaired comparison.

## Commands

Generate self-authored, matched scenarios:

```bash
python -m clarifysae_llama.runners.generate_synthetic_corpus \
  --config configs/synthetic/generate_clarification_counterfactuals.yaml
```

Extract vectors at several layers:

```bash
python -m clarifysae_llama.runners.extract_concept_vectors \
  --config configs/concept_vectors/extract_clarification_vectors.yaml
```

Run ambiguity-probe-gated dense steering:

```bash
python -m clarifysae_llama.runners.run_eval \
  --config configs/steering/dense_vector_probe_gated.yaml
```

Project a dense vector into a signed sparse set of SAE decoder features:

```bash
python -m clarifysae_llama.runners.project_concept_vector_to_sae \
  --config configs/sae_projection/project_dense_vector_to_sae.yaml
```

The last command emits a CSV of selected features and a YAML steering snippet. This provides a direct test of “one dense vector versus many SAE features.” Negative coefficients are allowed because a dense concept direction generally cannot be represented by only positive activation of one SAE feature.

## Recommended experimental matrix

Run the following in parallel on the same train/dev/test splits:

| Arm | Discovery | Intervention | Main purpose |
|---|---|---|---|
| A | Existing ClarifyScore + OutputScore | single SAE feature | reproduce current paper |
| B | Existing discovery | clustered/multi SAE features | current multi-feature ablation |
| C | Self-generated matched corpus | dense paired-difference vector | closest Anthropic analogue |
| D | Benchmark matched pairs | dense CAA vector | removes generator-model artifacts |
| E | Ridge ambiguity probe + ask-policy vector | conditional dense steering | reduce clear-input false positives |
| F | Dense vector projected to SAE | signed multi-feature SAE steering | compare sparse approximation with dense direction |
| G | Supervised vector in SAE latent space | SAE-SSV-style intervention | stronger supervised sparse baseline |
| H | Low-rank representation fine-tuning | ReFT/LoReFT | learned intervention baseline |
| I | Prompt / CLAM / small LoRA | input/post-training baselines | practical upper/lower comparisons |

Do not select the best arm on the final benchmark. Use a development split for layer, strength, threshold, and feature count.

## Pooling and timing

Use different token locations for different concepts:

- `ambiguity_state`: final prompt token before the assistant starts;
- `ask_policy`: final prompt token and early assistant positions should both be tested;
- `targeted_question`: mean over the assistant question span;
- `restraint_on_clear`: final prompt token or early response span.

The synthetic corpus records a recommended pooling mode for each row. The extractor also supports `mean_after_token`, which can imitate the story-level procedure more literally, but it is not necessarily the scientifically appropriate choice for an online decision policy.

## Conditional steering

A static ask vector is expected to increase questions on both ambiguous and clear prompts. The dense steerer can therefore use a separate `ambiguity_state` ridge-probe vector as a gate:

```yaml
steering:
  method: dense_vector
  vector_key: ask_policy__paired_difference__model_layers_10
  gate:
    enabled: true
    vector_key: ambiguity_state__ridge_probe__model_layers_10
    mode: sigmoid
```

The gate is computed on the last prefill token and cached for autoregressive generation. This is an experimental mechanism, not a claim that linear gating is optimal.

## Evaluation that must remain separate

**When to ask**

- AUROC/AUPRC of ambiguity probe;
- clarification true-positive rate on ambiguous inputs;
- unnecessary-question false-positive rate on paired clear inputs;
- selective risk/coverage and calibration.

**What to ask**

- gold-slot alignment;
- minimality (one question and one missing variable);
- relevance and answerability;
- information gain or reduction in valid interpretations;
- human judgments of specificity, naturalness, and actionability.

**End to end**

- task success after the provider/user answers;
- repeated-question and role-play artifacts;
- fluency/capability regressions;
- full strength curves and Pareto frontiers, not only the best point.

## Necessary controls

1. Generate an independent corpus with another writer model, and compare vector cosine and causal effects.
2. Include benchmark-authored matched pairs so the result is not circularly defined by the model's own prose.
3. Lexically match positive and negative texts; include generic questions as hard negatives.
4. Hold out ambiguity categories and domains during vector discovery.
5. Repeat across layers, random seeds, SAE checkpoints, and model sizes.
6. Test both vector signs. A concept's causal direction can be opposite to the intuitive label depending on pooling and layer.
7. Report all tried configurations or use a preregistered development-selection procedure.

## Interpretation caution

A dense concept vector is a direction in the residual stream, not evidence for a single localized “clarification neuron.” An SAE projection is an approximation in a learned dictionary. If a 16-feature signed reconstruction works better than every single feature, that supports a distributed representation; it does not prove the SAE features are independently causal or monosemantic.
