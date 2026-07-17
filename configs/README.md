# Configuration templates

- `baselines/`: unsteered AmbiK and CLAM templates.
- `clarq/`: ClarQ-LLM baselines, steered runs, and representative sweeps.
- `discovery/`: ClarifyScore and OutputScore examples for Llama 1B and 8B.
- `steering/`: original SAE sweeps plus probe-gated dense steering.
- `synthetic/`: structured matched-counterfactual corpus generation.
- `concept_vectors/`: multi-layer dense vector and probe extraction.
- `sae_projection/`: sparse signed approximation of a dense vector using SAE
  decoder directions.

Model names, dataset paths, SAE repositories, hookpoints, and GPU settings are
examples and should be checked before each run.
