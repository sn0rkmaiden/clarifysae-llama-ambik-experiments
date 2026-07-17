from __future__ import annotations

from clarifysae_llama.backends.hf_backend import HFCausalBackend
from clarifysae_llama.steering.config import SteeringConfig
from clarifysae_llama.steering.dense_vector_steerer import DenseVectorConfig, DenseVectorSteerer
from clarifysae_llama.steering.sparsify_steerer import SparsifySteerer


class SteeredHFCausalBackend(HFCausalBackend):
    def __init__(self, config: dict):
        super().__init__(config)

        steering_cfg = config["steering"]
        runtime_cfg = steering_cfg.get("runtime", {})
        model_device = next(self.model.parameters()).device
        method = str(steering_cfg.get("method", "sae")).strip().lower().replace("-", "_")

        if method in {"dense", "dense_vector", "concept_vector"}:
            gate_cfg = steering_cfg.get("gate", {})
            self.steering = DenseVectorSteerer(
                model=self.model,
                model_device=model_device,
                dtype=self.dtype,
                config=DenseVectorConfig(
                    vector_path=steering_cfg["vector_path"],
                    vector_key=steering_cfg["vector_key"],
                    hookpoint=steering_cfg["hookpoint"],
                    module_path=steering_cfg.get("module_path"),
                    strength=float(steering_cfg["strength"]),
                    apply_to=steering_cfg.get("apply_to", "last_position"),
                    steer_generated_tokens_only=bool(steering_cfg.get("steer_generated_tokens_only", True)),
                    normalize_vector=bool(steering_cfg.get("normalize_vector", True)),
                    scale_mode=steering_cfg.get("scale_mode", "absolute"),
                    norm_cap=steering_cfg.get("norm_cap"),
                    gate_enabled=bool(gate_cfg.get("enabled", False)),
                    gate_vector_path=gate_cfg.get("vector_path"),
                    gate_vector_key=gate_cfg.get("vector_key"),
                    gate_threshold=float(gate_cfg.get("threshold", 0.0)),
                    gate_temperature=float(gate_cfg.get("temperature", 1.0)),
                    gate_mode=gate_cfg.get("mode", "hard"),
                ),
            )
        elif method in {"sae", "sparse", "custom"}:
            self.steering = SparsifySteerer(
                model=self.model,
                model_device=model_device,
                dtype=self.dtype,
                config=SteeringConfig(
                    sae_repo=steering_cfg["sae_repo"],
                    hookpoint=steering_cfg["hookpoint"],
                    feature_indices=list(steering_cfg["feature_indices"]),
                    strength=float(steering_cfg["strength"]),
                    loader=steering_cfg.get("loader", "sparsify"),
                    sae_file=steering_cfg.get("sae_file"),
                    sae_id=steering_cfg.get("sae_id"),
                    module_path=steering_cfg.get("module_path"),
                    mode=steering_cfg.get("mode", "additive"),
                    apply_to=steering_cfg.get("apply_to", "all_positions"),
                    steer_generated_tokens_only=steering_cfg.get("steer_generated_tokens_only", False),
                    normalize_reconstruction=runtime_cfg.get("normalize_reconstruction", False),
                    preserve_unsteered_residual=runtime_cfg.get("preserve_unsteered_residual", False),
                    clamp_latents=runtime_cfg.get("clamp_latents"),
                    log_feature_acts=runtime_cfg.get("log_feature_acts", False),
                    max_act=steering_cfg.get("max_act", runtime_cfg.get("max_act")),
                    feature_weights=steering_cfg.get("feature_weights"),
                    normalize_each=bool(steering_cfg.get("normalize_each", runtime_cfg.get("normalize_each", False))),
                    norm_cap=steering_cfg.get("norm_cap", runtime_cfg.get("norm_cap")),
                ),
            )
        else:
            raise ValueError(
                f"Unsupported steering.method={method!r}. Expected 'sae' or 'dense_vector'."
            )

    def generate(self, prompt: str) -> str:
        self.steering.reset()
        self.steering.attach()
        try:
            return super().generate(prompt)
        finally:
            self.steering.detach()

    def generate_messages(self, messages: list[dict]) -> str:
        self.steering.reset()
        self.steering.attach()
        try:
            return super().generate_messages(messages)
        finally:
            self.steering.detach()

    def generate_batch(self, prompts: list[str]) -> list[str]:
        self.steering.reset()
        self.steering.attach()
        try:
            return super().generate_batch(prompts)
        finally:
            self.steering.detach()
