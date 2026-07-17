from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SteeringConfig:
    sae_repo: str
    hookpoint: str
    feature_indices: list[int]
    strength: float

    # Loader / repo compatibility.
    loader: str = "sparsify"  # "sparsify", "dictionary_learning", or "saelens"
    sae_file: Optional[str] = None
    sae_id: Optional[str] = None
    module_path: Optional[str] = None

    # Steering controls.
    mode: str = "additive"
    apply_to: str = "all_positions"
    steer_generated_tokens_only: bool = False
    normalize_reconstruction: bool = False
    preserve_unsteered_residual: bool = False
    clamp_latents: Optional[float] = None
    log_feature_acts: bool = False
    max_act: Optional[float] = None

    # Multi-feature controls. These mirror the old AmbiK Gemma combine mode:
    # v_S = sum_j weight_j * max_act_j * decoder_j, then h' = h + strength * v_S.
    # For latent-additive mode, weights scale the latent increment for each feature.
    feature_weights: Optional[list[float]] = None
    normalize_each: bool = False
    norm_cap: Optional[float] = None
