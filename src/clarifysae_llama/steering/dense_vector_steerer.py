from __future__ import annotations

"""Hook-based dense residual-stream steering with optional probe gating."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from clarifysae_llama.steering.sparsify_steerer import (
    get_submodule_by_path,
    infer_module_device,
    resolve_module_path,
)


@dataclass
class DenseVectorConfig:
    vector_path: str
    vector_key: str
    hookpoint: str
    strength: float
    module_path: str | None = None
    apply_to: str = "last_position"
    steer_generated_tokens_only: bool = True
    normalize_vector: bool = True
    # absolute | residual_norm_fraction | recorded_residual_norm_fraction
    scale_mode: str = "absolute"
    norm_cap: float | None = None
    gate_enabled: bool = False
    gate_vector_path: str | None = None
    gate_vector_key: str | None = None
    gate_threshold: float = 0.0
    # None means use the score scale stored in the vector bundle.
    gate_temperature: float | None = None
    gate_mode: str = "hard"  # hard | sigmoid | positive_score


def _load_bundle(path: str | Path) -> dict[str, Any]:
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Dense-vector bundle must be a dict, got {type(payload)!r}")
    return payload


def _load_vector_record(path: str, key: str) -> dict[str, Any]:
    payload = _load_bundle(path)
    records = payload.get("vectors", payload)
    if key not in records:
        raise KeyError(f"Vector key {key!r} not found in {path}; available={sorted(records.keys())}")
    record = records[key]
    if isinstance(record, torch.Tensor):
        return {"vector": record}
    if not isinstance(record, dict) or "vector" not in record:
        raise TypeError(f"Vector record {key!r} must be a tensor or dict containing 'vector'")
    return record


class DenseVectorSteerer:
    def __init__(self, model, model_device: torch.device, dtype: torch.dtype, config: DenseVectorConfig):
        self.model = model
        self.model_device = model_device
        self.dtype = dtype
        self.config = config
        self.handle = None
        self._cached_gate_weights: torch.Tensor | None = None
        self.last_gate_scores: torch.Tensor | None = None

        module_path = resolve_module_path(config.hookpoint, config.module_path)
        self.target_module = get_submodule_by_path(self.model, module_path)
        self.target_device = infer_module_device(self.target_module, fallback=model_device)

        steer_record = _load_vector_record(config.vector_path, config.vector_key)
        self.vector = steer_record["vector"].detach().to(dtype=torch.float32).flatten()
        if config.normalize_vector:
            self.vector = self.vector / self.vector.norm().clamp_min(1e-8)
        self.recorded_residual_norm = float(steer_record.get("average_residual_norm", 0.0))

        if config.gate_enabled:
            gate_path = config.gate_vector_path or config.vector_path
            gate_key = config.gate_vector_key or config.vector_key
            gate_record = _load_vector_record(gate_path, gate_key)
            self.gate_vector = gate_record["vector"].detach().to(dtype=torch.float32).flatten()
            self.gate_vector = self.gate_vector / self.gate_vector.norm().clamp_min(1e-8)
            self.gate_bias = float(
                gate_record.get("probe_bias", gate_record.get("score_bias", gate_record.get("bias", 0.0)))
            )
            stored_temperature = float(
                gate_record.get(
                    "probe_temperature",
                    gate_record.get("score_temperature", gate_record.get("score_std", 1.0)),
                )
            )
            self.gate_temperature = (
                stored_temperature if config.gate_temperature is None else float(config.gate_temperature)
            )
            if self.gate_temperature <= 0:
                raise ValueError("gate_temperature must be positive")
        else:
            self.gate_vector = None
            self.gate_bias = 0.0
            self.gate_temperature = 1.0

    def attach(self) -> None:
        if self.handle is None:
            self.handle = self.target_module.register_forward_hook(self._hook_fn)

    def detach(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def reset(self) -> None:
        self._cached_gate_weights = None
        self.last_gate_scores = None

    def _selected_position_mask(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden.shape
        if self.config.apply_to == "all_positions":
            mask = torch.ones((batch_size, seq_len), device=hidden.device, dtype=torch.bool)
        elif self.config.apply_to == "last_position":
            mask = torch.zeros((batch_size, seq_len), device=hidden.device, dtype=torch.bool)
            mask[:, -1] = True
        else:
            raise ValueError(f"Unsupported apply_to mode: {self.config.apply_to}")
        if self.config.steer_generated_tokens_only and seq_len > 1:
            # During prefill this selects the final prompt position, which is the
            # state that produces the first generated token. Subsequent cached
            # decoding calls contain one token and are selected as well.
            generated_mask = torch.zeros_like(mask)
            generated_mask[:, -1] = True
            mask &= generated_mask
        return mask

    def _compute_gate_weights(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, _seq_len, d_model = hidden.shape
        if not self.config.gate_enabled:
            return torch.ones(batch_size, device=hidden.device, dtype=hidden.dtype)

        # Compute exactly once per reset. Recomputing whenever seq_len > 1 makes
        # no-cache generation silently change the gate after every token.
        if self._cached_gate_weights is None:
            if self.gate_vector is None or self.gate_vector.numel() != d_model:
                raise ValueError(
                    f"Gate vector dimension {0 if self.gate_vector is None else self.gate_vector.numel()} "
                    f"does not match model hidden dimension {d_model}"
                )
            gate_vector = self.gate_vector.to(device=hidden.device, dtype=hidden.dtype)
            scores = hidden[:, -1, :] @ gate_vector + float(self.gate_bias)
            self.last_gate_scores = scores.detach().float().cpu()
            centered = scores - float(self.config.gate_threshold)
            mode = self.config.gate_mode.strip().lower().replace("-", "_")
            if mode == "hard":
                weights = (centered > 0).to(hidden.dtype)
            elif mode == "sigmoid":
                weights = torch.sigmoid(centered / max(float(self.gate_temperature), 1e-6))
            elif mode == "positive_score":
                weights = centered.clamp_min(0)
            else:
                raise ValueError(f"Unsupported gate_mode: {self.config.gate_mode}")
            self._cached_gate_weights = weights.detach()

        cached = self._cached_gate_weights.to(device=hidden.device, dtype=hidden.dtype)
        if cached.shape[0] == batch_size:
            return cached
        if batch_size % cached.shape[0] == 0:
            return cached.repeat_interleave(batch_size // cached.shape[0])
        raise ValueError(
            f"Cannot broadcast cached gate batch {cached.shape[0]} to current batch {batch_size}. "
            "Beam expansion must be an integer multiple of the prefill batch."
        )

    def _scales(self, base: torch.Tensor) -> torch.Tensor:
        mode = self.config.scale_mode.strip().lower().replace("-", "_")
        if mode == "absolute":
            return torch.full(
                (base.shape[0], 1),
                float(self.config.strength),
                device=base.device,
                dtype=base.dtype,
            )
        if mode == "residual_norm_fraction":
            return float(self.config.strength) * base.norm(dim=-1, keepdim=True)
        if mode == "recorded_residual_norm_fraction":
            if self.recorded_residual_norm <= 0:
                raise ValueError(
                    "recorded_residual_norm_fraction requires average_residual_norm "
                    "in the vector record; re-run extraction with the updated code"
                )
            return torch.full(
                (base.shape[0], 1),
                float(self.config.strength) * self.recorded_residual_norm,
                device=base.device,
                dtype=base.dtype,
            )
        raise ValueError(f"Unsupported scale_mode: {self.config.scale_mode}")

    @torch.inference_mode()
    def _hook_fn(self, module, inputs, output):
        del module, inputs
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden is None:
            return output
        if hidden.ndim != 3:
            raise ValueError(f"Expected [batch, seq, d_model], got {tuple(hidden.shape)}")
        if self.vector.numel() != hidden.shape[-1]:
            raise ValueError(
                f"Steering vector dimension {self.vector.numel()} does not match hidden dimension {hidden.shape[-1]}"
            )

        selected = self._selected_position_mask(hidden)
        if not bool(selected.any()):
            return output
        gate_weights = self._compute_gate_weights(hidden)
        vector = self.vector.to(device=hidden.device, dtype=hidden.dtype)

        updated = hidden.clone()
        positions = selected.nonzero(as_tuple=False)
        batch_idx = positions[:, 0]
        seq_idx = positions[:, 1]
        base = updated[batch_idx, seq_idx, :]

        scales = self._scales(base) * gate_weights[batch_idx].unsqueeze(-1)
        delta = scales * vector.unsqueeze(0)

        if self.config.norm_cap is not None:
            cap = float(self.config.norm_cap)
            norms = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            delta = delta * torch.clamp(cap / norms, max=1.0)

        updated[batch_idx, seq_idx, :] = base + delta
        if isinstance(output, tuple):
            return (updated,) + output[1:]
        return updated


class MultiLayerDenseVectorSteerer:
    """Attach one dense vector per layer, as in multi-layer activation steering.

    Each child has its own layer-specific vector. For the cleanest Anthropic-style
    replication use unconditional children. Probe-gated experiments are usually
    easier to interpret with a single decision layer, because independently
    gating every layer can yield inconsistent decisions.
    """

    def __init__(
        self,
        model,
        model_device: torch.device,
        dtype: torch.dtype,
        configs: Iterable[DenseVectorConfig],
    ):
        self.steerers = [
            DenseVectorSteerer(model, model_device, dtype, config) for config in configs
        ]
        if not self.steerers:
            raise ValueError("MultiLayerDenseVectorSteerer requires at least one layer config")

    def attach(self) -> None:
        for steerer in self.steerers:
            steerer.attach()

    def detach(self) -> None:
        for steerer in reversed(self.steerers):
            steerer.detach()

    def reset(self) -> None:
        for steerer in self.steerers:
            steerer.reset()
