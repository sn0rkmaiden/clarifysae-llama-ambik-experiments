from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from clarifysae_llama.discovery.sae_utils import (
    SparseLatents,
    encode_dense,
    encode_sparse,
    get_decoder_matrix,
    get_num_latents,
    sparse_to_dense,
)
from clarifysae_llama.steering.config import SteeringConfig


def get_submodule_by_path(root_module, path: str):
    current = root_module
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def normalize_hookpoint_to_module_path(hookpoint: str) -> str:
    hp = hookpoint.strip()

    if hp == "embed_tokens":
        return "model.embed_tokens"

    if hp == "model.embed_tokens":
        return hp

    if hp.startswith("model."):
        return hp

    if hp.startswith("layers."):
        return f"model.{hp}"

    # SAE Lens / TransformerLens-style residual-stream hook names used in the
    # older Gemma ClarQ repo, e.g. blocks.12.hook_resid_post.
    if hp.startswith("blocks.") and hp.endswith(".hook_resid_post"):
        parts = hp.split(".")
        if len(parts) >= 3 and parts[1].isdigit():
            return f"model.layers.{parts[1]}"

    raise ValueError(f"Unsupported hookpoint/module path: {hookpoint}")


def resolve_module_path(hookpoint: str, module_path: str | None = None) -> str:
    return module_path or normalize_hookpoint_to_module_path(hookpoint)


def infer_module_device(module, fallback: torch.device) -> torch.device:
    for tensor in module.parameters():
        if tensor.device.type != "meta":
            return tensor.device
    for tensor in module.buffers():
        if tensor.device.type != "meta":
            return tensor.device
    return fallback


def move_sae_to_device_dtype(sae: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if hasattr(sae, "to"):
        try:
            sae = sae.to(device=device, dtype=dtype)
        except TypeError:
            sae = sae.to(device=device)
    if hasattr(sae, "eval"):
        sae.eval()
    return sae


def _cuda_device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return torch.cuda.current_device()


def _device_supports_sparsify_cuda(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(_cuda_device_index(device))
    return major >= 8


def _resolve_sae_runtime(loader: str, target_device: torch.device, model_dtype: torch.dtype) -> tuple[torch.device, torch.dtype]:
    loader_name = str(loader).strip().lower().replace("-", "_")
    sae_device = target_device
    sae_dtype = model_dtype

    if loader_name == "sparsify" and target_device.type == "cuda" and not _device_supports_sparsify_cuda(target_device):
        major, minor = torch.cuda.get_device_capability(_cuda_device_index(target_device))
        print(
            "[WARN] Using a 'sparsify' SAE on CUDA sm_"
            f"{major}{minor} during steering can trigger Triton/xFormers CUDA failures. "
            "Running the SAE on CPU in float32 for this run."
        )
        sae_device = torch.device("cpu")
        sae_dtype = torch.float32

    return sae_device, sae_dtype


def _load_saelens_sae(*, sae_repo: str, sae_id: str, device: torch.device, dtype: torch.dtype):
    try:
        from sae_lens import SAE
    except ImportError as exc:
        raise ImportError(
            "sae_lens is required for steering.loader='saelens'. Install it with: pip install sae-lens"
        ) from exc

    loaded = SAE.from_pretrained(release=sae_repo, sae_id=sae_id, device=str(device))
    if isinstance(loaded, tuple):
        sae = loaded[0]
    else:
        sae = loaded
    return move_sae_to_device_dtype(sae, device=device, dtype=dtype)


def load_sae(
    *,
    loader: str,
    sae_repo: str,
    hookpoint: str,
    sae_file: str | None,
    sae_id: str | None,
    device: torch.device,
    dtype: torch.dtype,
):
    loader_name = str(loader).strip().lower().replace("-", "_")

    if loader_name == "sparsify":
        try:
            from sparsify import Sae
        except ImportError as exc:
            raise ImportError(
                "sparsify is required for steering.loader='sparsify'. Install it with: pip install eai-sparsify"
            ) from exc
        sae = Sae.load_from_hub(sae_repo, hookpoint=hookpoint)
        return move_sae_to_device_dtype(sae, device=device, dtype=dtype)

    if loader_name == "dictionary_learning":
        if not sae_file:
            raise ValueError(
                "steering.sae_file is required when steering.loader='dictionary_learning'."
            )
        try:
            from dictionary_learning import utils as dl_utils
        except ImportError as exc:
            raise ImportError(
                "dictionary_learning is required for steering.loader='dictionary_learning'. "
                "Install it with: pip install dictionary-learning"
            ) from exc

        if sae_file.endswith(".pt"):
            trainer_subdir = str(Path(sae_file).parent)
        else:
            trainer_subdir = sae_file.rstrip("/")

        snapshot_root = snapshot_download(
            repo_id=sae_repo,
            allow_patterns=[f"{trainer_subdir}/*"],
        )
        trainer_dir = Path(snapshot_root) / trainer_subdir

        sae, _config = dl_utils.load_dictionary(str(trainer_dir), device=str(device))
        return move_sae_to_device_dtype(sae, device=device, dtype=dtype)

    if loader_name == "saelens":
        if not sae_id:
            raise ValueError("steering.sae_id is required when steering.loader='saelens'.")
        return _load_saelens_sae(sae_repo=sae_repo, sae_id=sae_id, device=device, dtype=dtype)

    raise ValueError(
        f"Unsupported SAE loader '{loader}'. Expected 'sparsify', 'dictionary_learning', or 'saelens'."
    )


class SparsifySteerer:
    def __init__(self, model, model_device: torch.device, dtype: torch.dtype, config: SteeringConfig):
        self.model = model
        self.model_device = model_device
        self.dtype = dtype
        self.config = config
        self.handle = None
        self.last_feature_stats: dict[str, Any] | None = None
        self._cached_feature_max_acts: dict[int, float] | None = None

        module_path = resolve_module_path(config.hookpoint, config.module_path)
        self.target_module = get_submodule_by_path(self.model, module_path)
        target_module_device = infer_module_device(self.target_module, fallback=self.model_device)
        self.sae_device, self.sae_dtype = _resolve_sae_runtime(
            loader=config.loader,
            target_device=target_module_device,
            model_dtype=self.dtype,
        )

        self.sae = load_sae(
            loader=config.loader,
            sae_repo=config.sae_repo,
            hookpoint=config.hookpoint,
            sae_file=config.sae_file,
            sae_id=config.sae_id,
            device=self.sae_device,
            dtype=self.sae_dtype,
        )
        self._validate_feature_indices()
        self._validate_feature_weights()

    def _validate_feature_indices(self) -> None:
        if not self.config.feature_indices:
            raise ValueError("steering.feature_indices must contain at least one feature.")
        num_latents = int(get_num_latents(self.sae))
        invalid = [int(idx) for idx in self.config.feature_indices if int(idx) < 0 or int(idx) >= num_latents]
        if invalid:
            raise ValueError(
                f"Feature indices out of bounds for SAE with {num_latents} latents: {invalid[:10]}"
            )

    def _validate_feature_weights(self) -> None:
        if self.config.feature_weights is None:
            return
        if len(self.config.feature_weights) != len(self.config.feature_indices):
            raise ValueError(
                "steering.feature_weights must have the same length as steering.feature_indices "
                f"({len(self.config.feature_weights)} != {len(self.config.feature_indices)})."
            )

    def _feature_weight(self, feature_idx: int) -> float:
        if self.config.feature_weights is None:
            return 1.0
        for idx, weight in zip(self.config.feature_indices, self.config.feature_weights):
            if int(idx) == int(feature_idx):
                return float(weight)
        return 1.0

    def attach(self) -> None:
        if self.handle is None:
            self.handle = self.target_module.register_forward_hook(self._hook_fn)

    def detach(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def reset(self) -> None:
        self.last_feature_stats = None
        self._cached_feature_max_acts = None

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
            generated_mask = torch.zeros((batch_size, seq_len), device=hidden.device, dtype=torch.bool)
            generated_mask[:, -1] = True
            mask &= generated_mask

        return mask.reshape(-1)

    @staticmethod
    def _normalize_reconstruction(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_norm = target.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        recon_norm = reconstruction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return reconstruction * (target_norm / recon_norm)

    def _manual_decode_from_sparse(self, sparse: SparseLatents, *, dtype: torch.dtype) -> torch.Tensor:
        decoder = get_decoder_matrix(self.sae)
        if decoder.ndim != 2:
            raise ValueError(f"Expected decoder matrix with 2 dims, got shape {tuple(decoder.shape)}")

        acts = sparse.top_acts.to(device=decoder.device, dtype=decoder.dtype)
        indices = sparse.top_indices.to(device=decoder.device, dtype=torch.long)

        flat_indices = indices.reshape(-1)
        selected_rows = decoder.index_select(0, flat_indices)
        selected_rows = selected_rows.reshape(*indices.shape, decoder.shape[-1])
        recon = (selected_rows * acts.unsqueeze(-1)).sum(dim=-2)
        return recon.to(device=acts.device, dtype=dtype)

    def _decode_from_sparse(self, sparse: SparseLatents, *, dtype: torch.dtype) -> torch.Tensor:
        use_manual = self.sae_device.type != "cuda" or sparse.top_acts.device.type != "cuda"
        if use_manual:
            return self._manual_decode_from_sparse(sparse, dtype=dtype)

        try:
            return self.sae.decode(sparse.top_acts, sparse.top_indices)
        except TypeError:
            dense = sparse_to_dense(
                sparse,
                num_latents=get_num_latents(self.sae),
                dtype=dtype,
            )
            return self.sae.decode(dense)
        except (RuntimeError, ValueError) as exc:
            message = str(exc).lower()
            if "triton" in message or "cpu tensor" in message or "illegal memory access" in message:
                return self._manual_decode_from_sparse(sparse, dtype=dtype)
            raise

    def _feature_max_acts(self, hidden_2d: torch.Tensor) -> dict[int, float]:
        if self.config.max_act is not None:
            shared = float(self.config.max_act)
            return {int(feature_idx): shared for feature_idx in self.config.feature_indices}

        dense_latents = encode_dense(self.sae, hidden_2d)
        if dense_latents.ndim == 1:
            dense_latents = dense_latents.unsqueeze(0)

        feature_max_acts: dict[int, float] = {}
        for feature_idx in self.config.feature_indices:
            feature_idx = int(feature_idx)
            max_val = float(dense_latents[:, feature_idx].max().item())
            if max_val <= 0 or max_val == float("inf"):
                max_val = 1.0
            feature_max_acts[feature_idx] = max_val
        return feature_max_acts

    def _build_decoder_vector(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self._cached_feature_max_acts is None:
            raise RuntimeError("Feature max activations must be computed before decoder-vector steering.")

        decoder = get_decoder_matrix(self.sae).to(device=self.sae_device, dtype=self.sae_dtype)
        vector = torch.zeros(decoder.shape[-1], device=self.sae_device, dtype=self.sae_dtype)

        for feature_idx in self.config.feature_indices:
            feature_idx = int(feature_idx)
            max_act = float(self._cached_feature_max_acts[feature_idx])
            weight = self._feature_weight(feature_idx)
            direction = decoder[feature_idx]
            if self.config.normalize_each:
                direction = direction / direction.norm().clamp_min(1e-8)
            vector = vector + direction * (max_act * weight)

        # Match the old AmbiK generate_with_multi_feature_steering(...) semantics:
        #   combined = sum_j max_act_j * weight_j * W_dec[j]
        #   if norm_cap is not None: combined = cap_norm(combined)
        #   h' = h + strength * combined
        # In particular, norm_cap applies to the pre-strength combined vector.
        if self.config.norm_cap is not None:
            cap = float(self.config.norm_cap)
            norm = vector.norm().clamp_min(1e-8)
            if float(norm.item()) > cap:
                vector = vector * (cap / norm)

        vector = vector * float(self.config.strength)
        return vector.to(device=device, dtype=dtype)

    @torch.inference_mode()
    def _hook_decoder_vector(self, hidden: torch.Tensor):
        original_device = hidden.device
        original_dtype = hidden.dtype
        selected_mask = self._selected_position_mask(hidden)
        if not bool(selected_mask.any()):
            return None

        hidden_2d = hidden.reshape(-1, hidden.shape[-1]).to(device=self.sae_device, dtype=self.sae_dtype)
        if self._cached_feature_max_acts is None:
            self._cached_feature_max_acts = self._feature_max_acts(hidden_2d)

        decoder_vector = self._build_decoder_vector(dtype=hidden_2d.dtype, device=hidden_2d.device)
        updated_hidden = hidden_2d.clone()
        updated_hidden[selected_mask.to(device=self.sae_device)] += decoder_vector.unsqueeze(0)
        return updated_hidden.reshape(hidden.shape).to(device=original_device, dtype=original_dtype)

    @torch.inference_mode()
    def _hook_latent_additive(self, hidden: torch.Tensor):
        original_device = hidden.device
        original_dtype = hidden.dtype
        original_shape = hidden.shape

        selected_mask = self._selected_position_mask(hidden)
        if not bool(selected_mask.any()):
            return None

        selected_mask_device = selected_mask.to(device=self.sae_device)
        hidden_2d = hidden.reshape(-1, hidden.shape[-1]).to(device=self.sae_device, dtype=self.sae_dtype)
        selected_hidden = hidden_2d[selected_mask_device]

        sparse_latents = encode_sparse(self.sae, selected_hidden)
        if not isinstance(sparse_latents, SparseLatents):
            raise TypeError(f"encode_sparse(...) returned {type(sparse_latents)!r}, expected SparseLatents")

        base_top_acts = sparse_latents.top_acts.clone()
        base_top_indices = sparse_latents.top_indices.clone()
        steered_top_acts = base_top_acts.clone()
        steered_top_indices = base_top_indices.clone()

        if self.config.log_feature_acts:
            stats_mask = torch.zeros_like(base_top_acts, dtype=torch.bool)
            for feature_idx in self.config.feature_indices:
                stats_mask |= base_top_indices == int(feature_idx)
            selected = base_top_acts[stats_mask]
            if selected.numel() == 0:
                self.last_feature_stats = {
                    "mean_abs_activation": 0.0,
                    "max_abs_activation": 0.0,
                }
            else:
                self.last_feature_stats = {
                    "mean_abs_activation": float(selected.abs().mean().item()),
                    "max_abs_activation": float(selected.abs().max().item()),
                }

        if self.config.mode != "additive":
            raise ValueError(f"Unsupported latent steering mode: {self.config.mode}")

        for feature_idx in self.config.feature_indices:
            feature_idx = int(feature_idx)

            delta = float(self.config.strength) * self._feature_weight(feature_idx)
            hit_mask = steered_top_indices == feature_idx
            if hit_mask.any():
                steered_top_acts[hit_mask] += delta

            missing_rows = ~hit_mask.any(dim=1)
            if missing_rows.any():
                replacement_col = torch.argmin(steered_top_acts[missing_rows].abs(), dim=1)
                row_idx = torch.arange(replacement_col.shape[0], device=steered_top_acts.device)

                acts_missing = steered_top_acts[missing_rows].clone()
                idx_missing = steered_top_indices[missing_rows].clone()

                idx_missing[row_idx, replacement_col] = feature_idx
                acts_missing[row_idx, replacement_col] = delta

                steered_top_acts[missing_rows] = acts_missing
                steered_top_indices[missing_rows] = idx_missing

        if self.config.clamp_latents is not None:
            clamp_value = float(self.config.clamp_latents)
            steered_top_acts = steered_top_acts.clamp(min=-clamp_value, max=clamp_value)

        steered_recon = self._decode_from_sparse(
            SparseLatents(top_acts=steered_top_acts, top_indices=steered_top_indices),
            dtype=selected_hidden.dtype,
        )

        if self.config.preserve_unsteered_residual:
            base_recon = self._decode_from_sparse(
                SparseLatents(top_acts=base_top_acts, top_indices=base_top_indices),
                dtype=selected_hidden.dtype,
            )
        else:
            base_recon = None

        if self.config.normalize_reconstruction:
            norm_target = base_recon if base_recon is not None else selected_hidden
            steered_recon = self._normalize_reconstruction(steered_recon, norm_target)
            if base_recon is not None:
                base_recon = self._normalize_reconstruction(base_recon, selected_hidden)

        if self.config.preserve_unsteered_residual:
            assert base_recon is not None
            steered_selected = selected_hidden + (steered_recon - base_recon)
        else:
            steered_selected = steered_recon

        updated_hidden = hidden_2d.clone()
        updated_hidden[selected_mask_device] = steered_selected
        recon = updated_hidden.reshape(original_shape).to(device=original_device, dtype=original_dtype)
        return recon

    @torch.inference_mode()
    def _hook_fn(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden is None:
            return output

        if hidden.ndim != 3:
            raise ValueError(
                f"Expected hidden states with shape [batch, seq, d_model], got {tuple(hidden.shape)}"
            )

        mode = str(self.config.mode).strip().lower().replace("-", "_")
        if mode == "decoder_vector":
            recon = self._hook_decoder_vector(hidden)
        else:
            recon = self._hook_latent_additive(hidden)

        if recon is None:
            return output
        if isinstance(output, tuple):
            return (recon,) + output[1:]
        return recon
