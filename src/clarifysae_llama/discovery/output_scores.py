from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from clarifysae_llama.discovery.sae_utils import (
    SparseLatents,
    encode_sparse,
    get_decoder_matrix,
    get_num_latents,
    sparse_to_dense,
)
from clarifysae_llama.utils.io import ensure_dir


@dataclass
class OutputScoreResult:
    feature_idx: int
    top_token_ids: list[int]
    top_tokens: list[str]
    best_token_id: int
    best_token: str
    top_token_score: float
    best_rank_zero_based: int
    best_rank: int
    output_score: float
    prompt: str
    amp_factor: float
    local_max_act: float
    steering_delta: float


def _manual_decode_from_sparse(
    sae,
    sparse: SparseLatents,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    decoder = get_decoder_matrix(sae).to(device=device, dtype=dtype)
    top_acts = sparse.top_acts.to(device=device, dtype=dtype)
    top_indices = sparse.top_indices.to(device=device, dtype=torch.long)

    if top_acts.shape != top_indices.shape:
        raise ValueError(
            "SparseLatents.top_acts and SparseLatents.top_indices must have matching shapes, "
            f"got {tuple(top_acts.shape)} and {tuple(top_indices.shape)}."
        )
    if decoder.ndim != 2:
        raise ValueError(f"Expected decoder matrix with shape [n_features, d_model], got {tuple(decoder.shape)}")

    flat_acts = top_acts.reshape(-1, top_acts.shape[-1])
    flat_indices = top_indices.reshape(-1, top_indices.shape[-1])

    gathered = decoder.index_select(0, flat_indices.reshape(-1))
    gathered = gathered.view(flat_indices.shape[0], flat_indices.shape[1], decoder.shape[1])
    recon = (gathered * flat_acts.unsqueeze(-1)).sum(dim=-2)
    return recon.view(*top_acts.shape[:-1], decoder.shape[1])


def _decode_from_sparse(sae, sparse: SparseLatents, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # CPU sparsify decode can still dispatch into Triton/xFormers and fail, so bypass it.
    if sparse.top_acts.device.type != "cuda" or device.type != "cuda":
        return _manual_decode_from_sparse(sae, sparse, device=device, dtype=dtype)

    try:
        # sparsify-style
        return sae.decode(sparse.top_acts, sparse.top_indices)
    except TypeError:
        # dictionary_learning-style
        dense = sparse_to_dense(
            sparse,
            num_latents=get_num_latents(sae),
            dtype=dtype,
        )
        return sae.decode(dense)
    except (RuntimeError, ValueError) as exc:
        message = str(exc).lower()
        if "triton" in message or "cpu tensor" in message or "xformers" in message:
            return _manual_decode_from_sparse(sae, sparse, device=device, dtype=dtype)
        raise


class SingleFeatureIntervention:
    """
    Paper-faithful OutputScore intervention:
    - operate only on the final sequence position
    - scale by max current activation on that prompt/token
    - add back SAE reconstruction error
    """

    def __init__(
        self,
        target_module,
        sae,
        feature_idx: int,
        amp_factor: float,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.target_module = target_module
        self.sae = sae
        self.feature_idx = int(feature_idx)
        self.amp_factor = float(amp_factor)
        self.dtype = dtype
        self.device = device
        self._handle = None
        self.last_local_max_act: float | None = None
        self.last_delta: float | None = None

    def __enter__(self):
        self._handle = self.target_module.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @torch.inference_mode()
    def _hook_fn(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden is None:
            return output
        if hidden.ndim != 3:
            raise ValueError(
                f"Expected hidden states with shape [batch, seq, d_model], got {tuple(hidden.shape)}"
            )

        original_device = hidden.device
        original_dtype = hidden.dtype

        updated_hidden = hidden.clone()
        last_hidden = hidden[:, -1, :].to(device=self.device, dtype=self.dtype)

        sparse_latents = encode_sparse(self.sae, last_hidden)
        if not isinstance(sparse_latents, SparseLatents):
            raise TypeError(
                f"encode_sparse(...) returned {type(sparse_latents)!r}, expected SparseLatents"
            )

        base_top_acts = sparse_latents.top_acts.clone()
        base_top_indices = sparse_latents.top_indices.clone()

        base_recon = _decode_from_sparse(
            self.sae,
            SparseLatents(top_acts=base_top_acts, top_indices=base_top_indices),
            dtype=last_hidden.dtype,
            device=self.device,
        )

        # Paper code computes SAE error from clean reconstruction and adds it back.
        sae_error = (
            last_hidden.to(torch.float64) - base_recon.to(torch.float64)
        ).to(last_hidden.dtype)

        steered_top_acts = base_top_acts.clone()
        steered_top_indices = base_top_indices.clone()

        local_max_act = float(torch.max(base_top_acts).item()) if base_top_acts.numel() > 0 else 0.0
        steering_delta = local_max_act * self.amp_factor

        self.last_local_max_act = local_max_act
        self.last_delta = float(steering_delta)

        hit_mask = steered_top_indices == self.feature_idx
        if hit_mask.any():
            steered_top_acts[hit_mask] += steering_delta

        missing_rows = ~hit_mask.any(dim=1)
        if missing_rows.any():
            replacement_col = torch.argmin(steered_top_acts[missing_rows].abs(), dim=1)
            row_idx = torch.arange(replacement_col.shape[0], device=steered_top_acts.device)

            acts_missing = steered_top_acts[missing_rows].clone()
            idx_missing = steered_top_indices[missing_rows].clone()

            idx_missing[row_idx, replacement_col] = self.feature_idx
            acts_missing[row_idx, replacement_col] = steering_delta

            steered_top_indices[missing_rows] = idx_missing
            steered_top_acts[missing_rows] = acts_missing

        steered_recon = _decode_from_sparse(
            self.sae,
            SparseLatents(top_acts=steered_top_acts, top_indices=steered_top_indices),
            dtype=last_hidden.dtype,
            device=self.device,
        )
        steered_last_hidden = (steered_recon + sae_error).to(
            device=original_device,
            dtype=original_dtype,
        )

        updated_hidden[:, -1, :] = steered_last_hidden

        if isinstance(output, tuple):
            return (updated_hidden,) + output[1:]
        return updated_hidden


def compute_top_tokens_for_features(
    model,
    sae,
    tokenizer,
    feature_ids: list[int],
    top_k_tokens: int,
) -> dict[int, tuple[list[int], list[str]]]:
    """
    Paper-faithful logit-lens cache:
    decoder -> final layer norm -> lm_head -> softmax -> top-k tokens
    """
    decoder = get_decoder_matrix(sae)
    if decoder.ndim != 2:
        raise ValueError(
            f"Expected SAE decoder matrix to have shape [n_features, d_model], got {tuple(decoder.shape)}"
        )

    lm_head = model.lm_head.weight.detach()
    lm_head_device = lm_head.device
    lm_head_dtype = lm_head.dtype

    final_norm = getattr(model.model, "norm", None)
    results: dict[int, tuple[list[int], list[str]]] = {}

    with torch.inference_mode():
        for feature_idx in feature_ids:
            vec = decoder[int(feature_idx)].detach()
            vec = vec.to(device=lm_head_device, dtype=lm_head_dtype)

            if final_norm is not None:
                try:
                    norm_param = next(final_norm.parameters())
                    norm_device = norm_param.device
                    norm_dtype = norm_param.dtype

                    vec = vec.to(device=norm_device, dtype=norm_dtype)
                    vec = final_norm(vec.unsqueeze(0)).squeeze(0)
                    vec = vec.to(device=lm_head_device, dtype=lm_head_dtype)
                except Exception:
                    vec = vec.to(device=lm_head_device, dtype=lm_head_dtype)

            logits = lm_head @ vec
            confidence = torch.softmax(logits.float(), dim=0)
            top_ids = torch.topk(confidence, k=int(top_k_tokens)).indices.tolist()
            top_tokens = [tokenizer.decode([token_id]) for token_id in top_ids]
            results[int(feature_idx)] = (top_ids, top_tokens)

    return results


def compute_output_scores(
    model,
    tokenizer,
    sae,
    target_module,
    feature_ids: list[int],
    prompt: str,
    amp_factor: float,
    top_k_tokens: int,
    dtype: torch.dtype,
    sae_device: torch.device,
    model_input_device: torch.device,
) -> list[OutputScoreResult]:
    top_tokens_map = compute_top_tokens_for_features(
        model=model,
        sae=sae,
        tokenizer=tokenizer,
        feature_ids=feature_ids,
        top_k_tokens=top_k_tokens,
    )

    prompt_inputs = tokenizer(prompt, return_tensors="pt")
    prompt_inputs = {k: v.to(model_input_device) for k, v in prompt_inputs.items()}

    results: list[OutputScoreResult] = []
    vocab_size = int(model.lm_head.weight.shape[0])

    for feature_idx in feature_ids:
        top_ids, top_tokens = top_tokens_map[int(feature_idx)]

        with SingleFeatureIntervention(
            target_module=target_module,
            sae=sae,
            feature_idx=int(feature_idx),
            amp_factor=float(amp_factor),
            dtype=dtype,
            device=sae_device,
        ) as intervention:
            with torch.inference_mode():
                logits = model(**prompt_inputs, use_cache=False).logits[0, -1]
                probs = torch.softmax(logits.float(), dim=0).detach().cpu()

        logit_lens_probs = probs[top_ids]
        best_prob_idx = int(torch.argmax(logit_lens_probs).item())
        top_token_score = float(logit_lens_probs[best_prob_idx].item())

        tokens_argsort = torch.argsort(probs, dim=0, descending=True)
        ll_token_ranks_zero_based = [
            int((tokens_argsort == token_id).nonzero(as_tuple=True)[0].item())
            for token_id in top_ids
        ]
        best_rank_zero_based = int(min(ll_token_ranks_zero_based))
        rank_output_score = 1.0 - (best_rank_zero_based / vocab_size)
        output_score = float(rank_output_score * top_token_score)

        best_token_id = int(top_ids[best_prob_idx])
        best_token = str(top_tokens[best_prob_idx])

        results.append(
            OutputScoreResult(
                feature_idx=int(feature_idx),
                top_token_ids=[int(x) for x in top_ids],
                top_tokens=[str(x) for x in top_tokens],
                best_token_id=best_token_id,
                best_token=best_token,
                top_token_score=top_token_score,
                best_rank_zero_based=best_rank_zero_based,
                best_rank=best_rank_zero_based + 1,
                output_score=output_score,
                prompt=prompt,
                amp_factor=float(amp_factor),
                local_max_act=float(intervention.last_local_max_act or 0.0),
                steering_delta=float(intervention.last_delta or 0.0),
            )
        )

    return results


def save_output_score_results(
    output_dir: str | Path,
    feature_scores_path: str | Path,
    results: list[OutputScoreResult],
    config: dict[str, Any],
) -> None:
    output_dir = ensure_dir(output_dir)

    rows = [
        {
            "feature_idx": result.feature_idx,
            "output_score": result.output_score,
            "top_token_score": result.top_token_score,
            "best_rank_zero_based": result.best_rank_zero_based,
            "best_rank": result.best_rank,
            "best_token_id": result.best_token_id,
            "best_token": result.best_token,
            "steering_delta": result.steering_delta,
            "local_max_act": result.local_max_act,
            "amp_factor": result.amp_factor,
            "top_token_ids": result.top_token_ids,
            "top_tokens": result.top_tokens,
            "prompt": result.prompt,
        }
        for result in results
    ]

    df = pd.DataFrame(rows).sort_values("output_score", ascending=False)
    df.to_csv(output_dir / "output_scores.csv", index=False)
    (output_dir / "output_scores.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    torch.save({"rows": rows}, output_dir / "output_scores.pt")

    config_payload = {
        "feature_scores_path": str(feature_scores_path),
        "n_features_used": len(results),
        "top_features": [int(r.feature_idx) for r in results],
        "config": config,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )
