from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class VocabScoreResult:
    scores: torch.Tensor
    mean_pos: torch.Tensor
    mean_neg: torch.Tensor
    entropy: torch.Tensor
    single_means: torch.Tensor
    count_pos: int
    count_neg: int
    counts_per_group: list[int]


class SparseRollingStats:
    def __init__(
        self,
        num_features: int,
        token_groups: list[list[torch.Tensor]],
        ignore_token_ids: Iterable[int] | None = None,
        expand_range: tuple[int, int] = (0, 0),
        device: torch.device | str = 'cpu',
        dtype: torch.dtype = torch.float32,
    ):
        self.num_features = int(num_features)
        self.device = torch.device(device)
        self.dtype = dtype
        self.token_groups = [[seq.to(self.device) for seq in group] for group in token_groups]
        self.ignore_token_ids = list(ignore_token_ids or [])
        self.expand_range = expand_range

        self.single_sums = [torch.zeros(self.num_features, device=self.device, dtype=self.dtype) for _ in self.token_groups]
        self.single_counts = [0 for _ in self.token_groups]
        self.sum_pos = torch.zeros(self.num_features, device=self.device, dtype=self.dtype)
        self.count_pos = 0
        self.sum_neg = torch.zeros(self.num_features, device=self.device, dtype=self.dtype)
        self.count_neg = 0

    def _compute_single_mask(self, tokens: torch.Tensor, ids_of_interest: torch.Tensor) -> torch.Tensor:
        seq_len = tokens.size(1)
        ids_len = ids_of_interest.numel()
        mask = torch.zeros_like(tokens, dtype=torch.bool, device=tokens.device)
        if ids_len > seq_len:
            return mask

        ids_of_interest = ids_of_interest.view(1, 1, -1)
        windows = tokens.unfold(1, ids_len, 1)
        matches = (windows == ids_of_interest).all(dim=2)
        batch_indices, window_indices = torch.nonzero(matches, as_tuple=True)
        if len(batch_indices) == 0:
            return mask

        offsets = torch.arange(ids_len, device=tokens.device)
        spans = window_indices.unsqueeze(1) + offsets.unsqueeze(0)
        batch_expanded = batch_indices.unsqueeze(1).expand(-1, ids_len).reshape(-1)
        spans_flat = spans.reshape(-1)
        mask[batch_expanded, spans_flat] = True

        left, right = self.expand_range
        if left != 0 or right != 0:
            batch_idx, pos_idx = torch.nonzero(mask, as_tuple=True)
            if len(pos_idx) > 0:
                starts = torch.clamp(pos_idx - left, min=0)
                ends = torch.clamp(pos_idx + right, max=tokens.size(1) - 1)
                delta = torch.zeros(tokens.size(0), tokens.size(1) + 1, dtype=torch.int32, device=tokens.device)
                delta[batch_idx, starts] += 1
                delta[batch_idx, ends + 1] -= 1
                mask = delta.cumsum(dim=1)[:, :-1] > 0
        return mask

    def _scatter_sum(self, top_indices: torch.Tensor, top_acts: torch.Tensor, token_mask: torch.Tensor) -> tuple[torch.Tensor, int]:
        flat_mask = token_mask.reshape(-1)
        count = int(flat_mask.sum().item())
        feature_sum = torch.zeros(self.num_features, device=self.device, dtype=self.dtype)
        if count == 0:
            return feature_sum, count

        flat_indices = top_indices.reshape(-1, top_indices.shape[-1])
        flat_acts = top_acts.reshape(-1, top_acts.shape[-1])
        selected_indices = flat_indices[flat_mask].reshape(-1).long()
        selected_acts = flat_acts[flat_mask].reshape(-1).to(self.dtype)

        if selected_indices.numel() > 0:
            min_idx = int(selected_indices.min().item())
            max_idx = int(selected_indices.max().item())
            if min_idx < 0 or max_idx >= self.num_features:
                raise ValueError(
                    f"SAE feature index out of bounds: min={min_idx}, max={max_idx}, "
                    f"num_features={self.num_features}. "
                    "This usually means the SAE latent dimension was inferred incorrectly."
                )

        feature_sum.scatter_add_(0, selected_indices, selected_acts)
        return feature_sum, count

    def update(self, tokens: torch.Tensor, top_indices: torch.Tensor, top_acts: torch.Tensor) -> None:
        if tokens.ndim != 2:
            raise ValueError(f'Expected tokens with shape [batch, seq], got {tuple(tokens.shape)}')
        if top_indices.ndim != 3 or top_acts.ndim != 3:
            raise ValueError('Expected sparse activations with shape [batch, seq, k].')
        if top_indices.shape != top_acts.shape:
            raise ValueError('top_indices and top_acts must have matching shapes.')
        if tokens.shape != top_indices.shape[:2]:
            raise ValueError('Token batch shape must match the first two activation dimensions.')

        if self.ignore_token_ids:
            ignore_tensor = torch.tensor(self.ignore_token_ids, device=tokens.device, dtype=torch.long)
            ignore_mask = torch.isin(tokens, ignore_tensor)
        else:
            ignore_mask = torch.zeros_like(tokens, dtype=torch.bool)

        combined_mask = torch.zeros_like(tokens, dtype=torch.bool)
        for idx, token_group in enumerate(self.token_groups):
            group_mask = torch.zeros_like(tokens, dtype=torch.bool)
            for token_seq in token_group:
                group_mask |= self._compute_single_mask(tokens, token_seq)
            group_mask &= ~ignore_mask
            feature_sum, count = self._scatter_sum(top_indices, top_acts, group_mask)
            self.single_sums[idx] += feature_sum
            self.single_counts[idx] += count
            combined_mask |= group_mask

        pos_mask = combined_mask & (~ignore_mask)
        neg_mask = (~combined_mask) & (~ignore_mask)

        pos_sum, pos_count = self._scatter_sum(top_indices, top_acts, pos_mask)
        neg_sum, neg_count = self._scatter_sum(top_indices, top_acts, neg_mask)
        self.sum_pos += pos_sum
        self.count_pos += pos_count
        self.sum_neg += neg_sum
        self.count_neg += neg_count

    def finalize(self, alpha: float = 1.0, epsilon: float = 1e-12) -> VocabScoreResult:
        if self.count_pos == 0:
            raise ValueError('No positive tokens were matched. Check the vocabulary file and dataset selection.')
        if self.count_neg == 0:
            raise ValueError('No negative tokens remained after masking. Reduce expand_range or use a larger dataset.')

        mean_pos = self.sum_pos / max(self.count_pos, 1)
        mean_neg = self.sum_neg / max(self.count_neg, 1)

        single_means = torch.stack([
            sum_tensor / max(count, 1)
            for sum_tensor, count in zip(self.single_sums, self.single_counts)
        ], dim=1)

        probs = single_means / (single_means.sum(dim=1, keepdim=True) + epsilon)
        log_probs = torch.where(probs > 0, torch.log(probs), torch.zeros_like(probs))
        entropy = -(probs * log_probs).sum(dim=1)
        if single_means.size(1) > 1:
            entropy = entropy / math.log(single_means.size(1))
        else:
            entropy = torch.ones_like(entropy)

        sum_pos = mean_pos.sum().clamp_min(epsilon)
        sum_neg = mean_neg.sum().clamp_min(epsilon)
        scores = ((mean_pos / sum_pos) * (entropy ** alpha)) - (mean_neg / sum_neg)

        return VocabScoreResult(
            scores=scores.detach().cpu(),
            mean_pos=mean_pos.detach().cpu(),
            mean_neg=mean_neg.detach().cpu(),
            entropy=entropy.detach().cpu(),
            single_means=single_means.detach().cpu(),
            count_pos=int(self.count_pos),
            count_neg=int(self.count_neg),
            counts_per_group=[int(count) for count in self.single_counts],
        )
