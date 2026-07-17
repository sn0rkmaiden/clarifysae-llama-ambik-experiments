from __future__ import annotations

"""Utilities for dense concept-vector discovery.

The functions in this module intentionally do not depend on a particular model or
SAE implementation.  They support three useful discovery regimes:

* class difference-in-means (Anthropic-style concept centroids),
* matched-pair difference-in-means (CAA-style counterfactual vectors), and
* a regularized linear probe direction.

All inputs are expected to be 2-D tensors with shape ``[n_examples, d_model]``.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass
class ProbeResult:
    direction: torch.Tensor
    bias: float
    score_mean_positive: float
    score_mean_negative: float
    score_std: float


def _as_float_matrix(x: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)!r}")
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape [n, d], got {tuple(x.shape)}")
    if x.shape[0] == 0:
        raise ValueError(f"{name} contains no examples")
    return x.detach().to(dtype=torch.float32)


def l2_normalize(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    vector = vector.detach().to(dtype=torch.float32)
    norm = vector.norm().clamp_min(float(eps))
    return vector / norm


def difference_in_means(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Return the positive-minus-negative centroid direction."""

    positive = _as_float_matrix(positive, "positive")
    negative = _as_float_matrix(negative, "negative")
    if positive.shape[1] != negative.shape[1]:
        raise ValueError("positive and negative matrices must have the same hidden dimension")
    vector = positive.mean(dim=0) - negative.mean(dim=0)
    return l2_normalize(vector) if normalize else vector


def one_vs_rest_centroid(
    target: torch.Tensor,
    other_classes: Sequence[torch.Tensor],
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Anthropic-style target centroid minus the mean of comparison classes."""

    target = _as_float_matrix(target, "target")
    if not other_classes:
        raise ValueError("other_classes must contain at least one comparison class")
    class_means: list[torch.Tensor] = []
    for idx, matrix in enumerate(other_classes):
        matrix = _as_float_matrix(matrix, f"other_classes[{idx}]")
        if matrix.shape[1] != target.shape[1]:
            raise ValueError("all classes must have the same hidden dimension")
        class_means.append(matrix.mean(dim=0))
    vector = target.mean(dim=0) - torch.stack(class_means, dim=0).mean(dim=0)
    return l2_normalize(vector) if normalize else vector


def paired_difference_in_means(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Return the average matched-pair residual difference.

    This differs from an unpaired centroid difference only in how examples are
    weighted: each scenario contributes one positive-minus-negative difference,
    which prevents topics with more paraphrases from dominating.
    """

    positive = _as_float_matrix(positive, "positive")
    negative = _as_float_matrix(negative, "negative")
    if positive.shape != negative.shape:
        raise ValueError(
            "paired positive and negative matrices must have identical shape, "
            f"got {tuple(positive.shape)} and {tuple(negative.shape)}"
        )
    vector = (positive - negative).mean(dim=0)
    return l2_normalize(vector) if normalize else vector


def fit_ridge_probe(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    l2: float = 1.0,
    normalize_direction: bool = True,
) -> ProbeResult:
    """Fit a regularized least-squares binary probe.

    Targets are +1 for positives and -1 for negatives.  The implementation uses
    the primal system when d <= n and the dual system otherwise, avoiding a huge
    d-by-d inverse for small concept corpora.
    """

    positive = _as_float_matrix(positive, "positive")
    negative = _as_float_matrix(negative, "negative")
    if positive.shape[1] != negative.shape[1]:
        raise ValueError("positive and negative matrices must have the same hidden dimension")
    if l2 <= 0:
        raise ValueError("l2 must be positive")

    x = torch.cat([positive, negative], dim=0)
    y = torch.cat([
        torch.ones(positive.shape[0], dtype=torch.float32),
        -torch.ones(negative.shape[0], dtype=torch.float32),
    ])
    x_mean = x.mean(dim=0, keepdim=True)
    x_centered = x - x_mean

    n, d = x_centered.shape
    if d <= n:
        eye = torch.eye(d, dtype=torch.float32)
        direction = torch.linalg.solve(x_centered.T @ x_centered + float(l2) * eye, x_centered.T @ y)
    else:
        eye = torch.eye(n, dtype=torch.float32)
        alpha = torch.linalg.solve(x_centered @ x_centered.T + float(l2) * eye, y)
        direction = x_centered.T @ alpha

    raw_direction = direction
    scores = x_centered @ raw_direction
    pos_scores = scores[: positive.shape[0]]
    neg_scores = scores[positive.shape[0] :]
    threshold = 0.5 * (float(pos_scores.mean()) + float(neg_scores.mean()))
    # Score new h as h @ w + bias.  Centering and threshold are absorbed here.
    bias = -float(x_mean.squeeze(0) @ raw_direction) - threshold
    score_std = float(scores.std(unbiased=False).clamp_min(1e-8))

    if normalize_direction:
        norm = raw_direction.norm().clamp_min(1e-8)
        direction = raw_direction / norm
        bias /= float(norm)
        pos_mean = float((positive @ direction + bias).mean())
        neg_mean = float((negative @ direction + bias).mean())
        score_std /= float(norm)
    else:
        direction = raw_direction
        pos_mean = float((positive @ direction + bias).mean())
        neg_mean = float((negative @ direction + bias).mean())

    return ProbeResult(
        direction=direction,
        bias=float(bias),
        score_mean_positive=pos_mean,
        score_mean_negative=neg_mean,
        score_std=float(max(score_std, 1e-8)),
    )


def principal_components_for_variance(
    neutral: torch.Tensor,
    *,
    variance_fraction: float = 0.5,
    max_components: int | None = None,
) -> torch.Tensor:
    """Find the smallest neutral PCA subspace reaching ``variance_fraction``."""

    neutral = _as_float_matrix(neutral, "neutral")
    if not 0.0 <= variance_fraction <= 1.0:
        raise ValueError("variance_fraction must be in [0, 1]")
    centered = neutral - neutral.mean(dim=0, keepdim=True)
    if centered.shape[0] < 2 or variance_fraction == 0.0:
        return torch.empty((0, neutral.shape[1]), dtype=torch.float32)

    _u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    variances = singular_values.square()
    if float(variances.sum()) <= 0:
        return torch.empty((0, neutral.shape[1]), dtype=torch.float32)
    cumulative = variances.cumsum(dim=0) / variances.sum()
    k = int(torch.searchsorted(cumulative, torch.tensor(float(variance_fraction))).item()) + 1
    if max_components is not None:
        k = min(k, int(max_components))
    return vh[:k].contiguous()


def remove_subspace(vector: torch.Tensor, components: torch.Tensor, *, normalize: bool = True) -> torch.Tensor:
    """Project a vector orthogonally away from row-wise components."""

    vector = vector.detach().to(dtype=torch.float32)
    if vector.ndim != 1:
        raise ValueError(f"vector must be one-dimensional, got {tuple(vector.shape)}")
    components = components.detach().to(dtype=torch.float32)
    if components.ndim != 2 or components.shape[1] != vector.numel():
        raise ValueError(
            "components must have shape [k, d] matching vector, "
            f"got {tuple(components.shape)} and d={vector.numel()}"
        )
    if components.shape[0] == 0:
        cleaned = vector
    else:
        # SVD components are orthonormal, but QR keeps this safe for arbitrary inputs.
        basis, _ = torch.linalg.qr(components.T, mode="reduced")
        cleaned = vector - basis @ (basis.T @ vector)
    return l2_normalize(cleaned) if normalize else cleaned


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.detach().to(dtype=torch.float32).flatten()
    b = b.detach().to(dtype=torch.float32).flatten()
    if a.numel() != b.numel():
        raise ValueError("vectors must have the same dimension")
    return float((a @ b) / (a.norm().clamp_min(eps) * b.norm().clamp_min(eps)))
