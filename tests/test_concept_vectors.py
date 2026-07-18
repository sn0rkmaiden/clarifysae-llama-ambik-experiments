import torch

from clarifysae_llama.discovery.concept_vectors import (
    binary_score_diagnostics,
    calibrate_probe_direction,
    difference_in_means,
    fit_ridge_probe,
    paired_difference_in_means,
    principal_components_for_variance,
    remove_subspace,
)


def test_paired_direction_points_positive():
    pos = torch.tensor([[2.0, 0.0], [3.0, 0.0]])
    neg = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    vector = paired_difference_in_means(pos, neg)
    assert torch.allclose(vector, torch.tensor([1.0, 0.0]), atol=1e-6)


def test_difference_in_means():
    pos = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    neg = torch.zeros_like(pos)
    vector = difference_in_means(pos, neg)
    assert torch.isclose(vector.norm(), torch.tensor(1.0), atol=1e-6)
    assert vector[1] > vector[0] > 0


def test_ridge_probe_separates_classes():
    pos = torch.tensor([[2.0, 0.1], [1.8, -0.1], [2.2, 0.0]])
    neg = torch.tensor([[-2.0, 0.1], [-1.8, -0.1], [-2.2, 0.0]])
    probe = fit_ridge_probe(pos, neg, l2=0.1)
    assert probe.score_mean_positive > 0
    assert probe.score_mean_negative < 0


def test_neutral_pc_removal():
    neutral = torch.tensor([[3.0, 0.0], [2.0, 0.1], [-2.0, 0.0], [-3.0, -0.1]])
    pcs = principal_components_for_variance(neutral, variance_fraction=0.8)
    cleaned = remove_subspace(torch.tensor([1.0, 1.0]), pcs)
    assert abs(float(cleaned[0])) < 0.2
    assert float(cleaned[1]) > 0.9


def test_recalibrate_after_projection():
    pos = torch.tensor([[3.0, 2.0], [2.5, 1.5]])
    neg = torch.tensor([[1.0, -2.0], [0.5, -1.5]])
    # Imagine the first coordinate was removed as a nuisance PC.
    cleaned = torch.tensor([0.0, 1.0])
    probe = calibrate_probe_direction(cleaned, pos, neg)
    diagnostics = binary_score_diagnostics(probe.direction, probe.bias, pos, neg)
    assert probe.score_mean_positive > 0
    assert probe.score_mean_negative < 0
    assert diagnostics["auroc"] == 1.0
