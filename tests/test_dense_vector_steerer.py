from pathlib import Path

import torch

from clarifysae_llama.steering.dense_vector_steerer import DenseVectorConfig, DenseVectorSteerer


class Block(torch.nn.Module):
    def forward(self, x):
        return x


class Inner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block()])


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Inner()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return self.model.layers[0](x)


def _bundle(tmp_path: Path) -> str:
    path = tmp_path / "vectors.pt"
    torch.save({
        "vectors": {
            "steer": {"vector": torch.tensor([1.0, 0.0])},
            "gate": {"vector": torch.tensor([0.0, 1.0]), "probe_bias": 0.0},
        }
    }, path)
    return str(path)


def test_dense_vector_last_position(tmp_path):
    model = DummyModel()
    path = _bundle(tmp_path)
    steerer = DenseVectorSteerer(
        model, torch.device("cpu"), torch.float32,
        DenseVectorConfig(
            vector_path=path, vector_key="steer", hookpoint="model.layers.0",
            strength=2.0, apply_to="last_position", steer_generated_tokens_only=False,
        ),
    )
    x = torch.zeros((1, 3, 2))
    steerer.attach()
    try:
        y = model(x)
    finally:
        steerer.detach()
    assert torch.allclose(y[0, 0], torch.zeros(2))
    assert torch.allclose(y[0, -1], torch.tensor([2.0, 0.0]))


def test_hard_gate_blocks_clear_example(tmp_path):
    model = DummyModel()
    path = _bundle(tmp_path)
    steerer = DenseVectorSteerer(
        model, torch.device("cpu"), torch.float32,
        DenseVectorConfig(
            vector_path=path, vector_key="steer", hookpoint="model.layers.0",
            strength=1.0, apply_to="last_position", steer_generated_tokens_only=False,
            gate_enabled=True, gate_vector_key="gate", gate_mode="hard",
        ),
    )
    # First example has positive second coordinate at final prompt token; second is negative.
    x = torch.tensor([[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, -1.0]]])
    steerer.attach()
    try:
        y = model(x)
    finally:
        steerer.detach()
    assert torch.allclose(y[0, -1], torch.tensor([1.0, 1.0]))
    assert torch.allclose(y[1, -1], torch.tensor([0.0, -1.0]))
