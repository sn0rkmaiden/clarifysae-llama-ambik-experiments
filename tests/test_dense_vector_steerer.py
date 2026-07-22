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
            "steer": {"vector": torch.tensor([1.0, 0.0]), "average_residual_norm": 10.0},
            "gate": {
                "vector": torch.tensor([0.0, 1.0]),
                "probe_bias": 0.0,
                "probe_temperature": 2.5,
            },
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


def test_gate_is_cached_even_for_no_cache_full_sequence_calls(tmp_path):
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
    steerer.attach()
    try:
        first = model(torch.tensor([[[0.0, 0.0], [0.0, 1.0]]]))
        # A later full-sequence call has a negative last token. The original
        # positive prompt decision must remain cached.
        second = model(torch.tensor([[[0.0, 0.0], [0.0, -1.0], [0.0, -2.0]]]))
    finally:
        steerer.detach()
    assert torch.allclose(first[0, -1], torch.tensor([1.0, 1.0]))
    assert torch.allclose(second[0, -1], torch.tensor([1.0, -2.0]))


def test_stored_gate_temperature_and_recorded_norm_scaling(tmp_path):
    model = DummyModel()
    path = _bundle(tmp_path)
    steerer = DenseVectorSteerer(
        model, torch.device("cpu"), torch.float32,
        DenseVectorConfig(
            vector_path=path, vector_key="steer", hookpoint="model.layers.0",
            strength=0.2, scale_mode="recorded_residual_norm_fraction",
            apply_to="last_position", steer_generated_tokens_only=False,
            gate_enabled=True, gate_vector_key="gate", gate_mode="sigmoid",
            gate_temperature=None,
        ),
    )
    assert steerer.gate_temperature == 2.5
    x = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    steerer.attach()
    try:
        y = model(x)
    finally:
        steerer.detach()
    # sigmoid(0)=0.5; strength * recorded norm = 0.2 * 10 = 2.
    assert torch.allclose(y[0, -1], torch.tensor([1.0, 0.0]))


def test_gate_threshold_uses_standardized_score_units(tmp_path):
    model = DummyModel()
    path = _bundle(tmp_path)
    steerer = DenseVectorSteerer(
        model, torch.device("cpu"), torch.float32,
        DenseVectorConfig(
            vector_path=path, vector_key="steer", hookpoint="model.layers.0",
            strength=0.2, scale_mode="recorded_residual_norm_fraction",
            apply_to="last_position", steer_generated_tokens_only=False,
            gate_enabled=True, gate_vector_key="gate", gate_mode="sigmoid",
            gate_threshold=1.0, gate_temperature=None,
        ),
    )
    # Gate score is 2.5 and stored temperature is 2.5, hence standardized
    # score=1.0. Threshold=1.0 must yield sigmoid(0)=0.5.
    x = torch.tensor([[[0.0, 0.0], [0.0, 2.5]]])
    steerer.attach()
    try:
        y = model(x)
    finally:
        steerer.detach()
    assert torch.allclose(y[0, -1], torch.tensor([1.0, 2.5]), atol=1e-6)
