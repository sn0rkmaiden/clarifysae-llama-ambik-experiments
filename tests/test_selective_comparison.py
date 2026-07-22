from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import torch

from clarifysae_llama.data.ambik_loader import load_ambik_clarification_dataset


def load_script_module(name: str, filename: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_loader_preserves_pair_metadata(tmp_path):
    path = tmp_path / "paired.csv"
    pd.DataFrame(
        [
            {
                "id": 0,
                "source_id": "abc",
                "source_split": "test",
                "pair_variant": "ambiguous",
                "environment_full": "one red and one blue cup",
                "ambiguity_type": "preferences",
                "ambiguous_task": "move a cup",
                "question": "Which cup?",
                "answer": "blue",
                "plan_for_clear_task": "move blue cup",
            }
        ]
    ).to_csv(path, index=False)
    loaded = load_ambik_clarification_dataset(path)
    assert loaded.loc[0, "source_id"] == "abc"
    assert loaded.loc[0, "pair_variant"] == "ambiguous"


def test_random_control_is_orthogonal_to_ask_and_gate(tmp_path):
    module = load_script_module("build_selective_controls", "build_selective_controls.py")
    source = tmp_path / "vectors.pt"
    output = tmp_path / "random.pt"
    ask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    gate = torch.tensor([1.0, 0.0, 1.0, 0.0])
    torch.save(
        {
            "vectors": {
                "ask": {"vector": ask, "average_residual_norm": 7.0},
                "gate": {"vector": gate},
            }
        },
        source,
    )
    key = module.make_orthogonal_random_bundle(
        vector_path=source,
        ask_key="ask",
        gate_key="gate",
        output_path=output,
        seed=42,
    )
    record = torch.load(output, map_location="cpu", weights_only=False)["vectors"][key]
    random_vector = record["vector"]
    assert torch.isclose(random_vector.norm(), torch.tensor(1.0), atol=1e-6)
    assert abs(float(torch.dot(random_vector, ask / ask.norm()))) < 1e-5
    assert abs(float(torch.dot(random_vector, gate / gate.norm()))) < 1e-5
