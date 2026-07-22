from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def load_script_module(name: str, filename: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_corrective_partition_is_disjoint():
    module = load_script_module("prepare_dense_data", "prepare_dense_experiment_data.py")
    frame = pd.DataFrame({"id": list(range(100)), "value": list(range(100))})
    train, select, evaluate = module.deterministic_partition(
        frame, sizes=[60, 20, 20], seed=42, label="pilot"
    )
    ids = [set(part["id"].tolist()) for part in (train, select, evaluate)]
    assert [len(part) for part in ids] == [60, 20, 20]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert len(set.union(*ids)) == 100
