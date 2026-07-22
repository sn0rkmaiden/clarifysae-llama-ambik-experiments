from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_module():
    path = Path(__file__).parents[1] / "scripts/build_ambik_actor_corpus.py"
    spec = importlib.util.spec_from_file_location("build_ambik_actor_corpus", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


module = _load_module()


def test_read_split_builds_factorized_actor_rows(tmp_path: Path) -> None:
    source = tmp_path / "train.csv"
    pd.DataFrame(
        [
            {
                "id": "example-1",
                "environment_full": "A robot is in a kitchen.",
                "ambiguity_type": "missing_object",
                "ambiguous_task": "Put the drink in the refrigerator.",
                "unambiguous_direct": "Put the orange juice in the refrigerator.",
                "question": "Which drink should I put in the refrigerator?",
                "answer": "orange juice",
                "plan_for_clear_task": "Pick up the orange juice and place it in the refrigerator.",
            }
        ]
    ).to_csv(source, index=False)

    rows, failures, accepted = module._read_split(
        source,
        split="train",
        generic_question="Could you provide more details?",
    )

    assert accepted == 1
    assert failures == []
    assert len(rows) == 10
    assert {row["split"] for row in rows} == {"train"}
    assert {row["concept"] for row in rows} == {
        "ambiguity_state",
        "ask_trajectory",
        "targeted_question",
        "restraint_on_clear",
        "neutral_prompt",
        "neutral_response",
    }
    actor_rows = [row for row in rows if row["concept"] == "ask_trajectory"]
    assert {row["label"] for row in actor_rows} == {0, 1}
