from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_assess():
    path = Path(__file__).parents[1] / "scripts/assess_corrective_pilot.py"
    spec = importlib.util.spec_from_file_location("assess_corrective_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.assess


assess = _load_assess()


def test_corrective_pilot_assessment_go() -> None:
    rows = [
        {
            "arm": "baseline",
            "asking_rate_ambiguous": 0.50,
            "overasking_rate_clear": 0.50,
            "resolved_proxy_any_rate_ambiguous": 0.20,
            "selective_asking_gap": 0.00,
            "json_protocol_valid_rate": 1.00,
        },
        {
            "arm": "dense_ungated",
            "asking_rate_ambiguous": 0.60,
            "overasking_rate_clear": 0.70,
            "resolved_proxy_any_rate_ambiguous": 0.30,
            "selective_asking_gap": -0.10,
            "json_protocol_valid_rate": 1.00,
        },
        {
            "arm": "dense_selected",
            "asking_rate_ambiguous": 0.60,
            "overasking_rate_clear": 0.40,
            "resolved_proxy_any_rate_ambiguous": 0.30,
            "selective_asking_gap": 0.20,
            "json_protocol_valid_rate": 1.00,
        },
        {
            "arm": "dense_sign_flip",
            "asking_rate_ambiguous": 0.40,
            "overasking_rate_clear": 0.50,
            "resolved_proxy_any_rate_ambiguous": 0.10,
            "selective_asking_gap": -0.10,
            "json_protocol_valid_rate": 1.00,
        },
        {
            "arm": "random_direction",
            "asking_rate_ambiguous": 0.50,
            "overasking_rate_clear": 0.50,
            "resolved_proxy_any_rate_ambiguous": 0.20,
            "selective_asking_gap": 0.00,
            "json_protocol_valid_rate": 1.00,
        },
        {
            "arm": "sae_baseline",
            "asking_rate_ambiguous": 0.80,
            "overasking_rate_clear": 0.90,
            "resolved_proxy_any_rate_ambiguous": 0.10,
            "selective_asking_gap": -0.10,
            "json_protocol_valid_rate": 0.90,
        },
    ]
    result = assess(
        corpus_metadata={
            "acceptance_rate": 0.90,
            "accepted_by_topic": {"one": 2, "two": 1},
        },
        layer_selection={
            "actor_ask_vs_guess_eval_auroc": 0.80,
            "actor_targeted_vs_generic_eval_auroc": 0.70,
            "gate_eval_auroc": 0.75,
        },
        summary=pd.DataFrame(rows),
    )
    assert result["recommendation"] == "GO_TO_MAIN"
    assert not result["failed_checks"]


def test_corrective_pilot_assessment_blocks_weak_gate() -> None:
    rows = []
    for arm in (
        "baseline",
        "dense_ungated",
        "dense_selected",
        "dense_sign_flip",
        "random_direction",
        "sae_baseline",
    ):
        rows.append(
            {
                "arm": arm,
                "asking_rate_ambiguous": 0.50,
                "overasking_rate_clear": 0.50,
                "resolved_proxy_any_rate_ambiguous": 0.20,
                "selective_asking_gap": 0.00,
                "json_protocol_valid_rate": 1.00,
            }
        )
    result = assess(
        corpus_metadata={
            "acceptance_rate": 0.90,
            "accepted_by_topic": {"one": 1},
        },
        layer_selection={
            "actor_ask_vs_guess_eval_auroc": 0.80,
            "actor_targeted_vs_generic_eval_auroc": 0.70,
            "gate_eval_auroc": 0.55,
        },
        summary=pd.DataFrame(rows),
    )
    assert result["recommendation"] == "DO_NOT_RUN_MAIN"
    assert not result["checks"]["ambik_gate_auroc_at_least_0_70"]
