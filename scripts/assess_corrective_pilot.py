from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _arm(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    rows = frame[frame["arm"].astype(str) == name]
    if len(rows) != 1:
        raise ValueError(f"Expected one row for arm {name!r}, found {len(rows)}")
    return rows.iloc[0].to_dict()


def _ge(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _le(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def assess(
    *,
    corpus_metadata: dict[str, Any],
    layer_selection: dict[str, Any],
    summary: pd.DataFrame,
) -> dict[str, Any]:
    required_arms = {
        "baseline",
        "dense_ungated",
        "dense_selected",
        "dense_sign_flip",
        "random_direction",
        "sae_baseline",
    }
    missing = required_arms - set(summary["arm"].astype(str))
    if missing:
        raise ValueError(f"Pilot summary lacks arms: {sorted(missing)}")

    baseline = _arm(summary, "baseline")
    ungated = _arm(summary, "dense_ungated")
    gated = _arm(summary, "dense_selected")
    sign_flip = _arm(summary, "dense_sign_flip")
    random_control = _arm(summary, "random_direction")

    acceptance_rate = _finite(corpus_metadata.get("acceptance_rate"))
    accepted_by_topic = dict(corpus_metadata.get("accepted_by_topic", {}))
    missing_topics = sorted(
        str(topic) for topic, count in accepted_by_topic.items() if int(count) <= 0
    )
    actor_guess = _finite(layer_selection.get("actor_ask_vs_guess_eval_auroc"))
    actor_generic = _finite(
        layer_selection.get("actor_targeted_vs_generic_eval_auroc")
    )
    gate_auroc = _finite(layer_selection.get("gate_eval_auroc"))
    protocol_valid = _finite(gated.get("json_protocol_valid_rate"))

    baseline_ask = _finite(baseline.get("asking_rate_ambiguous"))
    ungated_ask = _finite(ungated.get("asking_rate_ambiguous"))
    baseline_resolution = _finite(
        baseline.get("resolved_proxy_any_rate_ambiguous")
    )
    ungated_resolution = _finite(
        ungated.get("resolved_proxy_any_rate_ambiguous")
    )
    ungated_overask = _finite(ungated.get("overasking_rate_clear"))
    gated_overask = _finite(gated.get("overasking_rate_clear"))

    ask_effect = (
        None if baseline_ask is None or ungated_ask is None
        else ungated_ask - baseline_ask
    )
    resolution_effect = (
        None
        if baseline_resolution is None or ungated_resolution is None
        else ungated_resolution - baseline_resolution
    )
    actor_causal_effect = any(
        value is not None and abs(value) >= 0.05
        for value in (ask_effect, resolution_effect)
    )
    gating_reduces_overask = (
        ungated_overask is not None
        and gated_overask is not None
        and gated_overask <= ungated_overask
    )

    checks = {
        "synthetic_acceptance_at_least_0_80": _ge(acceptance_rate, 0.80),
        "all_topics_covered": not missing_topics,
        "actor_ask_vs_guess_auroc_at_least_0_75": _ge(actor_guess, 0.75),
        "actor_targeted_vs_generic_auroc_at_least_0_65": _ge(
            actor_generic, 0.65
        ),
        "ambik_gate_auroc_at_least_0_70": _ge(gate_auroc, 0.70),
        "dense_selected_protocol_valid_at_least_0_95": _ge(
            protocol_valid, 0.95
        ),
        "ungated_actor_has_detectable_causal_effect": actor_causal_effect,
        "gating_does_not_increase_clear_overasking": gating_reduces_overask,
    }
    failed = [name for name, passed in checks.items() if not passed]

    selected_gap = _finite(gated.get("selective_asking_gap"))
    sign_gap = _finite(sign_flip.get("selective_asking_gap"))
    random_gap = _finite(random_control.get("selective_asking_gap"))
    selected_resolution = _finite(
        gated.get("resolved_proxy_any_rate_ambiguous")
    )
    sign_resolution = _finite(
        sign_flip.get("resolved_proxy_any_rate_ambiguous")
    )
    random_resolution = _finite(
        random_control.get("resolved_proxy_any_rate_ambiguous")
    )

    control_evidence = {
        "selected_selective_gap": selected_gap,
        "sign_flip_selective_gap": sign_gap,
        "random_selective_gap": random_gap,
        "selected_resolution": selected_resolution,
        "sign_flip_resolution": sign_resolution,
        "random_resolution": random_resolution,
        "selected_beats_both_controls_on_gap": (
            selected_gap is not None
            and sign_gap is not None
            and random_gap is not None
            and selected_gap > max(sign_gap, random_gap)
        ),
        "selected_beats_both_controls_on_resolution": (
            selected_resolution is not None
            and sign_resolution is not None
            and random_resolution is not None
            and selected_resolution > max(sign_resolution, random_resolution)
        ),
    }

    recommendation = "GO_TO_MAIN" if not failed else "DO_NOT_RUN_MAIN"
    return {
        "recommendation": recommendation,
        "checks": checks,
        "failed_checks": failed,
        "representation_diagnostics": {
            "synthetic_acceptance_rate": acceptance_rate,
            "missing_topics": missing_topics,
            "actor_ask_vs_guess_eval_auroc": actor_guess,
            "actor_targeted_vs_generic_eval_auroc": actor_generic,
            "gate_eval_auroc": gate_auroc,
        },
        "causal_diagnostics": {
            "baseline_asking_rate_ambiguous": baseline_ask,
            "ungated_asking_rate_ambiguous": ungated_ask,
            "ungated_minus_baseline_asking": ask_effect,
            "baseline_resolution": baseline_resolution,
            "ungated_resolution": ungated_resolution,
            "ungated_minus_baseline_resolution": resolution_effect,
            "ungated_overasking_rate_clear": ungated_overask,
            "gated_overasking_rate_clear": gated_overask,
            "gated_protocol_valid_rate": protocol_valid,
        },
        "control_evidence": control_evidence,
        "note": (
            "Control superiority is reported but is not a hard automatic gate on "
            "this 20-pair pilot. Inspect confidence intervals before publication."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-metadata",
        default="outputs/synthetic/clarification_counterfactuals_pilot.metadata.json",
    )
    parser.add_argument(
        "--layer-selection",
        default="outputs/selection/corrective_pilot_layer_selection.json",
    )
    parser.add_argument(
        "--summary",
        default="outputs/selective_corrective_pilot/comparison/summary.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/selective_corrective_pilot/comparison/pilot_assessment.json",
    )
    parser.add_argument("--require-go", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.corpus_metadata)
    selection_path = Path(args.layer_selection)
    summary_path = Path(args.summary)
    for path in (metadata_path, selection_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(path)

    result = assess(
        corpus_metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
        layer_selection=json.loads(selection_path.read_text(encoding="utf-8")),
        summary=pd.read_csv(summary_path),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Pilot assessment: {output}")
    if args.require_go and result["recommendation"] != "GO_TO_MAIN":
        raise SystemExit(
            "Corrective pilot did not pass the pre-registered go/no-go checks. "
            "Do not run the main experiment unless the failure has been reviewed."
        )


if __name__ == "__main__":
    main()
