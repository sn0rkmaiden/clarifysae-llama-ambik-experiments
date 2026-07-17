from __future__ import annotations

"""Generate matched clarification counterfactuals with the model under study.

The output is JSONL in a long format suitable for dense-vector extraction.  Each
underlying scenario contributes multiple tightly matched positive/negative pairs,
which is much less confounded than collecting arbitrary question-containing text.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from clarifysae_llama.backends.hf_backend import HFCausalBackend
from clarifysae_llama.config import load_yaml
from clarifysae_llama.utils.io import ensure_dir
from clarifysae_llama.utils.seed import set_seed


DEFAULT_TOPICS = [
    "cooking", "household organization", "travel planning", "office work",
    "shopping", "personal scheduling", "software configuration", "data analysis",
    "robot manipulation", "healthcare administration", "education", "finance",
    "creative writing", "event planning", "customer support", "navigation",
    "safety-critical maintenance", "accessibility", "collaboration", "research",
]


def _extract_json_object(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first:last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse a JSON object from model output: {text[:500]!r}")


def _validate_scenario(obj: dict[str, Any]) -> dict[str, str]:
    required = [
        "context", "ambiguous_instruction", "clear_instruction", "missing_slot",
        "targeted_question", "direct_response", "guessing_response", "generic_question",
        "unnecessary_question",
    ]
    missing = [key for key in required if not str(obj.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Generated scenario is missing fields: {missing}")
    return {key: str(obj[key]).strip() for key in required}


def _writer_prompt(topic: str, scenario_index: int) -> str:
    return f"""Create one realistic matched counterfactual scenario about {topic} for mechanistic-interpretability research.

The goal is to separate these concepts: (1) recognizing underspecification, (2) choosing to ask rather than guess, and (3) asking for the specific missing variable rather than merely producing an interrogative sentence.

Return exactly one JSON object with these string fields:
- context: all relevant environment/background facts
- ambiguous_instruction: an instruction with exactly one consequential missing slot and at least two plausible values
- clear_instruction: the same instruction with that slot explicitly filled; change as little wording as possible
- missing_slot: a short noun phrase naming the information that is missing
- targeted_question: one concise, actionable question that asks only for the missing slot
- direct_response: an appropriate direct response/action to the clear instruction
- guessing_response: a fluent response to the ambiguous instruction that silently selects one plausible value (this is a hard negative)
- generic_question: an interrogative response to the ambiguous instruction that does not identify the missing slot (hard negative)
- unnecessary_question: a plausible but needless question asked after the clear instruction (hard negative)

Constraints:
- Do not use meta-words such as “ambiguous”, “underspecified”, “clarification”, or “missing information” inside the scenario.
- Make the question content semantically necessary, not merely stylistic.
- Keep every field short enough that the whole JSON is under 350 words.
- Scenario id seed: {scenario_index}.
"""


def _transcript(context: str, instruction: str, response: str | None = None) -> str:
    text = f"Context: {context}\nUser instruction: {instruction}"
    if response is not None:
        text += f"\nAssistant: {response}"
    return text


def expand_scenario(scenario_id: str, topic: str, s: dict[str, str]) -> list[dict[str, Any]]:
    metadata = {"topic": topic, "missing_slot": s["missing_slot"]}
    rows: list[dict[str, Any]] = []

    def add(concept: str, pair_suffix: str, label: int, variant: str, text: str, pooling: str) -> None:
        rows.append({
            "id": f"{scenario_id}:{concept}:{variant}",
            "pair_id": f"{scenario_id}:{concept}:{pair_suffix}",
            "scenario_id": scenario_id,
            "concept": concept,
            "label": int(label),
            "variant": variant,
            "text": text,
            "recommended_pooling": pooling,
            "metadata": metadata,
        })

    # State vector: decide whether the prompt lacks a required variable, before
    # an assistant answer exists.
    add("ambiguity_state", "state", 1, "ambiguous_prompt",
        _transcript(s["context"], s["ambiguous_instruction"]), "last_nonpad")
    add("ambiguity_state", "state", 0, "clear_prompt",
        _transcript(s["context"], s["clear_instruction"]), "last_nonpad")

    # Ask-vs-guess policy under the exact same ambiguous prompt.
    add("ask_policy", "policy", 1, "targeted_question",
        _transcript(s["context"], s["ambiguous_instruction"], s["targeted_question"]), "assistant_mean")
    add("ask_policy", "policy", 0, "silent_guess",
        _transcript(s["context"], s["ambiguous_instruction"], s["guessing_response"]), "assistant_mean")

    # Question quality: targeted versus generic while holding the prompt fixed.
    add("targeted_question", "quality", 1, "targeted_question",
        _transcript(s["context"], s["ambiguous_instruction"], s["targeted_question"]), "assistant_mean")
    add("targeted_question", "quality", 0, "generic_question",
        _transcript(s["context"], s["ambiguous_instruction"], s["generic_question"]), "assistant_mean")

    # Restraint on clear inputs; positive means the behavior we want to increase.
    add("restraint_on_clear", "restraint", 1, "direct_response",
        _transcript(s["context"], s["clear_instruction"], s["direct_response"]), "assistant_mean")
    add("restraint_on_clear", "restraint", 0, "unnecessary_question",
        _transcript(s["context"], s["clear_instruction"], s["unnecessary_question"]), "assistant_mean")

    return rows


def run(config: dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    corpus_cfg = config.get("synthetic_corpus", {})
    output_path = Path(corpus_cfg.get("output_path", "outputs/synthetic/clarification_counterfactuals.jsonl"))
    ensure_dir(output_path.parent)

    topics = list(corpus_cfg.get("topics", DEFAULT_TOPICS))
    n_per_topic = int(corpus_cfg.get("scenarios_per_topic", 12))
    max_retries = int(corpus_cfg.get("max_retries", 2))
    backend = HFCausalBackend(config)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for topic_idx, topic in enumerate(tqdm(topics, desc="Topics")):
        for local_idx in range(n_per_topic):
            scenario_index = topic_idx * n_per_topic + local_idx
            prompt = _writer_prompt(str(topic), scenario_index)
            last_error: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    raw = backend.generate(prompt)
                    scenario = _validate_scenario(_extract_json_object(raw))
                    scenario_id = f"s{scenario_index:05d}"
                    rows.extend(expand_scenario(scenario_id, str(topic), scenario))
                    last_error = None
                    break
                except Exception as exc:  # preserve failed generations for audit
                    last_error = exc
            if last_error is not None:
                failures.append({"topic": topic, "scenario_index": scenario_index, "error": repr(last_error)})

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    failure_path = output_path.with_suffix(".failures.json")
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "rows": len(rows),
        "scenarios": len(rows) // 8,
        "failures": len(failures),
        "topics": topics,
        "scenarios_per_topic": n_per_topic,
        "writer_model": config["model"]["name"],
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(load_yaml(args.config))
