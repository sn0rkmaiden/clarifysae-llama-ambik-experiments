from __future__ import annotations

"""Generate matched clarification counterfactuals with a language model.

The JSONL is deliberately factorized.  It separates prompt-side ambiguity
recognition from response-side ask/guess, question targeting, and restraint.
Neutral prompt and response rows are also emitted so that nuisance principal
components can be estimated rather than merely configured in downstream code.
"""

import argparse
import difflib
import json
import random
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

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

FORBIDDEN_META_TERMS = {
    "ambiguous", "ambiguity", "underspecified", "under-specified",
    "clarification", "clarify", "missing information",
}


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


def _content_words(text: str) -> set[str]:
    stop = {"a", "an", "the", "of", "for", "to", "which", "what", "type", "kind", "choice"}
    return {
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in stop
    }


def _validate_scenario(obj: dict[str, Any], *, strict: bool = True) -> dict[str, str]:
    required = [
        "context", "ambiguous_instruction", "clear_instruction", "missing_slot",
        "targeted_question", "direct_response", "guessing_response", "generic_question",
        "unnecessary_question",
    ]
    missing = [key for key in required if not str(obj.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Generated scenario is missing fields: {missing}")
    scenario = {key: str(obj[key]).strip() for key in required}

    joined = " ".join(scenario.values()).lower()
    forbidden = sorted(term for term in FORBIDDEN_META_TERMS if term in joined)
    if forbidden:
        raise ValueError(f"Scenario contains forbidden meta-terms: {forbidden}")

    for key in ("targeted_question", "generic_question", "unnecessary_question"):
        if not scenario[key].rstrip().endswith("?"):
            raise ValueError(f"{key} must be a question ending in '?'")
    if scenario["targeted_question"].casefold() == scenario["generic_question"].casefold():
        raise ValueError("targeted_question and generic_question must differ")
    if scenario["ambiguous_instruction"].casefold() == scenario["clear_instruction"].casefold():
        raise ValueError("clear_instruction must fill the missing slot")

    if strict:
        similarity = difflib.SequenceMatcher(
            None, scenario["ambiguous_instruction"].casefold(), scenario["clear_instruction"].casefold()
        ).ratio()
        if similarity < 0.55:
            raise ValueError(
                "clear_instruction is not a minimal edit of ambiguous_instruction "
                f"(character similarity={similarity:.3f})"
            )
        slot_words = _content_words(scenario["missing_slot"])
        target_words = _content_words(scenario["targeted_question"])
        generic_words = _content_words(scenario["generic_question"])
        if slot_words and not (slot_words & target_words):
            raise ValueError(
                "targeted_question has no lexical overlap with missing_slot; "
                "rewrite the slot label or question more explicitly"
            )
        if slot_words & generic_words:
            raise ValueError("generic_question leaks the named missing slot")
        for key in ("direct_response", "guessing_response"):
            if scenario[key].rstrip().endswith("?"):
                raise ValueError(f"{key} should be a direct answer/action, not a question")

    return scenario


def _writer_prompt(topic: str, scenario_index: int) -> str:
    return f"""Create one realistic matched counterfactual scenario about {topic} for mechanistic-interpretability research.

The goal is to separate these concepts: (1) recognizing that a required variable is not specified, (2) choosing to ask rather than guess, and (3) asking for the specific variable rather than merely producing an interrogative sentence.

Return exactly one JSON object with these string fields:
- context: all relevant environment/background facts
- ambiguous_instruction: an instruction with exactly one consequential unspecified slot and at least two plausible values
- clear_instruction: the same instruction with that slot explicitly filled; change as little wording as possible
- missing_slot: a short noun phrase naming the information that is needed
- targeted_question: one concise, actionable question that asks only for that slot and uses at least one important word from missing_slot
- direct_response: an appropriate direct response/action to the clear instruction
- guessing_response: a fluent response to the ambiguous instruction that silently selects one plausible value (hard negative)
- generic_question: an interrogative response to the ambiguous instruction that does not identify or name the missing slot (hard negative)
- unnecessary_question: a plausible but needless question asked after the clear instruction (hard negative)

Constraints:
- Do not use meta-words such as “ambiguous”, “underspecified”, “clarification”, “clarify”, or “missing information” inside the scenario.
- Make the question content semantically necessary, not merely stylistic.
- The clear and unclear instructions must be a minimal pair.
- Keep every field short enough that the whole JSON is under 350 words.
- Scenario id seed: {scenario_index}.
"""


def _transcript(context: str, instruction: str, response: str | None = None) -> str:
    text = f"Context: {context}\nUser instruction: {instruction}"
    if response is not None:
        text += f"\nAssistant: {response}"
    return text


def expand_scenario(
    scenario_id: str,
    topic: str,
    s: dict[str, str],
    *,
    split: str = "train",
) -> list[dict[str, Any]]:
    metadata = {"topic": topic, "missing_slot": s["missing_slot"]}
    rows: list[dict[str, Any]] = []

    def add(
        concept: str,
        pair_suffix: str,
        label: int,
        variant: str,
        text: str,
        pooling: str,
    ) -> None:
        rows.append({
            "id": f"{scenario_id}:{concept}:{variant}",
            "pair_id": f"{scenario_id}:{concept}:{pair_suffix}",
            "scenario_id": scenario_id,
            "split": split,
            "concept": concept,
            "label": int(label),
            "variant": variant,
            "text": text,
            "recommended_pooling": pooling,
            "metadata": metadata,
        })

    # Prompt-side state: this can be used as a detector/gate because the two
    # examples differ before any assistant response is generated.
    add("ambiguity_state", "state", 1, "ambiguous_prompt",
        _transcript(s["context"], s["ambiguous_instruction"]), "last_nonpad")
    add("ambiguity_state", "state", 0, "clear_prompt",
        _transcript(s["context"], s["clear_instruction"]), "last_nonpad")

    # Response-trajectory direction under the same prompt. This is intentionally
    # not called a pre-response policy state: the signal only becomes observable
    # after the assistant's question/guess tokens are present.
    add("ask_trajectory", "policy", 1, "targeted_question",
        _transcript(s["context"], s["ambiguous_instruction"], s["targeted_question"]), "assistant_mean")
    add("ask_trajectory", "policy", 0, "silent_guess",
        _transcript(s["context"], s["ambiguous_instruction"], s["guessing_response"]), "assistant_mean")

    # Question content: targeted versus generic while holding the prompt fixed.
    add("targeted_question", "quality", 1, "targeted_question",
        _transcript(s["context"], s["ambiguous_instruction"], s["targeted_question"]), "assistant_mean")
    add("targeted_question", "quality", 0, "generic_question",
        _transcript(s["context"], s["ambiguous_instruction"], s["generic_question"]), "assistant_mean")

    # Restraint on clear inputs; positive means the behavior we want to increase.
    add("restraint_on_clear", "restraint", 1, "direct_response",
        _transcript(s["context"], s["clear_instruction"], s["direct_response"]), "assistant_mean")
    add("restraint_on_clear", "restraint", 0, "unnecessary_question",
        _transcript(s["context"], s["clear_instruction"], s["unnecessary_question"]), "assistant_mean")

    # Nuisance corpora for Anthropic-style PCA removal. Keep prompt and response
    # neutral rows separate because their pooling distributions are different.
    add("neutral_prompt", "neutral_prompt", 0, "clear_prompt",
        _transcript(s["context"], s["clear_instruction"]), "last_nonpad")
    add("neutral_response", "neutral_response", 0, "helpful_direct_response",
        _transcript(s["context"], s["clear_instruction"], s["direct_response"]), "assistant_mean")

    return rows


def _topic_split_map(
    topics: list[str],
    *,
    seed: int,
    train_fraction: float,
    dev_fraction: float,
) -> dict[str, str]:
    if train_fraction <= 0 or dev_fraction < 0 or train_fraction + dev_fraction >= 1:
        raise ValueError("split fractions must satisfy train>0, dev>=0, and train+dev<1")
    shuffled = list(topics)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(round(n * train_fraction)))
    n_dev = int(round(n * dev_fraction))
    if n_train + n_dev >= n:
        n_dev = max(0, n - n_train - 1)
    mapping: dict[str, str] = {}
    for i, topic in enumerate(shuffled):
        mapping[topic] = "train" if i < n_train else "dev" if i < n_train + n_dev else "test"
    return mapping


def run(config: dict[str, Any]) -> None:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    corpus_cfg = config.get("synthetic_corpus", {})
    output_path = Path(corpus_cfg.get("output_path", "outputs/synthetic/clarification_counterfactuals.jsonl"))
    ensure_dir(output_path.parent)

    topics = [str(x) for x in corpus_cfg.get("topics", DEFAULT_TOPICS)]
    n_per_topic = int(corpus_cfg.get("scenarios_per_topic", 12))
    max_retries = int(corpus_cfg.get("max_retries", 2))
    strict_validation = bool(corpus_cfg.get("strict_validation", True))
    split_cfg = corpus_cfg.get("splits", {})
    split_map = _topic_split_map(
        topics,
        seed=seed,
        train_fraction=float(split_cfg.get("train", 0.70)),
        dev_fraction=float(split_cfg.get("dev", 0.15)),
    )
    from clarifysae_llama.backends.hf_backend import HFCausalBackend

    backend = HFCausalBackend(config)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    accepted_scenarios = 0
    for topic_idx, topic in enumerate(tqdm(topics, desc="Topics")):
        for local_idx in range(n_per_topic):
            scenario_index = topic_idx * n_per_topic + local_idx
            prompt = _writer_prompt(topic, scenario_index)
            last_error: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    raw = backend.generate(prompt)
                    scenario = _validate_scenario(
                        _extract_json_object(raw), strict=strict_validation
                    )
                    scenario_id = f"s{scenario_index:05d}"
                    rows.extend(expand_scenario(
                        scenario_id, topic, scenario, split=split_map[topic]
                    ))
                    accepted_scenarios += 1
                    last_error = None
                    break
                except Exception as exc:  # preserve failed generations for audit
                    last_error = exc
            if last_error is not None:
                failures.append({
                    "topic": topic,
                    "scenario_index": scenario_index,
                    "split": split_map[topic],
                    "error": repr(last_error),
                })

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    failure_path = output_path.with_suffix(".failures.json")
    failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    split_counts = {
        split: sum(1 for row in rows if row["split"] == split)
        for split in ("train", "dev", "test")
    }
    metadata = {
        "rows": len(rows),
        "rows_per_scenario": 10,
        "scenarios": accepted_scenarios,
        "failures": len(failures),
        "topics": topics,
        "topic_splits": split_map,
        "row_split_counts": split_counts,
        "scenarios_per_topic": n_per_topic,
        "writer_model": config["model"]["name"],
        "strict_validation": strict_validation,
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
