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
    missing = [key for key in required if key not in obj or obj.get(key) is None]
    if missing:
        raise ValueError(f"Generated scenario is missing fields: {missing}")
    wrong_types = [key for key in required if not isinstance(obj.get(key), str)]
    if wrong_types:
        raise ValueError(
            "Every scenario field must be a JSON string; wrong types for: "
            f"{wrong_types}"
        )
    empty = [key for key in required if not obj[key].strip()]
    if empty:
        raise ValueError(f"Generated scenario has empty fields: {empty}")
    scenario = {key: obj[key].strip() for key in required}

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

Return exactly one JSON object and nothing else. Every value must be a JSON string, never a list, object, boolean, or number. Use exactly these keys:
- context
- ambiguous_instruction
- clear_instruction
- missing_slot
- targeted_question
- direct_response
- guessing_response
- generic_question
- unnecessary_question

Semantic requirements:
1. ambiguous_instruction leaves exactly one consequential choice unspecified, with at least two plausible values.
2. clear_instruction is the same instruction with only that choice filled.
3. missing_slot is a short noun phrase naming the choice.
4. targeted_question asks only for that choice and repeats at least one important content word from missing_slot.
5. guessing_response answers the ambiguous instruction by silently choosing one plausible value and must not be a question.
6. generic_question is a question such as asking for more details, but it must not contain any content word from missing_slot.
7. direct_response appropriately answers the clear instruction and must not be a question.
8. unnecessary_question is a plausible but needless question after the clear instruction.

Formatting requirements:
- targeted_question, generic_question, and unnecessary_question must end with ?.
- Do not use the words ambiguous, ambiguity, underspecified, under-specified, clarification, clarify, or missing information in any value.
- Keep the ambiguous/clear instructions as a minimal pair.
- Keep the complete JSON under 350 words.
- Do not wrap the JSON in markdown or add commentary.
- Scenario seed: {scenario_index}.
"""


def _repair_prompt(
    original_prompt: str,
    previous_output: str,
    error: Exception,
    attempt: int,
) -> str:
    previous = previous_output.strip()[:2400]
    return f"""{original_prompt}

Your previous attempt failed validation. Repair it and return a completely corrected JSON object only.
Validation error: {type(error).__name__}: {error}
Repair attempt: {attempt}
Previous output:
{previous}

Do not explain the repair. Return only the corrected JSON object with all nine values as JSON strings.
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
    completion_path = output_path.with_suffix(".complete")
    completion_path.unlink(missing_ok=True)

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
    accepted_by_topic = {topic: 0 for topic in topics}
    total_generation_calls = 0
    scenario_slots = len(topics) * n_per_topic
    for topic_idx, topic in enumerate(tqdm(topics, desc="Topics")):
        for local_idx in range(n_per_topic):
            scenario_index = topic_idx * n_per_topic + local_idx
            original_prompt = _writer_prompt(topic, scenario_index)
            prompt = original_prompt
            last_error: Exception | None = None
            attempts: list[dict[str, Any]] = []
            for attempt in range(max_retries + 1):
                raw = ""
                try:
                    total_generation_calls += 1
                    raw = backend.generate(prompt)
                    scenario = _validate_scenario(
                        _extract_json_object(raw), strict=strict_validation
                    )
                    scenario_id = f"s{scenario_index:05d}"
                    rows.extend(expand_scenario(
                        scenario_id, topic, scenario, split=split_map[topic]
                    ))
                    accepted_scenarios += 1
                    accepted_by_topic[topic] += 1
                    last_error = None
                    break
                except Exception as exc:  # preserve failed generations for audit
                    last_error = exc
                    attempts.append({
                        "attempt": attempt,
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_output": raw[:3000],
                    })
                    prompt = _repair_prompt(
                        original_prompt, raw, exc, attempt + 1
                    )
            if last_error is not None:
                failures.append({
                    "topic": topic,
                    "scenario_index": scenario_index,
                    "split": split_map[topic],
                    "error": repr(last_error),
                    "attempts": attempts,
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
    acceptance_rate = (
        float(accepted_scenarios / scenario_slots) if scenario_slots else 0.0
    )
    metadata = {
        "rows": len(rows),
        "rows_per_scenario": 10,
        "scenario_slots": scenario_slots,
        "scenarios": accepted_scenarios,
        "acceptance_rate": acceptance_rate,
        "failures": len(failures),
        "generation_calls": total_generation_calls,
        "topics": topics,
        "accepted_by_topic": accepted_by_topic,
        "topic_splits": split_map,
        "row_split_counts": split_counts,
        "scenarios_per_topic": n_per_topic,
        "writer_model": config["model"]["name"],
        "strict_validation": strict_validation,
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))

    minimum_acceptance_rate = float(
        corpus_cfg.get("minimum_acceptance_rate", 0.0)
    )
    require_topic_coverage = bool(
        corpus_cfg.get("require_topic_coverage", False)
    )
    missing_topics = [
        topic for topic, count in accepted_by_topic.items() if count == 0
    ]
    problems: list[str] = []
    if acceptance_rate < minimum_acceptance_rate:
        problems.append(
            f"acceptance_rate={acceptance_rate:.3f} < "
            f"minimum_acceptance_rate={minimum_acceptance_rate:.3f}"
        )
    if require_topic_coverage and missing_topics:
        problems.append(f"topics with zero accepted scenarios: {missing_topics}")
    if problems:
        raise RuntimeError(
            "Synthetic corpus failed its quality gate: " + "; ".join(problems)
            + f". Inspect {failure_path} and {metadata_path}."
        )
    completion_path.write_text(
        json.dumps({
            "output": str(output_path),
            "metadata": str(metadata_path),
            "acceptance_rate": acceptance_rate,
            "scenarios": accepted_scenarios,
        }, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(load_yaml(args.config))
