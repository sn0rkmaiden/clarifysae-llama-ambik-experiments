from __future__ import annotations

from typing import Any


CLASSIFICATION_HEADER = """You are evaluating instructions for a kitchen robot.
Judge whether the instruction leaves a task-relevant choice or condition unspecified in the described environment.

Answer True if more than one plausible execution remains and the missing information should be clarified before acting.
Answer False if the instruction itself specifies a single sufficiently determined execution.

Return only True or False. Do not explain your answer.
"""


QUESTION_HEADER = """You are a kitchen robot. The instruction below has been classified as ambiguous.
Ask exactly one concise clarification question that identifies the missing choice needed to execute the task correctly.
Ground the question in the described environment and mention concrete alternatives when useful.
Return only the question. Do not add an answer, explanation, list, numbering, or JSON.
"""


def _format_case(environment: str, task: str) -> str:
    return f"Environment: {environment}\nInstruction: {task}"


def build_clam_classification_prompt(
    *,
    environment: str,
    task: str,
    demonstrations: list[dict[str, Any]],
) -> str:
    """Build the CLAM-style binary ambiguity prompt.

    Demonstration files keep the human-readable labels ``AMBIGUOUS`` and
    ``CLEAR``. They are rendered as ``True`` and ``False`` because the
    classifier is scored from the next-token probabilities of those two
    verbalizers, following the decomposition used by CLAM.
    """
    parts = [CLASSIFICATION_HEADER]
    for demo in demonstrations:
        label = str(demo['label']).strip().upper()
        if label not in {'AMBIGUOUS', 'CLEAR'}:
            raise ValueError(f"Unsupported CLAM demonstration label: {label!r}")
        truth_value = 'True' if label == 'AMBIGUOUS' else 'False'
        parts.append(
            _format_case(str(demo['environment']), str(demo['task']))
            + f"\nThis instruction is ambiguous: {truth_value}\n"
        )
    parts.append(
        _format_case(environment, task)
        + "\nThis instruction is ambiguous:"
    )
    return "\n".join(parts)


def build_clam_question_prompt(
    *,
    environment: str,
    task: str,
    demonstrations: list[dict[str, Any]],
) -> str:
    parts = [QUESTION_HEADER]
    for demo in demonstrations:
        if str(demo.get('label', '')).strip().upper() != 'AMBIGUOUS':
            continue
        question = str(demo.get('question', '')).strip()
        if not question:
            continue
        parts.append(
            _format_case(str(demo['environment']), str(demo['task']))
            + f"\nClarification question: {question}\n"
        )
    parts.append(_format_case(environment, task) + "\nClarification question:")
    return "\n".join(parts)
