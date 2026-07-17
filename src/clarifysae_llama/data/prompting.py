from __future__ import annotations


def _base_context(description: str, task: str) -> str:
    return (
        'You are a robot operating in an office kitchen. '
        'You are in front of a counter with two closed drawers, a top one and a bottom one. '
        'There is also a landfill bin, a recycling bin, and a compost bin.\n'
        f'Apart from that, in the kitchen there is {description}.\n'
        f'You have received the task "{task}".\n'
    )


LABEL_TEMPLATE = (
    '{context}'
    'Determine whether the task is ambiguous.\n'
    'Reply with exactly one word: AMBIGUOUS or CLEAR.\n\n'
    'Answer:'
)


QUESTION_TEMPLATE = (
    '{context}'
    'If the task is ambiguous, ask up to {max_questions} short clarifying questions that would help you perform it correctly.\n'
    'If the task is clear, output NONE.\n\n'
    'Return only the answer itself.\n'
    'Use one question per line.\n'
    'Do not output explanations, bullets, numbering, JSON, or repeated instructions.\n'
    'Ask at most {max_questions} questions.\n\n'
    'Answer:\n'
)


JSON_TEMPLATE = (
    '{context}'
    'Determine whether the task is ambiguous and, if it is, generate up to {max_questions} clarifying questions that would help you perform it correctly.\n\n'
    'Return only one valid JSON object with exactly these keys: "ambiguous" and "question".\n'
    'Use this schema exactly:\n'
    '{{\n'
    '  "ambiguous": true or false,\n'
    '  "question": ["question 1", "question 2"]\n'
    '}}\n\n'
    'Rules:\n'
    '- If the task is clear, output {{"ambiguous": false, "question": []}}.\n'
    '- If the task is ambiguous, output {{"ambiguous": true, "question": [...]}} with between 1 and {max_questions} questions.\n'
    '- Do not output markdown, code fences, examples, explanations, or any text before or after the JSON.\n'
    '- Do not repeat the task or the instructions.\n\n'
    'Answer:\n'
)


CLARIFICATION_TEMPLATE = JSON_TEMPLATE


def build_ambiguity_prompt(description: str, task: str) -> str:
    return LABEL_TEMPLATE.format(context=_base_context(description, task))


def build_question_prompt(description: str, task: str, max_questions: int = 3) -> str:
    return QUESTION_TEMPLATE.format(
        context=_base_context(description, task),
        max_questions=max_questions,
    )


def build_json_compliance_prompt(description: str, task: str, max_questions: int = 3) -> str:
    return JSON_TEMPLATE.format(
        context=_base_context(description, task),
        max_questions=max_questions,
    )


def build_clarification_prompt(description: str, task: str, max_questions: int = 3) -> str:
    return build_json_compliance_prompt(description=description, task=task, max_questions=max_questions)
