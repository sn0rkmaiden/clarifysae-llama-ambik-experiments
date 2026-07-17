from __future__ import annotations

import json
import re
from typing import Any


# QUESTION_BULLET_PREFIX_RE = re.compile(r'^\s*(?:[-*•]+|\d+[\).:-]?|[A-Za-z][\).:-]?)\s*')
QUESTION_BULLET_PREFIX_RE = re.compile(r'^\s*(?:[-*•]+|\d+[\).:-]?)\s*')


def _strip_fences_and_eos(text: str) -> str:
    text = (text or '').replace('<eos>', '').strip()

    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.I)
    if fenced:
        return fenced.group(1).strip()

    text = re.sub(r"```(?:json)?", '', text, flags=re.I)
    text = text.replace('```', '')
    return text.strip()



def _first_curly_to_end(text: str) -> str | None:
    start = text.find('{')
    return text[start:] if start != -1 else None



def _extract_first_balanced_json_object(text: str) -> str | None:
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaping = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if in_string:
            if escaping:
                escaping = False
            elif ch == '\\':
                escaping = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]

    return None



def _balance_closers(text: str) -> str:
    in_string = False
    escaping = False
    braces = 0
    brackets = 0

    for ch in text:
        if ch == '"' and not escaping:
            in_string = not in_string
        escaping = (ch == '\\' and not escaping) if in_string else False
        if in_string:
            continue
        if ch == '{':
            braces += 1
        elif ch == '}':
            braces = max(0, braces - 1)
        elif ch == '[':
            brackets += 1
        elif ch == ']':
            brackets = max(0, brackets - 1)

    text = text.rstrip()
    if brackets > 0:
        text += ']' * brackets
    if braces > 0:
        text += '}' * braces
    return text



def _scan_string_list(text: str, start_idx: int) -> tuple[list[str], int]:
    items: list[str] = []
    current: list[str] = []
    in_string = False
    escaping = False
    idx = start_idx

    while idx < len(text):
        ch = text[idx]

        if in_string:
            if escaping:
                current.append(ch)
                escaping = False
            elif ch == '\\':
                escaping = True
            elif ch == '"':
                in_string = False
                items.append(''.join(current))
                current = []
            else:
                current.append(ch)
            idx += 1
            continue

        if ch == '"':
            in_string = True
            idx += 1
            continue
        if ch == ',':
            if current:
                items.append(''.join(current))
                current = []
            idx += 1
            continue
        if ch == ']':
            if current:
                items.append(''.join(current))
            idx += 1
            break
        if ch in '{}':
            if current:
                items.append(''.join(current))
            break
        idx += 1

    return items, idx



def _normalize_json_questions(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return []



def _coerce_json_schema_dict(obj: Any) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None

    ambiguous = obj.get('ambiguous')
    if isinstance(ambiguous, str):
        lowered = ambiguous.strip().lower()
        if lowered in {'true', 'false'}:
            ambiguous = lowered == 'true'
    if not isinstance(ambiguous, bool):
        return None

    questions_field = obj.get('question', obj.get('questions', []))
    questions = _normalize_json_questions(questions_field)

    return {
        'ambiguous': ambiguous,
        'question': questions,
    }



def _schema_parse_fallback(text: str) -> dict[str, Any] | None:
    ambiguous_match = re.search(r'"ambiguous"\s*:\s*(true|false)', text, flags=re.I)
    question_start = re.search(r'"question"\s*:\s*\[', text, flags=re.I)

    if ambiguous_match is None and question_start is None:
        return None

    ambiguous = None
    if ambiguous_match is not None:
        ambiguous = ambiguous_match.group(1).lower() == 'true'

    questions: list[str] = []
    if question_start:
        questions, _ = _scan_string_list(text, question_start.end())

    if not isinstance(ambiguous, bool):
        return None

    return {
        'ambiguous': ambiguous,
        'question': [question.strip() for question in questions if question.strip()],
    }



def _json_candidate(raw_output: str) -> str:
    body = _strip_fences_and_eos(raw_output)
    candidate = _extract_first_balanced_json_object(body)
    if candidate is None:
        candidate = _first_curly_to_end(body) or body
    return candidate



def parse_model_json(raw_output: str) -> dict[str, Any] | None:
    candidate = _json_candidate(raw_output)

    try:
        return _coerce_json_schema_dict(json.loads(candidate))
    except Exception:
        pass

    repaired = re.sub(r',(\s*[}\]])', r'\1', candidate)
    repaired = re.sub(r'\bTrue\b', 'true', repaired)
    repaired = re.sub(r'\bFalse\b', 'false', repaired)
    repaired = re.sub(r'\bNone\b', 'null', repaired)
    repaired = _balance_closers(repaired)

    try:
        return _coerce_json_schema_dict(json.loads(repaired))
    except Exception:
        pass

    return _schema_parse_fallback(repaired)



def parse_model_json_strict(raw_output: str) -> dict[str, Any] | None:
    body = _strip_fences_and_eos(raw_output)
    try:
        return _coerce_json_schema_dict(json.loads(body))
    except Exception:
        return None



def assess_json_output(raw_output: str) -> dict[str, Any]:
    strict = parse_model_json_strict(raw_output)
    recovered = parse_model_json(raw_output)
    return {
        'json_exact_valid': strict is not None,
        'json_schema_valid': recovered is not None,
        'json_recoverable_parse': recovered is not None and strict is None,
        'json_parsed_output': strict if strict is not None else recovered,
    }



def parse_label_output(raw_output: str) -> bool | None:
    body = _strip_fences_and_eos(raw_output)
    if not body:
        return None

    normalized = re.sub(r'\s+', ' ', body).strip()
    upper = normalized.upper().strip(' .,:;!')
    if upper == 'AMBIGUOUS':
        return True
    if upper in {'CLEAR', 'NOT AMBIGUOUS', 'UNAMBIGUOUS'}:
        return False

    lowered = normalized.lower()
    if re.search(r'\b(not ambiguous|unambiguous|not unclear|clear)\b', lowered):
        return False
    if re.search(r'\bambiguous\b', lowered):
        return True
    return None



def _clean_question_text(text: str) -> str:
    cleaned = QUESTION_BULLET_PREFIX_RE.sub('', text.strip())
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned



def extract_questions(raw_output: str, max_questions: int = 3) -> list[str]:
    body = _strip_fences_and_eos(raw_output)
    if not body:
        return []

    if re.fullmatch(r'(?is)\s*(?:none|no questions?)\.?\s*', body):
        return []

    questions: list[str] = []

    for line in body.splitlines():
        if '?' not in line:
            continue
        for match in re.findall(r'[^?\n]*\?', line):
            cleaned = _clean_question_text(match)
            if cleaned:
                questions.append(cleaned)

    if not questions and '?' in body:
        for match in re.findall(r'[^?]*\?', body):
            cleaned = _clean_question_text(match)
            if cleaned:
                questions.append(cleaned)

    deduped: list[str] = []
    seen: set[str] = set()
    for question in questions:
        key = question.lower()
        if key in seen:
            continue
        deduped.append(question)
        seen.add(key)
        if len(deduped) >= max_questions:
            break

    return deduped
