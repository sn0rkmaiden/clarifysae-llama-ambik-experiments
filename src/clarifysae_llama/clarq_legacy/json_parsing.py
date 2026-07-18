from __future__ import annotations

import ast
import json
import re
from typing import Any


def _extract_first_braced_object(text: str) -> str | None:
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    quote_char = ''
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == quote_char:
                in_string = False
        else:
            if ch in ('"', "'"):
                in_string = True
                quote_char = ch
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

    return None


def _extract_bool(text: str, key: str, default: bool) -> bool:
    pattern = rf'["\']?{re.escape(key)}["\']?\s*[:=]\s*(true|false|True|False)'
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return default
    return match.group(1).lower() == 'true'


def _extract_int(text: str, key: str, default: int) -> int:
    pattern = rf'["\']?{re.escape(key)}["\']?\s*[:=]\s*(-?\d+)'
    match = re.search(pattern, text)
    if not match:
        return default
    try:
        return int(match.group(1))
    except ValueError:
        return default


def _extract_string(text: str, key: str, default: str) -> str:
    # Quoted value
    pattern = rf'["\']?{re.escape(key)}["\']?\s*[:=]\s*([\'"])(.*?)\1'
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return match.group(2).strip()

    # Unquoted single-line value
    pattern = rf'["\']?{re.escape(key)}["\']?\s*[:=]\s*([^\n\r,}}]+)'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip().strip('"').strip("'")

    return default


def _schema_fallback(raw: str, prompt: str) -> dict[str, Any]:
    prompt_l = prompt.lower()

    if "'response'" in prompt or '"response"' in prompt:
        return {
            'response': _extract_string(
                raw,
                'response',
                'I am not sure. Could you clarify your question?',
            ),
            'index': _extract_int(raw, 'index', -1),
        }

    if "'type'" in prompt or '"type"' in prompt:
        return {
            'type': _extract_int(raw, 'type', 6),
        }

    if "'related'" in prompt or '"related"' in prompt:
        return {
            'related': _extract_bool(raw, 'related', True),
        }

    if "'repeat'" in prompt or '"repeat"' in prompt:
        return {
            'analysis': _extract_string(raw, 'analysis', ''),
            'repeat': _extract_bool(raw, 'repeat', False),
        }

    if "'answerable'" in prompt or '"answerable"' in prompt:
        return {
            'analysis': _extract_string(raw, 'analysis', ''),
            'answerable': _extract_bool(raw, 'answerable', False),
        }

    if "'correct'" in prompt or '"correct"' in prompt:
        return {
            'analysis': _extract_string(raw, 'analysis', ''),
            'correct': _extract_bool(raw, 'correct', False),
        }

    # Last-resort fallback
    return {
        'response': raw.strip() if raw.strip() else 'I am not sure. Could you clarify your question?',
        'index': -1,
    }


def parse_jsonish_response(raw: str, prompt: str) -> dict[str, Any]:
    text = (raw or '').strip()
    if not text:
        return _schema_fallback('', prompt)

    candidates: list[str] = []
    block = _extract_first_braced_object(text)
    if block:
        candidates.append(block)
    candidates.append(text)

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)

        normalized = (
            candidate
            .replace('“', '"')
            .replace('”', '"')
            .replace('’', "'")
        )

        try:
            obj = json.loads(normalized)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        try:
            obj = ast.literal_eval(normalized)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return _schema_fallback(text, prompt)