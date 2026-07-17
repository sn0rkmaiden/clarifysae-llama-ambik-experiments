from __future__ import annotations

from typing import Any, Iterable


class BackendLLMAdapter:
    """
    Adapter that makes the current backend look like the old ClarQ code's LLM interface.

    Goals:
    - keep only a single seeker turn for normal dialogue generation
    - preserve JSON-ish outputs for provider/judge prompts
    - accept legacy kwargs like previous_message
    """

    def __init__(self, backend: Any):
        self.backend = backend

    def _extract_first_braced_object(self, text: str) -> str | None:
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        quote_char = ""
        escape = False

        for i in range(start, len(text)):
            ch = text[i]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote_char:
                    in_string = False
            else:
                if ch in ('"', "'"):
                    in_string = True
                    quote_char = ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]

        return None

    def _coerce_json_text(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "{}"

        block = self._extract_first_braced_object(text)
        if block is not None:
            return block.strip()

        return text

    def _apply_stop(self, text: str, stop: str | Iterable[str] | None) -> str:
        if not text or stop is None:
            return text

        if isinstance(stop, str):
            stops = [stop]
        else:
            stops = [s for s in stop if s]

        cut = len(text)
        for marker in stops:
            idx = text.find(marker)
            if idx != -1:
                cut = min(cut, idx)

        return text[:cut]

    def _cut_at_any_marker(self, text: str, markers: list[str]) -> str:
        cut = len(text)
        for marker in markers:
            idx = text.find(marker)
            if idx != -1:
                cut = min(cut, idx)
        return text[:cut]

    def _strip_wrapping_quotes(self, text: str) -> str:
        text = text.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
            return text[1:-1].strip()
        return text

    def _truncate_to_single_turn(self, text: str) -> str:
        """
        Keep only the first actual seeker utterance.

        This removes:
        - fake continuation into another speaker turn
        - leaked prompt instructions
        - markdown/code artifacts
        - ClarQ template leftovers like 'Please respond...' / 'Please generate...'
        """
        text = (text or "").strip()

        # Hard cuts for extra dialogue turns
        turn_markers = [
            "\nJax:",
            "\nYou:",
            "\nUser:",
            "\nAssistant:",
            "\nHuman:",
        ]
        text = self._cut_at_any_marker(text, turn_markers).strip()

        # Hard cuts for leaked instructions / prompt remnants
        instruction_markers = [
            "\nPlease respond",
            "\nPlease generate",
            "\nRespond to Jax",
            "\nGenerate a response from Jax",
            "\nNow, based on the previous conversation",
            "\nThe final answer is:",
            "\n## ",
            "\n---",
            "\n(Note:",
            "\nNote:",
            "```",
            "Please respond to Jax's previous message",
            "Please generate a response from Jax",
            "Now, based on the previous conversation, generate a reply to Jax.",
            "The final answer is:",
            "## Step",
            "---",
            "(Note:",
            "Note:",
        ]
        text = self._cut_at_any_marker(text, instruction_markers).strip()

        # Remove common speaker prefixes if they remain
        prefixes = ["You:", "User:", "Assistant:", "Human:"]
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if text.startswith(prefix):
                    text = text[len(prefix) :].strip()
                    changed = True

        text = self._strip_wrapping_quotes(text)

        # Remove trailing unmatched quote if generation got cut mid-string
        if text.count('"') % 2 == 1 and text.endswith('"'):
            text = text[:-1].rstrip()
        if text.count("'") % 2 == 1 and text.endswith("'"):
            text = text[:-1].rstrip()

        return text


    def _normalize_chat_messages(self, previous_message: Any, prompt_text: str) -> list[dict[str, str]] | None:
        if not previous_message:
            return None
        if not isinstance(previous_message, list):
            return None

        messages: list[dict[str, str]] = []
        for message in previous_message:
            if not isinstance(message, dict):
                continue
            role = str(message.get('role', 'user')).strip().lower() or 'user'
            if role not in {'system', 'user', 'assistant'}:
                role = 'user'
            content = message.get('content', '')
            messages.append({'role': role, 'content': '' if content is None else str(content)})

        if not messages:
            return None

        prompt_text = (prompt_text or '').strip()
        if prompt_text:
            last_message = messages[-1]
            if not (last_message['role'] == 'user' and last_message['content'].strip() == prompt_text):
                messages.append({'role': 'user', 'content': prompt_text})

        return messages

    def request(
        self,
        prompt_text: str,
        stop: str | Iterable[str] | None = None,
        previous_message: Any = None,
        json_format: bool = False,
        **kwargs,
    ):
        """
        Returns (response_text, metadata) to match the legacy ClarQ interface.
        """
        _ = kwargs

        chat_messages = self._normalize_chat_messages(previous_message, prompt_text)
        if chat_messages is not None and hasattr(self.backend, 'generate_messages'):
            response = self.backend.generate_messages(chat_messages)
        else:
            # The in-repo HF backend does not implement provider-side stopping;
            # this adapter applies stop strings after generation below. Do not
            # pass legacy kwargs here, because catching TypeError hides real
            # generation bugs and adds overhead on every call.
            response = self.backend.generate(prompt_text)

        response = "" if response is None else str(response)
        response = self._apply_stop(response, stop)

        if json_format:
            response = self._coerce_json_text(response)
        else:
            response = self._truncate_to_single_turn(response)

        return response, None