from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f'Unsupported torch dtype: {dtype_name}')
    return mapping[dtype_name]


def normalize_generation_kwargs(generation_kwargs: dict[str, Any], tokenizer) -> dict[str, Any]:
    kwargs = dict(generation_kwargs)

    if 'max_tokens' in kwargs and 'max_new_tokens' not in kwargs:
        kwargs['max_new_tokens'] = kwargs.pop('max_tokens')

    # Avoid repeated HF warning:
    # "Both `max_new_tokens` and `max_length` seem to have been set"
    if 'max_new_tokens' in kwargs and 'max_length' in kwargs:
        kwargs.pop('max_length', None)

    kwargs.pop('logprobs', None)
    kwargs.pop('logit_bias', None)

    temperature = kwargs.get('temperature', None)
    if temperature is not None and float(temperature) == 0.0:
        kwargs['do_sample'] = False
        kwargs.pop('temperature', None)
        kwargs.pop('top_p', None)

    if kwargs.get('pad_token_id') is None:
        kwargs['pad_token_id'] = tokenizer.pad_token_id
    if kwargs.get('eos_token_id') is None:
        kwargs['eos_token_id'] = tokenizer.eos_token_id

    return kwargs


class HFCausalBackend:
    def __init__(self, config: dict[str, Any]):
        model_cfg = config['model']
        generation_cfg = config['generation']
        prompting_cfg = config.get('prompting', {})

        self.model_name = model_cfg['name']
        self.dtype = _resolve_torch_dtype(model_cfg.get('torch_dtype', 'bfloat16'))
        self.chat_template_mode = prompting_cfg.get('use_chat_template', 'auto')
        self.system_prompt = prompting_cfg.get(
            'system_prompt',
            'You are a careful assistant. Follow the user instruction exactly and return only the requested output format.',
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Llama-style causal generation should be left-padded in batch mode.
        self.tokenizer.padding_side = 'left'
        truncation_side = prompting_cfg.get('truncation_side')
        if truncation_side is not None:
            truncation_side = str(truncation_side).strip().lower()
            if truncation_side not in {'left', 'right'}:
                raise ValueError(
                    f'Unsupported tokenizer truncation side: {truncation_side!r}'
                )
            self.tokenizer.truncation_side = truncation_side

        model_kwargs: dict[str, Any] = {'torch_dtype': self.dtype}
        if model_cfg.get('device_map', None) is not None:
            model_kwargs['device_map'] = model_cfg['device_map']
        if model_cfg.get('attn_implementation', None) is not None:
            model_kwargs['attn_implementation'] = model_cfg['attn_implementation']
        if model_cfg.get('low_cpu_mem_usage', None) is not None:
            model_kwargs['low_cpu_mem_usage'] = bool(model_cfg['low_cpu_mem_usage'])
        if model_cfg.get('max_memory', None) is not None:
            model_kwargs['max_memory'] = model_cfg['max_memory']
        if model_cfg.get('offload_folder', None) is not None:
            model_kwargs['offload_folder'] = model_cfg['offload_folder']

        if bool(model_cfg.get('load_in_4bit', False)):
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_quant_type=model_cfg.get('bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_use_double_quant=bool(model_cfg.get('bnb_4bit_use_double_quant', True)),
            )
        elif bool(model_cfg.get('load_in_8bit', False)):
            model_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()

        self.generation_kwargs = normalize_generation_kwargs(generation_cfg, self.tokenizer)

    def _model_input_device(self) -> torch.device:
        # With device_map='auto' some parameters may be meta/CPU-offloaded. The
        # embedding weight's concrete device is a better target for input IDs.
        try:
            embed = self.model.get_input_embeddings()
            if embed is not None and hasattr(embed, 'weight') and embed.weight.device.type != 'meta':
                return embed.weight.device
        except Exception:
            pass

        for parameter in self.model.parameters():
            if parameter.device.type != 'meta':
                return parameter.device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _inputs_to_model_device(self, tokenized):
        model_device = self._model_input_device()
        return {k: v.to(model_device) for k, v in tokenized.items()}

    def _should_use_chat_template(self) -> bool:
        mode = self.chat_template_mode
        if isinstance(mode, bool):
            requested = mode
        else:
            lowered = str(mode).strip().lower()
            if lowered in {'true', '1', 'yes', 'on'}:
                requested = True
            elif lowered in {'false', '0', 'no', 'off'}:
                requested = False
            else:
                requested = bool(getattr(self.tokenizer, 'chat_template', None)) and any(
                    token in self.model_name.lower() for token in ('instruct', 'chat')
                )
        return requested and bool(getattr(self.tokenizer, 'chat_template', None))

    def _stringify_message_content(self, content: Any) -> str:
        if content is None:
            return ''
        if isinstance(content, str):
            return content
        return str(content)

    def _normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise TypeError(f'Message must be a dict, got {type(message)!r}')
            role = str(message.get('role', 'user')).strip().lower() or 'user'
            if role not in {'system', 'user', 'assistant'}:
                role = 'user'
            normalized.append({
                'role': role,
                'content': self._stringify_message_content(message.get('content', '')),
            })
        return normalized

    def _flatten_messages(self, messages: list[dict[str, str]]) -> str:
        role_labels = {
            'system': 'System',
            'user': 'User',
            'assistant': 'Assistant',
        }
        parts: list[str] = []
        if self.system_prompt and not any(msg['role'] == 'system' for msg in messages):
            parts.append(f"System: {self.system_prompt}")
        for message in messages:
            role_label = role_labels.get(message['role'], 'User')
            parts.append(f"{role_label}: {message['content']}")
        parts.append('Assistant:')
        return '\n\n'.join(parts)

    def _format_messages(self, messages: list[dict[str, Any]]) -> str:
        normalized = self._normalize_messages(messages)
        if not self._should_use_chat_template():
            return self._flatten_messages(normalized)

        if self.system_prompt and not any(msg['role'] == 'system' for msg in normalized):
            normalized = [{'role': 'system', 'content': self.system_prompt}, *normalized]

        return self.tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _format_prompt(self, prompt: str) -> str:
        return self._format_messages([{'role': 'user', 'content': prompt}])

    def _format_prompts(self, prompts: list[str]) -> list[str]:
        return [self._format_prompt(prompt) for prompt in prompts]

    @staticmethod
    def _decode_new_tokens(tokenizer, sequence: torch.Tensor, prompt_width: int) -> str:
        continuation_ids = sequence[int(prompt_width):]
        return tokenizer.decode(
            continuation_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        prompt = self._format_prompt(prompt)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = self._inputs_to_model_device(inputs)
        output = self.model.generate(**inputs, **self.generation_kwargs)
        prompt_width = int(inputs['input_ids'].shape[1])
        return self._decode_new_tokens(self.tokenizer, output[0], prompt_width)

    @torch.inference_mode()
    def generate_messages(self, messages: list[dict[str, Any]]) -> str:
        prompt = self._format_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = self._inputs_to_model_device(inputs)
        output = self.model.generate(**inputs, **self.generation_kwargs)
        prompt_width = int(inputs['input_ids'].shape[1])
        return self._decode_new_tokens(self.tokenizer, output[0], prompt_width)


    def candidate_tokenization(self, candidates: list[str]) -> list[dict[str, Any]]:
        """Return tokenization diagnostics for fixed verbalizers."""
        diagnostics: list[dict[str, Any]] = []
        for candidate in candidates:
            token_ids = self.tokenizer(
                candidate,
                add_special_tokens=False,
            )['input_ids']
            diagnostics.append({
                'candidate': candidate,
                'token_ids': [int(token_id) for token_id in token_ids],
                'tokens': self.tokenizer.convert_ids_to_tokens(token_ids),
                'n_tokens': len(token_ids),
            })
        return diagnostics

    @torch.inference_mode()
    def score_next_token_candidates_batch(
        self,
        prompts: list[str],
        candidates: list[str],
    ) -> list[list[float]]:
        """Score single-token verbalizers at the next generation position.

        This is the appropriate scorer for CLAM's binary ambiguity decision.
        It avoids comparing mean log-probabilities of differently tokenized
        words such as ``AMBIGUOUS`` and ``CLEAR``. Every candidate must be a
        single token under the current model tokenizer.
        """
        if not prompts:
            return []
        if not candidates:
            raise ValueError('At least one candidate continuation is required.')

        diagnostics = self.candidate_tokenization(candidates)
        invalid = [item for item in diagnostics if item['n_tokens'] != 1]
        if invalid:
            details = '; '.join(
                f"{item['candidate']!r} -> {item['tokens']}"
                for item in invalid
            )
            raise ValueError(
                'CLAM next-token candidates must each tokenize to exactly one '
                f'token. Invalid verbalizers: {details}'
            )

        formatted_prompts = self._format_prompts(prompts)
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
        )
        inputs = self._inputs_to_model_device(inputs)
        outputs = self.model(**inputs)

        # Prompts are left padded, so the last column is the final real prompt
        # token for every row. Its logits predict the first generated token.
        next_token_log_probs = torch.log_softmax(
            outputs.logits[:, -1, :].float(),
            dim=-1,
        )
        candidate_ids = torch.tensor(
            [item['token_ids'][0] for item in diagnostics],
            dtype=torch.long,
            device=next_token_log_probs.device,
        )
        selected = next_token_log_probs.index_select(dim=-1, index=candidate_ids)
        return selected.detach().cpu().tolist()



    @torch.inference_mode()
    def score_candidate_continuations_batch(
        self,
        prompts: list[str],
        candidates: list[str],
        *,
        length_normalize: bool = True,
    ) -> list[list[float]]:
        """Score fixed candidate continuations after each prompt.

        The return value has shape ``[len(prompts), len(candidates)]``. Each
        score is the conditional log-probability of the candidate tokens given
        the formatted prompt. Mean token log-probability is used by default so
        candidates with different token lengths remain comparable.
        """
        if not prompts:
            return []
        if not candidates:
            raise ValueError('At least one candidate continuation is required.')

        formatted_prompts = self._format_prompts(prompts)
        examples: list[tuple[int, int, list[int], list[int]]] = []
        for prompt_idx, formatted_prompt in enumerate(formatted_prompts):
            prompt_ids = self.tokenizer(
                formatted_prompt,
                add_special_tokens=False,
            )['input_ids']
            if not prompt_ids:
                raise ValueError('Formatted prompt tokenized to an empty sequence.')

            for candidate_idx, candidate in enumerate(candidates):
                candidate_ids = self.tokenizer(
                    candidate,
                    add_special_tokens=False,
                )['input_ids']
                if not candidate_ids:
                    raise ValueError(f'Candidate {candidate!r} tokenized to an empty sequence.')
                examples.append((prompt_idx, candidate_idx, prompt_ids, candidate_ids))

        pad_id = int(self.tokenizer.pad_token_id)
        max_len = max(len(prompt_ids) + len(candidate_ids) for _, _, prompt_ids, candidate_ids in examples)
        input_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        label_rows: list[list[int]] = []

        for _, _, prompt_ids, candidate_ids in examples:
            full_ids = [*prompt_ids, *candidate_ids]
            left_pad = max_len - len(full_ids)
            input_rows.append([pad_id] * left_pad + full_ids)
            mask_rows.append([0] * left_pad + [1] * len(full_ids))
            label_rows.append([-100] * (left_pad + len(prompt_ids)) + candidate_ids)

        device = self._model_input_device()
        input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
        attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)
        labels = torch.tensor(label_rows, dtype=torch.long, device=device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        valid = shifted_labels.ne(-100)
        safe_labels = shifted_labels.masked_fill(~valid, 0)
        token_log_probs = torch.log_softmax(logits.float(), dim=-1)
        selected = token_log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        selected = selected.masked_fill(~valid, 0.0)

        sequence_scores = selected.sum(dim=-1)
        if length_normalize:
            lengths = valid.sum(dim=-1).clamp_min(1)
            sequence_scores = sequence_scores / lengths

        matrix = [[float('-inf')] * len(candidates) for _ in prompts]
        for example_idx, (prompt_idx, candidate_idx, _, _) in enumerate(examples):
            matrix[prompt_idx][candidate_idx] = float(sequence_scores[example_idx].item())
        return matrix

    @torch.inference_mode()
    def generate_batch(self, prompts: list[str]) -> list[str]:
        prompts = self._format_prompts(prompts)
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
        inputs = self._inputs_to_model_device(inputs)
        outputs = self.model.generate(**inputs, **self.generation_kwargs)

        # In batched generation with left padding, new tokens begin after the
        # full padded prompt width for every row, not after each row's
        # non-padding token count.
        prompt_width = int(inputs['input_ids'].shape[1])
        return [
            self._decode_new_tokens(self.tokenizer, outputs[row_idx], prompt_width)
            for row_idx in range(outputs.shape[0])
        ]
