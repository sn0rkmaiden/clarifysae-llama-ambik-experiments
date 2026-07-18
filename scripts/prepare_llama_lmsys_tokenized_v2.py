#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

"""
1B model:
python scripts/prepare_llama_lmsys_tokenized_v2.py \
  --dataset lmsys/lmsys-chat-1m \
  --split train \
  --model unsloth/Llama-3.2-1B-Instruct \
  --output-dir data/processed/lmsys_llama32_1b_tokenized \
  --conversation-column conversation \
  --language English \
  --max-samples 4096 \
  --max-length 2048 \
  --streaming

8B model (base):
python scripts/prepare_llama_lmsys_tokenized_v2.py \
  --dataset lmsys/lmsys-chat-1m \
  --split train \
  --model unsloth/Llama-3.1-8B \
  --output-dir data/processed/lmsys_llama31_8b_tokenized \
  --conversation-column conversation \
  --language English \
  --max-samples 4096 \
  --max-length 2048 \
  --streaming

8B model (instruct):
python scripts/prepare_llama_lmsys_tokenized_v2.py \
  --dataset lmsys/lmsys-chat-1m \
  --split train \
  --model unsloth/Llama-3.1-8B-Instruct \
  --output-dir data/processed/lmsys_llama31_8b_tokenized \
  --conversation-column conversation \
  --language English \
  --max-samples 4096 \
  --max-length 2048 \
  --streaming
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--conversation-column", type=str, default="conversation")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--min-tokens", type=int, default=2)
    parser.add_argument("--streaming", action="store_true")
    return parser.parse_args()


def normalize_messages(sample: dict[str, Any], conversation_column: str) -> list[dict[str, str]]:
    conv = sample.get(conversation_column)

    if isinstance(conv, str):
        try:
            conv = json.loads(conv)
        except Exception:
            return []

    if isinstance(conv, dict):
        for key in ["messages", "conversation", "turns", "chat"]:
            if key in conv and isinstance(conv[key], list):
                conv = conv[key]
                break

    if not isinstance(conv, list):
        return []

    messages: list[dict[str, str]] = []
    for msg in conv:
        if isinstance(msg, str):
            text = msg.strip()
            if text:
                messages.append({"role": "user", "content": text})
            continue

        if not isinstance(msg, dict):
            continue

        role = (
            msg.get("role")
            or msg.get("from")
            or msg.get("speaker")
            or msg.get("author")
            or msg.get("name")
            or "user"
        )
        content = (
            msg.get("content")
            or msg.get("value")
            or msg.get("text")
            or msg.get("message")
            or msg.get("utterance")
            or ""
        )

        if isinstance(content, list):
            content = " ".join(str(x) for x in content)
        elif not isinstance(content, str):
            content = str(content)

        role_l = str(role).lower()
        if role_l in {"assistant", "gpt", "model", "bot", "chatgpt"}:
            role = "assistant"
        elif role_l == "system":
            role = "system"
        else:
            role = "user"

        content = content.strip()
        if content:
            messages.append({"role": role, "content": content})

    return messages


def render_conversation_text(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for m in messages:
        parts.append(f"{m['role']}: {m['content']}")
    return "\n".join(parts)


def tokenize_messages(tokenizer, messages: list[dict[str, str]], max_length: int) -> tuple[list[int], str]:
    text = ""
    tokens: list[int] = []

    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            text = rendered
            toks = tokenizer(
                rendered,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )["input_ids"]
            tokens = list(map(int, toks))
            return tokens, text
    except Exception:
        pass

    text = render_conversation_text(messages)
    toks = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )["input_ids"]
    tokens = list(map(int, toks))
    return tokens, text


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    ds = load_dataset(args.dataset, split=args.split, streaming=args.streaming)

    kept_rows: list[dict[str, Any]] = []
    seen = 0

    iterator = ds
    pbar = tqdm(total=args.max_samples, desc="Tokenizing LMSYS with Llama tokenizer")

    for row in iterator:
        if args.language is not None:
            row_lang = row.get("language")
            if row_lang != args.language:
                continue

        messages = normalize_messages(row, args.conversation_column)
        if not messages:
            continue

        try:
            tokens, text = tokenize_messages(tokenizer, messages, args.max_length)
        except Exception:
            continue

        if len(tokens) < args.min_tokens:
            continue

        kept_rows.append(
            {
                "conversation_id": row.get("conversation_id"),
                "model": row.get("model"),
                "language": row.get("language"),
                "text": text,
                "tokens": tokens,
            }
        )

        seen += 1
        pbar.update(1)
        if seen >= args.max_samples:
            break

    pbar.close()

    dataset = Dataset.from_list(kept_rows)

    dataset_dir = output_dir / "dataset"
    dataset.save_to_disk(str(dataset_dir))

    parquet_path = output_dir / "train.parquet"
    dataset.to_parquet(str(parquet_path))

    metadata = {
        "source_dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "conversation_column": args.conversation_column,
        "language": args.language,
        "max_samples": args.max_samples,
        "max_length": args.max_length,
        "min_tokens": args.min_tokens,
        "streaming": args.streaming,
        "saved_examples": len(kept_rows),
        "columns": list(dataset.column_names),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved {len(kept_rows)} examples to {output_dir}")
    print("Use column_name: tokens in your discovery config.")


if __name__ == "__main__":
    main()