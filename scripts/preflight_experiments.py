#!/usr/bin/env python3
from __future__ import annotations

"""Static preflight checks for the ClarifySAE experiment repository."""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import yaml


OPTIONAL_IMPORTS = [
    "torch", "transformers", "accelerate", "datasets", "sentence_transformers",
    "sparsify", "sae_lens", "dictionary_learning",
]


def _walk(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, value
            yield from _walk(value, path)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            path = f"{prefix}[{idx}]"
            yield path, value
            yield from _walk(value, path)


def _is_local_input_key(key: str) -> bool:
    leaf = key.rsplit(".", 1)[-1]
    return leaf in {
        "path", "dataset_path", "vector_path", "base_config",
    }


def run(root: Path) -> dict[str, Any]:
    yaml_files = sorted(root.glob("configs/**/*.yaml"))
    parse_errors: list[dict[str, str]] = []
    missing_inputs: list[dict[str, str]] = []
    generated_prerequisites: list[dict[str, str]] = []
    stale_terms: list[dict[str, str]] = []

    for path in yaml_files:
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            parse_errors.append({"config": str(path.relative_to(root)), "error": repr(exc)})
            continue
        for key, value in _walk(cfg):
            if isinstance(value, str) and "ask_policy" in value:
                stale_terms.append({"config": str(path.relative_to(root)), "key": key, "value": value})
            if not isinstance(value, str) or not _is_local_input_key(key):
                continue
            if value.startswith(("http://", "https://")):
                continue
            candidate = root / value
            if candidate.exists():
                continue
            record = {"config": str(path.relative_to(root)), "key": key, "path": value}
            if value.startswith("outputs/"):
                generated_prerequisites.append(record)
            elif value in {"null", "None", ""}:
                continue
            else:
                missing_inputs.append(record)

    imports = {
        name: bool(importlib.util.find_spec(name)) for name in OPTIONAL_IMPORTS
    }
    cuda = {"torch_available": imports.get("torch", False), "cuda_available": False, "device_count": 0}
    if imports.get("torch"):
        import torch
        cuda.update({
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        })

    return {
        "root": str(root),
        "yaml_files": len(yaml_files),
        "yaml_parse_errors": parse_errors,
        "missing_local_inputs": missing_inputs,
        "generated_prerequisites_not_yet_present": generated_prerequisites,
        "stale_ask_policy_references": stale_terms,
        "imports": imports,
        "runtime": cuda,
        "status": "ok" if not parse_errors and not stale_terms else "needs_attention",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run(Path(args.root).resolve())
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
