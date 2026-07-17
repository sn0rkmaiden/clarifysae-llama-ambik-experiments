from __future__ import annotations

"""Extract dense clarification concept vectors from a labeled JSONL corpus."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from clarifysae_llama.backends.hf_backend import HFCausalBackend
from clarifysae_llama.config import load_yaml
from clarifysae_llama.discovery.concept_vectors import (
    cosine_similarity,
    difference_in_means,
    fit_ridge_probe,
    paired_difference_in_means,
    principal_components_for_variance,
    remove_subspace,
)
from clarifysae_llama.steering.sparsify_steerer import (
    get_submodule_by_path,
    resolve_module_path,
)
from clarifysae_llama.utils.io import ensure_dir
from clarifysae_llama.utils.seed import set_seed


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"Line {line_no} is not a JSON object")
            rows.append(obj)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


class MultiHookCapture:
    def __init__(self, model, hookpoints: list[str], module_paths: dict[str, str] | None = None):
        self.model = model
        self.hookpoints = hookpoints
        self.module_paths = module_paths or {}
        self.handles = []
        self.outputs: dict[str, torch.Tensor] = {}

    def __enter__(self):
        for hookpoint in self.hookpoints:
            path = resolve_module_path(hookpoint, self.module_paths.get(hookpoint))
            module = get_submodule_by_path(self.model, path)

            def hook(_module, _inputs, output, hp=hookpoint):
                hidden = output[0] if isinstance(output, tuple) else output
                self.outputs[hp] = hidden.detach()

            self.handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def pop(self) -> dict[str, torch.Tensor]:
        if set(self.outputs) != set(self.hookpoints):
            missing = set(self.hookpoints) - set(self.outputs)
            raise RuntimeError(f"Missing hook outputs for {sorted(missing)}")
        outputs = self.outputs
        self.outputs = {}
        return outputs


def _assistant_start_char(text: str) -> int:
    marker = "\nAssistant:"
    idx = text.rfind(marker)
    if idx < 0:
        return len(text)
    return idx + len(marker)


def _pool_hidden(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer,
    texts: list[str],
    mode: str,
    mean_after_token: int,
) -> torch.Tensor:
    pooled: list[torch.Tensor] = []
    mode = mode.strip().lower().replace("-", "_")
    for i in range(hidden.shape[0]):
        valid_len = int(attention_mask[i].sum().item())
        h = hidden[i, :valid_len]
        if mode == "last_nonpad":
            pooled.append(h[-1])
        elif mode == "mean_all":
            pooled.append(h.mean(dim=0))
        elif mode == "mean_after_token":
            start = min(max(int(mean_after_token), 0), max(valid_len - 1, 0))
            pooled.append(h[start:].mean(dim=0))
        elif mode == "assistant_mean":
            # Re-tokenize the prefix only. Since extraction uses right padding,
            # its token count is a stable response-span boundary.
            prefix = texts[i][:_assistant_start_char(texts[i])]
            start = len(tokenizer(prefix, add_special_tokens=True)["input_ids"])
            start = min(max(start, 0), max(valid_len - 1, 0))
            pooled.append(h[start:].mean(dim=0))
        else:
            raise ValueError(f"Unsupported pooling mode: {mode}")
    return torch.stack(pooled, dim=0).float().cpu()


def _safe_key(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _paired_matrices(rows: list[dict[str, Any]], acts: torch.Tensor, pair_field: str):
    by_pair: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
    for row, act in zip(rows, acts):
        by_pair[str(row[pair_field])][int(row["label"])] = act
    positives, negatives = [], []
    for pair in by_pair.values():
        if 1 in pair and 0 in pair:
            positives.append(pair[1])
            negatives.append(pair[0])
    if not positives:
        raise ValueError("No complete positive/negative pairs found")
    return torch.stack(positives), torch.stack(negatives)


def run(config: dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    cfg = config["concept_discovery"]
    rows = _read_jsonl(cfg["dataset_path"])
    concepts_filter = cfg.get("concepts")
    if concepts_filter:
        allowed = set(map(str, concepts_filter))
        rows = [row for row in rows if str(row.get("concept")) in allowed]
    if not rows:
        raise ValueError("No rows remain after concept filtering")

    backend = HFCausalBackend(config)
    tokenizer, model = backend.tokenizer, backend.model
    tokenizer.padding_side = "right"
    hookpoints = list(cfg["hookpoints"])
    module_paths = dict(cfg.get("module_paths", {}))
    batch_size = int(cfg.get("batch_size", 8))
    max_length = int(cfg.get("max_length", 512))
    default_pooling = str(cfg.get("pooling", "recommended"))
    mean_after_token = int(cfg.get("mean_after_token", 50))

    activation_rows: dict[str, list[torch.Tensor]] = {hp: [] for hp in hookpoints}
    model_device = backend._model_input_device()
    with MultiHookCapture(model, hookpoints, module_paths) as capture:
        for start in tqdm(range(0, len(rows), batch_size), desc="Extracting residual activations"):
            batch_rows = rows[start:start + batch_size]
            texts = [str(row["text"]) for row in batch_rows]
            tokenized = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=max_length, add_special_tokens=True,
            )
            tokenized = {k: v.to(model_device) for k, v in tokenized.items()}
            with torch.inference_mode():
                _ = model(**tokenized, use_cache=False)
            outputs = capture.pop()
            for hp, hidden in outputs.items():
                # Recommended pooling can differ per corpus row; group modes in-batch.
                if default_pooling == "recommended":
                    pooled = torch.empty((len(batch_rows), hidden.shape[-1]), dtype=torch.float32)
                    mode_groups: dict[str, list[int]] = defaultdict(list)
                    for i, row in enumerate(batch_rows):
                        mode_groups[str(row.get("recommended_pooling", "last_nonpad"))].append(i)
                    for mode, indices in mode_groups.items():
                        hidden_index = torch.tensor(indices, device=hidden.device, dtype=torch.long)
                        token_index = torch.tensor(indices, device=tokenized["attention_mask"].device, dtype=torch.long)
                        sub = _pool_hidden(
                            hidden.index_select(0, hidden_index),
                            tokenized["attention_mask"].index_select(0, token_index),
                            tokenized["input_ids"].index_select(0, token_index),
                            tokenizer,
                            [texts[i] for i in indices],
                            mode,
                            mean_after_token,
                        )
                        pooled[indices] = sub
                else:
                    pooled = _pool_hidden(
                        hidden, tokenized["attention_mask"], tokenized["input_ids"],
                        tokenizer, texts, default_pooling, mean_after_token,
                    )
                activation_rows[hp].append(pooled)

    activations = {hp: torch.cat(parts, dim=0) for hp, parts in activation_rows.items()}
    pca_cfg = cfg.get("neutral_pca", {})
    neutral_concept = pca_cfg.get("concept")
    neutral_fraction = float(pca_cfg.get("variance_fraction", 0.5))
    max_pcs = pca_cfg.get("max_components")
    methods = list(cfg.get("methods", ["paired_difference", "difference_in_means", "ridge_probe"]))

    vectors: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    concepts = sorted({str(row["concept"]) for row in rows if str(row["concept"]) != str(neutral_concept)})
    for hp in hookpoints:
        all_acts = activations[hp]
        neutral_components = torch.empty((0, all_acts.shape[1]))
        if neutral_concept is not None:
            neutral_idx = [i for i, row in enumerate(rows) if str(row["concept"]) == str(neutral_concept)]
            if neutral_idx:
                neutral_components = principal_components_for_variance(
                    all_acts[neutral_idx], variance_fraction=neutral_fraction, max_components=max_pcs
                )

        for concept in concepts:
            idx = [i for i, row in enumerate(rows) if str(row["concept"]) == concept]
            concept_rows = [rows[i] for i in idx]
            concept_acts = all_acts[idx]
            pos = concept_acts[[int(row["label"]) == 1 for row in concept_rows]]
            neg = concept_acts[[int(row["label"]) == 0 for row in concept_rows]]
            if pos.shape[0] == 0 or neg.shape[0] == 0:
                continue

            for method in methods:
                probe = None
                method_name = str(method).strip().lower().replace("-", "_")
                if method_name == "difference_in_means":
                    vector = difference_in_means(pos, neg)
                elif method_name == "paired_difference":
                    pair_pos, pair_neg = _paired_matrices(concept_rows, concept_acts, cfg.get("pair_id_field", "pair_id"))
                    vector = paired_difference_in_means(pair_pos, pair_neg)
                elif method_name == "ridge_probe":
                    probe = fit_ridge_probe(pos, neg, l2=float(cfg.get("ridge_l2", 1.0)))
                    vector = probe.direction
                else:
                    raise ValueError(f"Unsupported concept-vector method: {method}")

                raw_vector = vector.clone()
                if neutral_components.shape[0] > 0:
                    vector = remove_subspace(vector, neutral_components)
                key = f"{_safe_key(concept)}__{method_name}__{_safe_key(hp)}"
                record: dict[str, Any] = {
                    "vector": vector.cpu(),
                    "raw_vector": raw_vector.cpu(),
                    "concept": concept,
                    "method": method_name,
                    "hookpoint": hp,
                    "n_positive": int(pos.shape[0]),
                    "n_negative": int(neg.shape[0]),
                    "neutral_pcs_removed": int(neutral_components.shape[0]),
                }
                if probe is not None:
                    record.update({
                        "probe_bias": probe.bias,
                        "probe_temperature": probe.score_std,
                        "probe_positive_mean": probe.score_mean_positive,
                        "probe_negative_mean": probe.score_mean_negative,
                    })
                vectors[key] = record
                diagnostics.append({
                    "vector_key": key,
                    "concept": concept,
                    "method": method_name,
                    "hookpoint": hp,
                    "n_positive": int(pos.shape[0]),
                    "n_negative": int(neg.shape[0]),
                    "vector_norm": float(vector.norm()),
                    "raw_clean_cosine": cosine_similarity(raw_vector, vector),
                    "neutral_pcs_removed": int(neutral_components.shape[0]),
                })

    output_path = Path(cfg.get("output_path", "outputs/concept_vectors/clarification_vectors.pt"))
    ensure_dir(output_path.parent)
    torch.save({
        "format": "clarifysae_dense_vector_v1",
        "model_name": config["model"]["name"],
        "dataset_path": cfg["dataset_path"],
        "vectors": vectors,
        "metadata": {"pooling": default_pooling, "mean_after_token": mean_after_token},
    }, output_path)
    pd.DataFrame(diagnostics).to_csv(output_path.with_suffix(".csv"), index=False)
    output_path.with_suffix(".config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved {len(vectors)} vectors to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(load_yaml(args.config))
