from __future__ import annotations

"""Extract dense clarification concept vectors from a labeled JSONL corpus.

The extractor supports matched differences, class centroids, and ridge probes;
neutral-PCA removal can be configured separately for prompt-side and
assistant-span representations.  Probe thresholds are always recalibrated after
subspace projection, avoiding a subtle but consequential gating mismatch.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from tqdm import tqdm

from clarifysae_llama.backends.hf_backend import HFCausalBackend
from clarifysae_llama.config import load_yaml
from clarifysae_llama.discovery.concept_vectors import (
    binary_score_diagnostics,
    calibrate_probe_direction,
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
            for required in ("concept", "text", "recommended_pooling"):
                if required not in obj:
                    raise ValueError(f"Line {line_no} is missing required field {required!r}")
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


def _assistant_token_start(tokenizer, text: str, valid_len: int) -> int:
    """Locate the first assistant-response token in the exact full tokenization."""

    char_start = _assistant_start_char(text)
    if char_start >= len(text):
        raise ValueError("assistant_mean pooling requested for text without an Assistant response")

    if getattr(tokenizer, "is_fast", False):
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=valid_len,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping", [])[:valid_len]
        for idx, pair in enumerate(offsets):
            start, end = int(pair[0]), int(pair[1])
            if end > char_start and end > start:
                return idx
        raise ValueError(
            "Assistant response was fully truncated; increase concept_discovery.max_length"
        )

    # Slow-tokenizer fallback: find the common prefix between exact full and
    # prefix tokenizations.  This is safer than blindly using prefix length,
    # because BPE boundaries can change at the response boundary.
    full_ids = tokenizer(
        text, add_special_tokens=True, truncation=True, max_length=valid_len
    )["input_ids"]
    prefix_ids = tokenizer(text[:char_start], add_special_tokens=True)["input_ids"]
    common = 0
    for left, right in zip(full_ids, prefix_ids):
        if left != right:
            break
        common += 1
    start = min(common, valid_len)
    if start >= valid_len:
        raise ValueError(
            "Assistant response was fully truncated or could not be located; "
            "use a fast tokenizer or increase max_length"
        )
    return start


def _pool_hidden(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer,
    texts: list[str],
    mode: str,
    mean_after_token: int,
) -> torch.Tensor:
    del input_ids  # retained in the signature for backwards-compatible callers
    pooled: list[torch.Tensor] = []
    mode = mode.strip().lower().replace("-", "_")
    for i in range(hidden.shape[0]):
        valid_len = int(attention_mask[i].sum().item())
        if valid_len <= 0:
            raise ValueError("Encountered a fully padded example")
        h = hidden[i, :valid_len]
        if mode == "last_nonpad":
            pooled.append(h[-1])
        elif mode == "mean_all":
            pooled.append(h.mean(dim=0))
        elif mode == "mean_after_token":
            start = min(max(int(mean_after_token), 0), max(valid_len - 1, 0))
            pooled.append(h[start:].mean(dim=0))
        elif mode == "assistant_mean":
            start = _assistant_token_start(tokenizer, texts[i], valid_len)
            pooled.append(h[start:].mean(dim=0))
        else:
            raise ValueError(f"Unsupported pooling mode: {mode}")
    return torch.stack(pooled, dim=0).float().cpu()


def _safe_key(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _paired_matrices(rows: list[dict[str, Any]], acts: torch.Tensor, pair_field: str):
    by_pair: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)
    for row, act in zip(rows, acts):
        if pair_field not in row:
            raise ValueError(f"Row {row.get('id', '<unknown>')} lacks pair field {pair_field!r}")
        label = int(row["label"])
        pair_id = str(row[pair_field])
        if label in by_pair[pair_id]:
            raise ValueError(f"Duplicate label {label} for pair {pair_id!r}")
        by_pair[pair_id][label] = act
    positives, negatives = [], []
    for pair in by_pair.values():
        if 1 in pair and 0 in pair:
            positives.append(pair[1])
            negatives.append(pair[0])
    if not positives:
        raise ValueError("No complete positive/negative pairs found")
    return torch.stack(positives), torch.stack(negatives)


def _normalise_string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


def _neutral_concepts_by_pooling(pca_cfg: dict[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for pooling, concepts in dict(pca_cfg.get("concepts_by_pooling", {})).items():
        result[str(pooling).replace("-", "_")] = _normalise_string_set(concepts)
    fallback = _normalise_string_set(pca_cfg.get("concepts"))
    fallback |= _normalise_string_set(pca_cfg.get("concept"))
    if fallback:
        result["*"] = fallback
    return result


def _row_split(row: dict[str, Any]) -> str:
    return str(row.get("split", "train"))


def _indices_for_splits(rows: list[dict[str, Any]], splits: set[str] | None) -> list[int]:
    if not splits:
        return list(range(len(rows)))
    return [i for i, row in enumerate(rows) if _row_split(row) in splits]


def _class_matrices(
    rows: list[dict[str, Any]],
    acts: torch.Tensor,
    *,
    allowed_indices: Iterable[int],
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    indices = list(allowed_indices)
    selected_rows = [rows[i] for i in indices]
    selected_acts = acts[indices]
    pos_mask = torch.tensor([int(row.get("label", 0)) == 1 for row in selected_rows], dtype=torch.bool)
    neg_mask = ~pos_mask
    return selected_rows, selected_acts[pos_mask], selected_acts[neg_mask]


def _concept_pooling(rows: list[dict[str, Any]]) -> str:
    modes = {str(row.get("recommended_pooling", "last_nonpad")).replace("-", "_") for row in rows}
    if len(modes) != 1:
        raise ValueError(f"A concept must use one pooling distribution, found {sorted(modes)}")
    return next(iter(modes))


def run(config: dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    cfg = config["concept_discovery"]
    all_rows = _read_jsonl(cfg["dataset_path"])

    pca_cfg = dict(cfg.get("neutral_pca", {}))
    neutral_by_pooling = _neutral_concepts_by_pooling(pca_cfg)
    neutral_concepts = set().union(*neutral_by_pooling.values()) if neutral_by_pooling else set()
    configured_concepts = _normalise_string_set(cfg.get("concepts"))
    if configured_concepts:
        keep = configured_concepts | neutral_concepts
        rows = [row for row in all_rows if str(row.get("concept")) in keep]
        target_concepts = sorted(configured_concepts)
    else:
        rows = all_rows
        target_concepts = sorted({str(row["concept"]) for row in rows} - neutral_concepts)
    if not rows:
        raise ValueError("No rows remain after concept/neutral filtering")

    missing_targets = [c for c in target_concepts if not any(str(r["concept"]) == c for r in rows)]
    if missing_targets:
        raise ValueError(f"Configured concepts absent from the corpus: {missing_targets}")

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
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            tokenized = {k: v.to(model_device) for k, v in tokenized.items()}
            with torch.inference_mode():
                _ = model(**tokenized, use_cache=False)
            outputs = capture.pop()
            for hp, hidden in outputs.items():
                if default_pooling == "recommended":
                    pooled = torch.empty((len(batch_rows), hidden.shape[-1]), dtype=torch.float32)
                    mode_groups: dict[str, list[int]] = defaultdict(list)
                    for i, row in enumerate(batch_rows):
                        mode_groups[str(row.get("recommended_pooling", "last_nonpad"))].append(i)
                    for mode, indices in mode_groups.items():
                        hidden_index = torch.tensor(indices, device=hidden.device, dtype=torch.long)
                        token_index = torch.tensor(
                            indices, device=tokenized["attention_mask"].device, dtype=torch.long
                        )
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
                        hidden,
                        tokenized["attention_mask"],
                        tokenized["input_ids"],
                        tokenizer,
                        texts,
                        default_pooling,
                        mean_after_token,
                    )
                activation_rows[hp].append(pooled)

    activations = {hp: torch.cat(parts, dim=0) for hp, parts in activation_rows.items()}
    neutral_fraction = float(pca_cfg.get("variance_fraction", 0.5))
    max_pcs = pca_cfg.get("max_components")
    require_neutral = bool(pca_cfg.get("required", False))
    methods = list(cfg.get("methods", ["paired_difference", "difference_in_means", "ridge_probe"]))
    fit_splits = _normalise_string_set(cfg.get("fit_splits")) or None
    eval_splits = _normalise_string_set(cfg.get("eval_splits")) or None
    pair_field = str(cfg.get("pair_id_field", "pair_id"))

    vectors: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    for hp in hookpoints:
        all_acts = activations[hp]
        average_residual_norm = float(all_acts.norm(dim=-1).mean())
        neutral_components_by_pooling: dict[str, torch.Tensor] = {}
        for pooling, concepts in neutral_by_pooling.items():
            neutral_idx = [
                i for i, row in enumerate(rows)
                if str(row["concept"]) in concepts
                and (not fit_splits or _row_split(row) in fit_splits)
            ]
            if neutral_idx:
                neutral_components_by_pooling[pooling] = principal_components_for_variance(
                    all_acts[neutral_idx],
                    variance_fraction=neutral_fraction,
                    max_components=max_pcs,
                )
            elif require_neutral:
                raise ValueError(
                    f"No neutral rows found for pooling={pooling!r}, concepts={sorted(concepts)}"
                )

        for concept in target_concepts:
            concept_idx = [i for i, row in enumerate(rows) if str(row["concept"]) == concept]
            concept_rows_all = [rows[i] for i in concept_idx]
            concept_acts_all = all_acts[concept_idx]
            pooling_mode = _concept_pooling(concept_rows_all)
            neutral_components = neutral_components_by_pooling.get(
                pooling_mode,
                neutral_components_by_pooling.get("*", torch.empty((0, all_acts.shape[1]))),
            )

            fit_local_idx = [
                i for i, row in enumerate(concept_rows_all)
                if not fit_splits or _row_split(row) in fit_splits
            ]
            fit_rows, pos, neg = _class_matrices(
                concept_rows_all, concept_acts_all, allowed_indices=fit_local_idx
            )
            if pos.shape[0] == 0 or neg.shape[0] == 0:
                raise ValueError(
                    f"Concept {concept!r} lacks both classes in fit_splits={sorted(fit_splits or [])}"
                )

            eval_local_idx = [
                i for i, row in enumerate(concept_rows_all)
                if eval_splits and _row_split(row) in eval_splits
            ]
            eval_rows: list[dict[str, Any]] = []
            eval_pos = eval_neg = None
            if eval_local_idx:
                eval_rows, eval_pos, eval_neg = _class_matrices(
                    concept_rows_all, concept_acts_all, allowed_indices=eval_local_idx
                )
                if eval_pos.shape[0] == 0 or eval_neg.shape[0] == 0:
                    eval_rows, eval_pos, eval_neg = [], None, None

            for method in methods:
                method_name = str(method).strip().lower().replace("-", "_")
                if method_name == "difference_in_means":
                    vector = difference_in_means(pos, neg)
                elif method_name == "paired_difference":
                    fit_acts = torch.cat([pos, neg], dim=0)  # placeholder overwritten below
                    # Preserve row/activation order for pair matching.
                    fit_global_local = fit_local_idx
                    fit_acts = concept_acts_all[fit_global_local]
                    vector = paired_difference_in_means(
                        *_paired_matrices(fit_rows, fit_acts, pair_field)
                    )
                elif method_name == "ridge_probe":
                    vector = fit_ridge_probe(
                        pos, neg, l2=float(cfg.get("ridge_l2", 1.0))
                    ).direction
                else:
                    raise ValueError(f"Unsupported concept-vector method: {method}")

                raw_vector = vector.clone()
                if neutral_components.shape[0] > 0:
                    vector = remove_subspace(vector, neutral_components)
                # Recalibrate every direction after PCA projection. For ridge
                # probes this replaces the invalid pre-projection bias/scale.
                calibration = calibrate_probe_direction(vector, pos, neg)
                vector = calibration.direction
                fit_diag = binary_score_diagnostics(vector, calibration.bias, pos, neg)
                eval_diag: dict[str, float] = {}
                if eval_pos is not None and eval_neg is not None:
                    eval_diag = binary_score_diagnostics(
                        vector, calibration.bias, eval_pos, eval_neg
                    )

                key = f"{_safe_key(concept)}__{method_name}__{_safe_key(hp)}"
                record: dict[str, Any] = {
                    "vector": vector.cpu(),
                    "raw_vector": raw_vector.cpu(),
                    "concept": concept,
                    "method": method_name,
                    "hookpoint": hp,
                    "pooling": pooling_mode,
                    "n_positive": int(pos.shape[0]),
                    "n_negative": int(neg.shape[0]),
                    "neutral_pcs_removed": int(neutral_components.shape[0]),
                    "average_residual_norm": average_residual_norm,
                    "score_bias": calibration.bias,
                    "score_temperature": calibration.score_std,
                    "score_positive_mean": calibration.score_mean_positive,
                    "score_negative_mean": calibration.score_mean_negative,
                    "fit_diagnostics": fit_diag,
                    "eval_diagnostics": eval_diag,
                }
                if method_name == "ridge_probe":
                    record.update({
                        "probe_bias": calibration.bias,
                        "probe_temperature": calibration.score_std,
                        "probe_positive_mean": calibration.score_mean_positive,
                        "probe_negative_mean": calibration.score_mean_negative,
                    })
                vectors[key] = record

                row_diag: dict[str, Any] = {
                    "vector_key": key,
                    "concept": concept,
                    "method": method_name,
                    "hookpoint": hp,
                    "pooling": pooling_mode,
                    "n_positive": int(pos.shape[0]),
                    "n_negative": int(neg.shape[0]),
                    "vector_norm": float(vector.norm()),
                    "raw_clean_cosine": cosine_similarity(raw_vector, vector),
                    "neutral_pcs_removed": int(neutral_components.shape[0]),
                    "average_residual_norm": average_residual_norm,
                }
                row_diag.update({f"fit_{k}": v for k, v in fit_diag.items()})
                row_diag.update({f"eval_{k}": v for k, v in eval_diag.items()})
                diagnostics.append(row_diag)

    output_path = Path(cfg.get("output_path", "outputs/concept_vectors/clarification_vectors.pt"))
    ensure_dir(output_path.parent)
    torch.save({
        "format": "clarifysae_dense_vector_v2",
        "model_name": config["model"]["name"],
        "dataset_path": cfg["dataset_path"],
        "vectors": vectors,
        "metadata": {
            "pooling": default_pooling,
            "mean_after_token": mean_after_token,
            "fit_splits": sorted(fit_splits or []),
            "eval_splits": sorted(eval_splits or []),
            "neutral_concepts_by_pooling": {
                key: sorted(value) for key, value in neutral_by_pooling.items()
            },
        },
    }, output_path)
    pd.DataFrame(diagnostics).to_csv(output_path.with_suffix(".csv"), index=False)
    output_path.with_suffix(".config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(vectors)} vectors to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(load_yaml(args.config))
