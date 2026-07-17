from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from huggingface_hub import snapshot_download
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae

from clarifysae_llama.config import load_yaml
from clarifysae_llama.discovery.dataset import load_token_chunks
from clarifysae_llama.discovery.scoring import SparseRollingStats
from clarifysae_llama.discovery.sae_utils import encode_sparse, get_num_latents
from clarifysae_llama.discovery.vocab import load_vocab_groups
from clarifysae_llama.utils.io import ensure_dir
from clarifysae_llama.utils.logging import log_run
from clarifysae_llama.utils.seed import set_seed

try:
    from clarifysae_llama.steering.hook_utils import get_submodule_by_path, resolve_module_path
except ImportError:
    from clarifysae_llama.steering.hook_utils import get_submodule_by_path, map_sae_hookpoint_to_hf_module_path

    def resolve_module_path(hookpoint: str, module_path: str | None = None) -> str:
        return module_path or map_sae_hookpoint_to_hf_module_path(hookpoint)


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def _check_free_disk_space(path: Path, min_free_gb: float = 2.0) -> None:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Not enough free disk space in {path}. "
            f"Free: {free_gb:.2f} GB, required at least: {min_free_gb:.2f} GB."
        )


def _warn_if_running_without_cuda(model) -> None:
    try:
        has_cuda_param = any(
            p.device.type == "cuda"
            for p in model.parameters()
            if p.device.type != "meta"
        )
    except Exception:
        has_cuda_param = False

    if not torch.cuda.is_available() or not has_cuda_param:
        print(
            "[WARN] Model does not appear to be running on CUDA. "
            "This run will likely be much slower. "
            "Check your NVIDIA driver / PyTorch CUDA compatibility."
        )


class HiddenActivationExtractor:
    def __init__(
        self,
        model,
        hookpoint: str | None = None,
        target_module=None,
        module_path: str | None = None,
    ):
        self.model = model
        if target_module is not None:
            self.target_module = target_module
        elif hookpoint is not None:
            resolved_module_path = resolve_module_path(hookpoint, module_path)
            self.target_module = get_submodule_by_path(self.model, resolved_module_path)
        else:
            raise ValueError("Provide either hookpoint or target_module.")
        self._captured = None
        self._handle = None

    def __enter__(self):
        self._handle = self.target_module.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._captured = None

    def _hook_fn(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self._captured = hidden

    def pop(self) -> torch.Tensor:
        if self._captured is None:
            raise RuntimeError("No hidden activations were captured on the last forward pass.")
        hidden = self._captured
        self._captured = None
        return hidden


def _get_module_device(module) -> torch.device:
    for param in module.parameters(recurse=True):
        if param.device.type != "meta":
            return param.device
    for buf in module.buffers(recurse=True):
        if buf.device.type != "meta":
            return buf.device
    return torch.device("cpu")


def _get_model_input_device(model) -> torch.device:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def _load_model_and_tokenizer(model_cfg: dict[str, Any]):
    model_name = model_cfg["name"]
    dtype = _resolve_torch_dtype(model_cfg.get("torch_dtype", "bfloat16"))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"dtype": dtype}
    if model_cfg.get("device_map", None) is not None:
        model_kwargs["device_map"] = model_cfg["device_map"]
    if model_cfg.get("attn_implementation", None) is not None:
        model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except TypeError:
        model_kwargs.pop("dtype", None)
        model_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    model.eval()
    return model, tokenizer, dtype


def _move_sae_to_device_dtype(sae: Any, device: torch.device, dtype: torch.dtype) -> Any:
    if hasattr(sae, "to"):
        try:
            sae = sae.to(device=device, dtype=dtype)
        except TypeError:
            sae = sae.to(device=device)
    if hasattr(sae, "eval"):
        sae.eval()
    return sae


def _load_sae(discovery_cfg: dict[str, Any], device: torch.device, dtype: torch.dtype):
    loader = discovery_cfg.get("loader", "sparsify")
    loader_name = str(loader).strip().lower().replace("-", "_")

    if loader_name == "sparsify":
        sae = Sae.load_from_hub(
            discovery_cfg["sae_repo"],
            hookpoint=discovery_cfg["hookpoint"],
        )
        return _move_sae_to_device_dtype(sae, device=device, dtype=dtype)

    if loader_name == "dictionary_learning":
        sae_file = discovery_cfg.get("sae_file")
        if not sae_file:
            raise ValueError(
                "discovery.sae_file is required when discovery.loader='dictionary_learning'."
            )

        try:
            from dictionary_learning import utils as dl_utils
        except ImportError as exc:
            raise ImportError(
                "dictionary_learning is required for discovery.loader='dictionary_learning'. "
                "Install it with: pip install dictionary-learning"
            ) from exc

        if sae_file.endswith(".pt"):
            trainer_subdir = str(Path(sae_file).parent)
        else:
            trainer_subdir = sae_file.rstrip("/")

        snapshot_root = snapshot_download(
            repo_id=discovery_cfg["sae_repo"],
            allow_patterns=[f"{trainer_subdir}/*"],
        )
        trainer_dir = Path(snapshot_root) / trainer_subdir

        sae, _cfg = dl_utils.load_dictionary(str(trainer_dir), device=str(device))
        return _move_sae_to_device_dtype(sae, device=device, dtype=dtype)

    raise ValueError(
        f"Unsupported discovery.loader '{loader}'. Expected 'sparsify' or 'dictionary_learning'."
    )


def _build_sparse_stats(config: dict[str, Any], tokenizer, num_features: int, device: torch.device):
    discovery_cfg = config["discovery"]
    scoring_cfg = discovery_cfg.get("scoring", {})
    expand_range = tuple(scoring_cfg.get("expand_range", [0, 0]))
    if len(expand_range) != 2:
        raise ValueError("discovery.scoring.expand_range must contain exactly two integers.")

    ignore_token_ids = []
    if scoring_cfg.get("ignore_special_tokens", True):
        ignore_token_ids.extend(tokenizer.all_special_ids)
    extra_ignore_ids = scoring_cfg.get("ignore_token_ids", [])
    ignore_token_ids.extend(int(token_id) for token_id in extra_ignore_ids)
    ignore_token_ids = sorted(set(ignore_token_ids))

    vocab_stats: dict[str, SparseRollingStats] = {}
    vocab_paths = discovery_cfg["vocab_paths"]
    for vocab_name, vocab_path in vocab_paths.items():
        token_groups = load_vocab_groups(vocab_path, tokenizer)
        vocab_stats[vocab_name] = SparseRollingStats(
            num_features=num_features,
            token_groups=token_groups,
            ignore_token_ids=ignore_token_ids,
            expand_range=(int(expand_range[0]), int(expand_range[1])),
            device=device,
            dtype=torch.float32,
        )
    return vocab_stats


def _save_result(result_dir: Path, vocab_name: str, result, top_k: int) -> None:
    vocab_dir = ensure_dir(result_dir / vocab_name)
    torch.save(
        {
            "scores": result.scores,
            "mean_pos": result.mean_pos,
            "mean_neg": result.mean_neg,
            "entropy": result.entropy,
            "single_means": result.single_means,
            "count_pos": result.count_pos,
            "count_neg": result.count_neg,
            "counts_per_group": result.counts_per_group,
        },
        vocab_dir / "feature_scores.pt",
    )

    scores_df = pd.DataFrame(
        {
            "feature_idx": torch.arange(result.scores.numel()).tolist(),
            "score": result.scores.tolist(),
            "mean_pos": result.mean_pos.tolist(),
            "mean_neg": result.mean_neg.tolist(),
            "entropy": result.entropy.tolist(),
        }
    ).sort_values("score", ascending=False)
    scores_df.to_csv(vocab_dir / "feature_scores.csv", index=False)
    scores_df.head(top_k).to_csv(vocab_dir / f"top_{top_k}_features.csv", index=False)

    metadata = {
        "vocab_name": vocab_name,
        "count_pos": result.count_pos,
        "count_neg": result.count_neg,
        "counts_per_group": result.counts_per_group,
        "top_features": scores_df.head(top_k)["feature_idx"].tolist(),
    }
    (vocab_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_discovery(config: dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    experiment_name = config["experiment_name"]
    discovery_cfg = config["discovery"]

    output_cfg = discovery_cfg.get("output", {})
    output_root = Path(output_cfg.get("root_dir", "outputs/discovery"))
    result_dir = ensure_dir(output_root / experiment_name)
    ensure_dir(output_root / "logs")

    min_free_gb = float(output_cfg.get("min_free_gb", 2.0))
    _check_free_disk_space(output_root, min_free_gb=min_free_gb)

    for vocab_name in discovery_cfg["vocab_paths"].keys():
        ensure_dir(result_dir / vocab_name)

    model, tokenizer, dtype = _load_model_and_tokenizer(config["model"])
    _warn_if_running_without_cuda(model)

    token_chunks = load_token_chunks(
        dataset_cfg=discovery_cfg["dataset"],
        tokenizer=tokenizer,
        tokenization_cfg=discovery_cfg.get("tokenization", {}),
    )

    extractor = HiddenActivationExtractor(
        model,
        hookpoint=discovery_cfg["hookpoint"],
        module_path=discovery_cfg.get("module_path"),
    )
    sae_device = _get_module_device(extractor.target_module)

    sae = _load_sae(
        discovery_cfg=discovery_cfg,
        device=sae_device,
        dtype=dtype,
    )

    vocab_stats = _build_sparse_stats(
        config,
        tokenizer,
        num_features=get_num_latents(sae),
        device=sae_device,
    )
    batch_size = int(discovery_cfg.get("batching", {}).get("token_batch_size", 8))
    model_input_device = _get_model_input_device(model)

    with extractor:
        for start in tqdm(range(0, len(token_chunks), batch_size), desc="Collecting feature statistics"):
            batch_tokens = token_chunks[start:start + batch_size]
            padded = pad_sequence(batch_tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_mask = (padded != tokenizer.pad_token_id).long()
            model_inputs = {
                "input_ids": padded.to(model_input_device),
                "attention_mask": attention_mask.to(model_input_device),
            }

            with torch.inference_mode():
                _ = model(**model_inputs, use_cache=False)
                hidden = extractor.pop()
                if hidden.ndim != 3:
                    raise ValueError(
                        f"Expected hidden states with shape [batch, seq, d_model], got {tuple(hidden.shape)}"
                    )

                hidden_2d = hidden.reshape(-1, hidden.shape[-1]).to(device=sae_device, dtype=dtype)
                sparse_latents = encode_sparse(sae, hidden_2d)

                top_acts = sparse_latents.top_acts.reshape(hidden.shape[0], hidden.shape[1], -1)
                top_indices = sparse_latents.top_indices.reshape(hidden.shape[0], hidden.shape[1], -1)

            tokens_for_masking = model_inputs["input_ids"].to(sae_device)
            for stats in vocab_stats.values():
                stats.update(tokens=tokens_for_masking, top_indices=top_indices, top_acts=top_acts)

    alpha = float(discovery_cfg.get("scoring", {}).get("alpha", 1.0))
    epsilon = float(discovery_cfg.get("scoring", {}).get("epsilon", 1e-12))
    top_k = int(output_cfg.get("top_k", 100))

    for vocab_name, stats in vocab_stats.items():
        result = stats.finalize(alpha=alpha, epsilon=epsilon)
        _save_result(result_dir=result_dir, vocab_name=vocab_name, result=result, top_k=top_k)

    run_metadata = {
        "experiment_name": experiment_name,
        "model_name": config["model"]["name"],
        "sae_repo": discovery_cfg["sae_repo"],
        "hookpoint": discovery_cfg["hookpoint"],
        "module_path": discovery_cfg.get("module_path"),
        "loader": discovery_cfg.get("loader", "sparsify"),
        "sae_file": discovery_cfg.get("sae_file"),
        "num_features": int(get_num_latents(sae)),
        "n_token_chunks": len(token_chunks),
        "token_batch_size": batch_size,
        "vocab_paths": discovery_cfg["vocab_paths"],
        "results_dir": str(result_dir),
    }
    (result_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log_run(output_root / "logs" / "runs.jsonl", run_metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_discovery(load_yaml(args.config))