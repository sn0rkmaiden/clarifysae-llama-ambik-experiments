from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch

from clarifysae_llama.config import get_by_dotted_path, load_yaml
from clarifysae_llama.discovery.output_scores import compute_output_scores, save_output_score_results
from clarifysae_llama.discovery.sae_utils import get_num_latents
from clarifysae_llama.utils.io import ensure_dir
from clarifysae_llama.utils.logging import log_run
from clarifysae_llama.utils.seed import set_seed

try:
    from clarifysae_llama.runners.discover_features import (
        _get_model_input_device,
        _get_module_device,
        _load_model_and_tokenizer,
        _load_sae,
    )
except ImportError:
    from clarifysae_llama.runners.discover_features import (
        _get_model_input_device,
        _get_module_device,
        _load_model_and_tokenizer,
    )
    _load_sae = None

try:
    from clarifysae_llama.steering.hook_utils import get_submodule_by_path, resolve_module_path
except ImportError:
    from clarifysae_llama.steering.hook_utils import (
        get_submodule_by_path,
        map_sae_hookpoint_to_hf_module_path,
    )

    def resolve_module_path(hookpoint: str, module_path: str | None = None) -> str:
        return module_path or map_sae_hookpoint_to_hf_module_path(hookpoint)


_DTYPE_NAME_TO_TORCH: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _load_feature_scores(path: str | Path, tensor_key: str | None) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if torch.is_tensor(payload):
        return payload.float()
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported feature score payload type: {type(payload)!r}")
    if tensor_key is None:
        tensor_key = "scores" if "scores" in payload else None
    if tensor_key is None:
        raise KeyError(
            "Could not infer tensor key from feature score file. "
            "Set output_scoring.feature_score_key explicitly."
        )
    value = get_by_dotted_path(payload, tensor_key)
    if not torch.is_tensor(value):
        raise TypeError(f"Feature score entry {tensor_key!r} is not a tensor.")
    return value.float()


def _select_features(score_tensor: torch.Tensor, cfg: dict[str, Any]) -> list[int]:
    if "feature_indices" in cfg and cfg["feature_indices"] is not None:
        return [int(x) for x in cfg["feature_indices"]]
    top_k = int(cfg["top_k_features"])
    return score_tensor.topk(k=top_k).indices.tolist()


def _fallback_load_sae(output_cfg: dict[str, Any], device: torch.device, dtype: torch.dtype):
    from sparsify import Sae

    sae = Sae.load_from_hub(output_cfg["sae_repo"], hookpoint=output_cfg["hookpoint"])
    sae = sae.to(device=device, dtype=dtype)
    sae.eval()
    return sae


def _torch_dtype_to_name(dtype: torch.dtype) -> str:
    for name, candidate in _DTYPE_NAME_TO_TORCH.items():
        if candidate == dtype:
            return name
    return str(dtype).replace("torch.", "")


def _resolve_requested_torch_dtype(model_cfg: dict[str, Any]) -> torch.dtype:
    dtype_name = str(model_cfg.get("torch_dtype", "bfloat16")).strip().lower()
    if dtype_name not in _DTYPE_NAME_TO_TORCH:
        raise ValueError(
            f"Unsupported model.torch_dtype {dtype_name!r}. "
            f"Expected one of {sorted(_DTYPE_NAME_TO_TORCH)}."
        )
    return _DTYPE_NAME_TO_TORCH[dtype_name]


def _cuda_device_index(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return torch.cuda.current_device()


def _device_supports_bfloat16(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(_cuda_device_index(device))
    return major >= 8


def _resolve_runtime_model_dtype(model_cfg: dict[str, Any]) -> tuple[dict[str, Any], torch.dtype]:
    requested_dtype = _resolve_requested_torch_dtype(model_cfg)
    runtime_cfg = copy.deepcopy(model_cfg)

    if requested_dtype != torch.bfloat16:
        return runtime_cfg, requested_dtype

    if not torch.cuda.is_available():
        print(
            "[WARN] Requested model.torch_dtype=bfloat16, but CUDA is unavailable. "
            "Falling back to float32 for this run."
        )
        runtime_cfg["torch_dtype"] = "float32"
        return runtime_cfg, torch.float32

    device = torch.device("cuda", torch.cuda.current_device())
    if _device_supports_bfloat16(device):
        return runtime_cfg, requested_dtype

    major, minor = torch.cuda.get_device_capability(_cuda_device_index(device))
    print(
        "[WARN] Requested model.torch_dtype=bfloat16, but the active CUDA device only supports "
        f"sm_{major}{minor}. Falling back to float16 for this run."
    )
    runtime_cfg["torch_dtype"] = "float16"
    return runtime_cfg, torch.float16


def _resolve_sae_runtime(
    output_cfg: dict[str, Any],
    target_module_device: torch.device,
    model_dtype: torch.dtype,
) -> tuple[torch.device, torch.dtype]:
    loader_name = str(output_cfg.get("loader", "sparsify")).strip().lower().replace("-", "_")

    force_cpu = bool(output_cfg.get("force_cpu_sae", False))
    if force_cpu:
        requested_dtype_name = str(output_cfg.get("cpu_sae_dtype", "float32")).strip().lower()
        if requested_dtype_name not in _DTYPE_NAME_TO_TORCH:
            raise ValueError(
                f"Unsupported output_scoring.cpu_sae_dtype {requested_dtype_name!r}. "
                f"Expected one of {sorted(_DTYPE_NAME_TO_TORCH)}."
            )
        return torch.device("cpu"), _DTYPE_NAME_TO_TORCH[requested_dtype_name]

    sae_device = target_module_device
    sae_dtype = model_dtype

    if loader_name == "sparsify" and target_module_device.type == "cuda":
        if not _device_supports_bfloat16(target_module_device):
            major, minor = torch.cuda.get_device_capability(_cuda_device_index(target_module_device))
            print(
                "[WARN] Using a 'sparsify' SAE on CUDA sm_"
                f"{major}{minor} during output scoring can trigger illegal memory access errors. "
                "Running the SAE on CPU in float32 for this run."
            )
            sae_device = torch.device("cpu")
            sae_dtype = torch.float32

    return sae_device, sae_dtype


def _validate_feature_ids(feature_ids: list[int], sae: Any) -> None:
    num_latents = int(get_num_latents(sae))
    invalid_ids = [int(idx) for idx in feature_ids if idx < 0 or idx >= num_latents]
    if invalid_ids:
        preview = invalid_ids[:10]
        suffix = "" if len(invalid_ids) <= 10 else f" ... (+{len(invalid_ids) - 10} more)"
        raise ValueError(
            "Selected feature indices are out of bounds for the loaded SAE. "
            f"SAE num_latents={num_latents}, invalid feature ids={preview}{suffix}. "
            "Check that output_scoring.feature_scores_path, output_scoring.sae_repo, "
            "output_scoring.sae_file, and output_scoring.hookpoint all come from the same SAE run."
        )


def run_output_score_pipeline(config: dict[str, Any]) -> None:
    set_seed(int(config.get("seed", 42)))
    output_cfg = config["output_scoring"]
    experiment_name = config["experiment_name"]
    output_root = Path(output_cfg.get("root_dir", "outputs/discovery"))
    run_root = ensure_dir(output_root / experiment_name)
    output_dir = ensure_dir(run_root / "output_scores" / output_cfg.get("name", "default"))
    ensure_dir(output_root / "logs")

    feature_scores_path = Path(output_cfg["feature_scores_path"])
    score_tensor = _load_feature_scores(feature_scores_path, output_cfg.get("feature_score_key"))
    feature_ids = _select_features(score_tensor, output_cfg)

    runtime_model_cfg, dtype = _resolve_runtime_model_dtype(config["model"])
    runtime_config = copy.deepcopy(config)
    runtime_config["model"] = runtime_model_cfg

    model, tokenizer, dtype = _load_model_and_tokenizer(runtime_model_cfg)

    module_path = resolve_module_path(
        output_cfg["hookpoint"],
        output_cfg.get("module_path"),
    )
    target_module = get_submodule_by_path(model, module_path)
    target_module_device = _get_module_device(target_module)
    model_input_device = _get_model_input_device(model)
    sae_device, sae_dtype = _resolve_sae_runtime(
        output_cfg=output_cfg,
        target_module_device=target_module_device,
        model_dtype=dtype,
    )

    if _load_sae is not None:
        sae = _load_sae(
            discovery_cfg={
                "loader": output_cfg.get("loader", "sparsify"),
                "sae_repo": output_cfg["sae_repo"],
                "sae_file": output_cfg.get("sae_file"),
                "hookpoint": output_cfg["hookpoint"],
            },
            device=sae_device,
            dtype=sae_dtype,
        )
    else:
        sae = _fallback_load_sae(output_cfg=output_cfg, device=sae_device, dtype=sae_dtype)

    _validate_feature_ids(feature_ids, sae)

    results = compute_output_scores(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        target_module=target_module,
        feature_ids=feature_ids,
        prompt=output_cfg.get("prompt", "From my experience,"),
        amp_factor=float(output_cfg.get("amp_factor", output_cfg.get("steering_strength", 10.0))),
        top_k_tokens=int(output_cfg.get("top_k_tokens", output_cfg.get("logit_lens_top_k", 20))),
        dtype=sae_dtype,
        sae_device=sae_device,
        model_input_device=model_input_device,
    )

    save_output_score_results(
        output_dir=output_dir,
        feature_scores_path=feature_scores_path,
        results=results,
        config=runtime_config,
    )

    run_metadata = {
        "experiment_name": experiment_name,
        "feature_scores_path": str(feature_scores_path),
        "output_dir": str(output_dir),
        "n_features_used": len(feature_ids),
        "feature_ids": feature_ids,
        "sae_repo": output_cfg["sae_repo"],
        "hookpoint": output_cfg["hookpoint"],
        "module_path": output_cfg.get("module_path"),
        "loader": output_cfg.get("loader", "sparsify"),
        "sae_file": output_cfg.get("sae_file"),
        "prompt": output_cfg.get("prompt", "From my experience,"),
        "amp_factor": float(output_cfg.get("amp_factor", output_cfg.get("steering_strength", 10.0))),
        "top_k_tokens": int(output_cfg.get("top_k_tokens", output_cfg.get("logit_lens_top_k", 20))),
        "model_dtype_requested": str(config["model"].get("torch_dtype", "bfloat16")),
        "model_dtype_runtime": _torch_dtype_to_name(dtype),
        "sae_device": str(sae_device),
        "sae_dtype": _torch_dtype_to_name(sae_dtype),
    }
    (output_dir / "run_config.json").write_text(json.dumps(runtime_config, indent=2), encoding="utf-8")
    log_run(output_root / "logs" / "runs.jsonl", run_metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_output_score_pipeline(load_yaml(args.config))