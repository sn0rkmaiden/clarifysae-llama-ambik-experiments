from __future__ import annotations


def get_submodule_by_path(root_module, path: str):
    current = root_module
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def normalize_hookpoint_to_module_path(hookpoint: str) -> str:
    hp = hookpoint.strip()

    if hp == "embed_tokens":
        return "model.embed_tokens"
    if hp == "model.embed_tokens":
        return hp

    # Already fully qualified, e.g. "model.layers.10"
    if hp.startswith("model."):
        return hp

    # sparsify / EleutherAI style, e.g. "layers.23.mlp"
    if hp.startswith("layers."):
        return f"model.{hp}"

    raise ValueError(f"Unsupported hookpoint for Llama-style model: {hookpoint}")


def resolve_module_path(hookpoint: str, module_path: str | None = None) -> str:
    return module_path or normalize_hookpoint_to_module_path(hookpoint)


# Backward-compatibility for old imports.
def map_sae_hookpoint_to_hf_module_path(hookpoint: str) -> str:
    return normalize_hookpoint_to_module_path(hookpoint)