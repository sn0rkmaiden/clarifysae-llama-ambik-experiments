from __future__ import annotations

import argparse
import gc
import hashlib
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers.utils import logging as hf_logging

from clarifysae_llama.backends.hf_backend import HFCausalBackend
from clarifysae_llama.backends.steered_hf_backend import SteeredHFCausalBackend
from clarifysae_llama.clarq_legacy.backend_adapter import BackendLLMAdapter
from clarifysae_llama.clarq_legacy.multi_info_provider_agent import helpers_m as MultiInfoProvider
from clarifysae_llama.clarq_legacy.provider_agent import helpers as GeneralProvider
from clarifysae_llama.clarq_legacy.seeker_agent import player as SeekerPlayer
from clarifysae_llama.clarq_legacy.utils import data_combination, read_path
from clarifysae_llama.config import load_yaml
from clarifysae_llama.eval.clarq_html_report import build_clarq_html_report
from clarifysae_llama.eval.clarq_metrics import (
    compute_metrics_for_payload,
    metrics_to_dataframes,
    parse_evaluation_set,
)
from clarifysae_llama.utils.io import ensure_dir, write_csv, write_json
from clarifysae_llama.utils.logging import log_run
from clarifysae_llama.utils.seed import set_seed


def _configure_console(config: dict[str, Any]) -> dict[str, Any]:
    console_cfg = config.get('console', {})
    suppress_tf_warnings = bool(console_cfg.get('suppress_transformers_warnings', True))
    show_progress = bool(console_cfg.get('show_progress', True))

    if suppress_tf_warnings:
        os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
        hf_logging.set_verbosity_error()
        warnings.filterwarnings('ignore', message=r'.*Both `max_new_tokens` .* and `max_length`.*')
        warnings.filterwarnings('ignore', message=r'.*The following generation flags are not valid and may be ignored:.*')
    else:
        hf_logging.set_verbosity_warning()

    return {
        'show_progress': show_progress,
        'suppress_transformers_warnings': suppress_tf_warnings,
    }


def _cleanup_backend(backend) -> None:
    if backend is None:
        return
    try:
        if hasattr(backend, 'steering') and getattr(backend, 'steering', None) is not None:
            try:
                backend.steering.detach()
            except Exception:
                pass
            try:
                if hasattr(backend.steering, 'sae'):
                    del backend.steering.sae
            except Exception:
                pass
            try:
                if hasattr(backend.steering, 'target_module'):
                    del backend.steering.target_module
            except Exception:
                pass
            try:
                del backend.steering
            except Exception:
                pass
        try:
            if hasattr(backend, 'model'):
                del backend.model
        except Exception:
            pass
        try:
            if hasattr(backend, 'tokenizer'):
                del backend.tokenizer
        except Exception:
            pass
        try:
            if hasattr(backend, 'generation_kwargs'):
                del backend.generation_kwargs
        except Exception:
            pass
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def _backend_cfgs_match(
    left_model_cfg: dict[str, Any],
    left_generation_cfg: dict[str, Any],
    left_prompting_cfg: dict[str, Any],
    right_model_cfg: dict[str, Any] | None,
    right_generation_cfg: dict[str, Any] | None,
    right_prompting_cfg: dict[str, Any] | None,
) -> bool:
    return (
        right_model_cfg is not None
        and right_generation_cfg is not None
        and dict(left_model_cfg) == dict(right_model_cfg)
        and dict(left_generation_cfg) == dict(right_generation_cfg)
        and dict(left_prompting_cfg or {}) == dict(right_prompting_cfg or {})
    )


def _build_unsteered_backend(
    model_cfg: dict[str, Any],
    generation_cfg: dict[str, Any],
    prompting_cfg: dict[str, Any],
) -> HFCausalBackend:
    return HFCausalBackend(
        {
            'model': model_cfg,
            'generation': generation_cfg,
            'prompting': prompting_cfg,
        }
    )


def _build_seeker_backend(config: dict[str, Any]):
    steering_enabled = bool(config.get('steering', {}).get('enabled', False))
    if steering_enabled:
        return SteeredHFCausalBackend(config)
    return HFCausalBackend(config)


def _conversation_meta(config: dict[str, Any], clarq_cfg: dict[str, Any]) -> dict[str, Any]:
    steering_cfg = config.get('steering') or {}
    steering_enabled = bool(steering_cfg.get('enabled', False))
    feature_indices = [int(x) for x in steering_cfg.get('feature_indices', [])] if steering_enabled else []
    run_metadata = config.get('run_metadata') or {}

    steering_meta = None
    if steering_enabled:
        steering_meta = {
            # Backwards-compatible scalar field used by older ClarQ tables.
            'feature': feature_indices[0] if len(feature_indices) == 1 else None,
            # New multi-feature fields.
            'feature_indices': feature_indices,
            'feature_count': len(feature_indices),
            'feature_set_label': run_metadata.get('feature_set_label'),
            'feature_weights': steering_cfg.get('feature_weights'),
            'normalize_each': steering_cfg.get('normalize_each'),
            'norm_cap': steering_cfg.get('norm_cap'),
            'strength': steering_cfg.get('strength'),
            'hookpoint': steering_cfg.get('hookpoint'),
            'module_path': steering_cfg.get('module_path'),
            'loader': steering_cfg.get('loader'),
            'sae_repo': steering_cfg.get('sae_repo'),
            'sae_file': steering_cfg.get('sae_file'),
            'sae_id': steering_cfg.get('sae_id'),
            'mode': steering_cfg.get('mode'),
            'max_act': (steering_cfg.get('runtime') or {}).get('max_act'),
        }

    return {
        'task_data_path': clarq_cfg['dataset_path'],
        'language': 'En' if 'English' in clarq_cfg['dataset_path'] else 'Ch',
        'mode': 'Chat' if clarq_cfg.get('player_chat_mode', False) else 'Comp',
        'evaluation_set_arg': clarq_cfg.get('evaluation_set', '0-25'),
        'evaluation_set': parse_evaluation_set(clarq_cfg.get('evaluation_set', '0-25')),
        'seeker_agent_llm': config['model']['name'],
        'provider_agent_llm': config['provider_model']['name'],
        'multi_info_provider_agent': bool(clarq_cfg.get('multi_info_provider_agent', False)),
        'steering': steering_meta,
        'judge_model': (config.get('judge_model') or {}).get('name'),
    }


def _slugify_filename_part(value: str) -> str:
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', value.strip())
    slug = re.sub(r'_+', '_', slug).strip('._-')
    return slug or 'run'


def _short_filename_stem(stem: str, max_len: int = 120) -> str:
    stem = _slugify_filename_part(stem)
    if len(stem) <= max_len:
        return stem
    digest = hashlib.md5(stem.encode('utf-8')).hexdigest()[:10]
    keep = max(1, max_len - len(digest) - 2)
    return f'{stem[:keep].rstrip("._-")}__{digest}'


def _build_artifact_basename(config: dict[str, Any], clarq_cfg: dict[str, Any]) -> str:
    experiment_slug = _slugify_filename_part(str(config.get('experiment_name', 'clarq_eval')))
    mode = 'chat' if clarq_cfg.get('player_chat_mode', False) else 'comp'
    eval_set = _slugify_filename_part(str(clarq_cfg.get('evaluation_set', '0-25')))
    steering_enabled = bool((config.get('steering') or {}).get('enabled', False))
    steering_tag = 'steered' if steering_enabled else 'baseline'
    return _short_filename_stem(f'{experiment_slug}__{mode}__evalset_{eval_set}__{steering_tag}')


def run_clarq_eval(config: dict[str, Any]) -> dict[str, Any]:
    console_cfg = _configure_console(config)
    set_seed(int(config.get('seed', 42)))

    clarq_cfg = config['clarq']
    eval_indices = parse_evaluation_set(str(clarq_cfg.get('evaluation_set', '0-25')))
    max_turns_cap = int(clarq_cfg.get('max_turns_cap', 22))
    show_progress = bool(console_cfg.get('show_progress', True))

    experiment_name = config['experiment_name']
    root_dir = Path(config['output']['root_dir'])
    run_dir = ensure_dir(root_dir / experiment_name)
    ensure_dir(root_dir / 'logs')
    artifact_basename = _build_artifact_basename(config, clarq_cfg)

    raw_data = read_path(clarq_cfg['dataset_path'])
    if not raw_data:
        raise ValueError(
            f"No ClarQ files were loaded from {clarq_cfg['dataset_path']!r}. "
            "Check that the directory exists and contains the ClarQ JSON files."
        )

    all_conv = data_combination(raw_data)
    if not all_conv:
        raise ValueError(
            f"ClarQ data combination produced no conversations from {clarq_cfg['dataset_path']!r}."
        )

    provider_cls = (
        MultiInfoProvider
        if clarq_cfg.get('multi_info_provider_agent', False)
        else GeneralProvider
    )

    total_dialogues = sum(len(one_type) for i, one_type in enumerate(all_conv) if i in eval_indices)

    print(f"\n=== run_clarq_eval :: {experiment_name} ===")
    print(f"dataset: {clarq_cfg['dataset_path']}")
    print(f"eval task types: {len(eval_indices)} / {len(all_conv)}")
    print(f"eval dialogues: {total_dialogues}")
    print(f"max_turns_cap: {max_turns_cap}")

    seeker_backend = provider_backend = judge_backend = None
    started_at = time.perf_counter()

    try:
        seeker_backend = _build_seeker_backend(config)
        provider_backend = _build_unsteered_backend(
            config['provider_model'],
            config['provider_generation'],
            config.get('provider_prompting', {}),
        )
        seeker_llm = BackendLLMAdapter(seeker_backend)
        provider_llm = BackendLLMAdapter(provider_backend)

        dialog_pbar = None
        if show_progress:
            dialog_pbar = tqdm(
                total=total_dialogues,
                desc=f'{experiment_name} | ClarQ dialogues',
                dynamic_ncols=True,
            )

        for i, one_type in enumerate(all_conv):
            if i not in eval_indices:
                continue

            for conv in one_type:
                gold_r = conv['all_response'].strip().split('\n')
                provider = provider_cls(
                    gold_r,
                    conv['background_splitted'],
                    conv['gold_structure'],
                    conv,
                    provider_llm,
                )
                seeker = SeekerPlayer(
                    conv['background_splitted'],
                    seeker_llm,
                    clarq_cfg.get('player_chat_mode', False),
                )
                l2l_conv: list[str] = []
                while True:
                    l2l_conv.append(provider.generate_response(l2l_conv))
                    l2l_conv.append(seeker.generate_response(l2l_conv))
                    if provider.is_conv_end(l2l_conv) or len(l2l_conv) > max_turns_cap:
                        break
                conv['l2l'][0] = l2l_conv

                if dialog_pbar is not None:
                    dialog_pbar.update(1)

        if dialog_pbar is not None:
            dialog_pbar.close()

        payload = {
            'meta': _conversation_meta(config, clarq_cfg),
            'data': all_conv,
        }
        results_path = run_dir / f'{artifact_basename}__clarq_results.json'
        write_json(results_path, payload)

        metrics_path = None
        summary_path = None
        report_path = None

        if config.get('judge_model'):
            unload_before_judge = bool(clarq_cfg.get('unload_models_before_judge', False))
            provider_can_judge = _backend_cfgs_match(
                config['provider_model'],
                config['provider_generation'],
                config.get('provider_prompting', {}),
                config.get('judge_model'),
                config.get('judge_generation'),
                config.get('judge_prompting', {}),
            )

            if unload_before_judge:
                # Dialogue generation is complete. Free the seeker before judging;
                # otherwise a one-GPU run tries to hold seeker + provider + judge.
                _cleanup_backend(seeker_backend)
                seeker_backend = None
                seeker_llm = None

            if provider_can_judge:
                # In the current ClarQ configs provider and judge are often the
                # same model. Reuse it instead of loading another 8B copy.
                judge_backend = provider_backend
            else:
                if unload_before_judge:
                    _cleanup_backend(provider_backend)
                    provider_backend = None
                    provider_llm = None
                judge_backend = _build_unsteered_backend(
                    config['judge_model'],
                    config['judge_generation'],
                    config.get('judge_prompting', {}),
                )

            judge_llm = BackendLLMAdapter(judge_backend)
            metrics = compute_metrics_for_payload(payload, judge_llm, eval_indices)
            metrics_df, summary_df = metrics_to_dataframes(metrics)
            metrics_path = run_dir / 'tables' / f'{artifact_basename}__clarq_metrics.csv'
            summary_path = run_dir / 'tables' / f'{artifact_basename}__clarq_summary.csv'
            write_csv(metrics_path, metrics_df)
            write_csv(summary_path, summary_df)

            if provider_can_judge:
                # Avoid double-cleaning the same backend object in the finally block.
                judge_backend = None

        if bool(clarq_cfg.get('write_html_report', True)):
            report_path = run_dir / 'report' / f'{artifact_basename}__clarq_report.html'
            build_clarq_html_report(
                payload=payload,
                output_path=report_path,
                metrics_path=metrics_path,
                summary_path=summary_path,
            )

        elapsed_sec = time.perf_counter() - started_at
        print(f"completed: {experiment_name} in {elapsed_sec:.1f}s")
        print(f"  results: {results_path}")
        if metrics_path:
            print(f"  metrics: {metrics_path}")
        if summary_path:
            print(f"  summary: {summary_path}")
        if report_path:
            print(f"  report: {report_path}")

        log_payload = {
            'experiment_name': experiment_name,
            'task_data_path': clarq_cfg['dataset_path'],
            'evaluation_set': eval_indices,
            'seeker_model': config['model']['name'],
            'provider_model': config['provider_model']['name'],
            'judge_model': (config.get('judge_model') or {}).get('name'),
            'steering_enabled': config.get('steering', {}).get('enabled', False),
            'results_path': str(results_path),
            'metrics_path': str(metrics_path) if metrics_path else None,
            'summary_path': str(summary_path) if summary_path else None,
            'report_path': str(report_path) if report_path else None,
            'elapsed_sec': elapsed_sec,
        }
        log_run(root_dir / 'logs' / 'runs.jsonl', log_payload)

        return {
            'experiment_name': experiment_name,
            'results_path': str(results_path),
            'metrics_path': str(metrics_path) if metrics_path else None,
            'summary_path': str(summary_path) if summary_path else None,
            'report_path': str(report_path) if report_path else None,
            'elapsed_sec': elapsed_sec,
        }
    finally:
        _cleanup_backend(judge_backend)
        _cleanup_backend(provider_backend)
        _cleanup_backend(seeker_backend)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    result = run_clarq_eval(load_yaml(args.config))
    print(result)