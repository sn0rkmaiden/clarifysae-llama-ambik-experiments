from __future__ import annotations

import argparse
import gc
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers.utils import logging as hf_logging

from clarifysae_llama.backends.hf_backend import HFCausalBackend
from clarifysae_llama.backends.steered_hf_backend import SteeredHFCausalBackend
from clarifysae_llama.config import load_yaml
from clarifysae_llama.data.ambik_loader import load_ambik_clarification_dataset
from clarifysae_llama.data.prompting import (
    build_clarification_prompt,
)
from clarifysae_llama.eval.metrics import aggregate_metrics, compute_example_metrics, normalize_questions
from clarifysae_llama.eval.reporting import save_metric_tables
from clarifysae_llama.utils.io import ensure_dir, write_json, write_jsonl
from clarifysae_llama.utils.logging import log_run
from clarifysae_llama.utils.parsing import assess_json_output, parse_model_json
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


def _evaluation_settings(config: dict[str, Any]) -> dict[str, Any]:
    eval_cfg = config.get('evaluation', {})
    return {
        'protocol': str(eval_cfg.get('protocol', 'combined_json')),
        'max_questions': int(eval_cfg.get('max_questions', 3)),
        'embed_threshold': float(eval_cfg.get('embed_threshold', 0.75)),
        'nli_threshold': eval_cfg.get('nli_threshold'),
        'enable_nli': bool(eval_cfg.get('enable_nli', False)),
        'brevity_max': int(eval_cfg.get('brevity_max', 1)),
    }


def build_backend(config: dict):
    backend_name = config['model'].get('backend', 'hf')
    steering_enabled = config.get('steering', {}).get('enabled', False)

    if backend_name != 'hf':
        raise ValueError(f'Only hf backend is supported in this repo, got: {backend_name}')
    if steering_enabled:
        return SteeredHFCausalBackend(config)
    return HFCausalBackend(config)


def build_prompts(dataset: pd.DataFrame, eval_settings: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_questions = int(eval_settings['max_questions'])

    for _, row in dataset.iterrows():
        description = str(row['environment_full'])
        task = str(row['ambiguous_task'])
        prompt_row = {
            'id': int(row['id']),
            'ambiguity_type': str(row['ambiguity_type']),
            'environment': description,
            'ambiguous_instruction': task,
            'gold_question': str(row.get('question', '') or ''),
            'gold_answer': str(row.get('answer', '') or ''),
            'gold_plan_for_clear': str(row.get('plan_for_clear_task', '') or ''),
            'prompt': build_clarification_prompt(
                description=description,
                task=task,
                max_questions=max_questions,
            ),
        }
        rows.append(prompt_row)
    return rows


def _print_run_header(config: dict[str, Any], n_examples: int, batch_size: int, eval_settings: dict[str, Any]) -> None:
    experiment_name = config['experiment_name']
    dataset_path = config['dataset']['path']
    n_batches = math.ceil(n_examples / batch_size) if n_examples else 0

    print(f"\n=== run_eval :: {experiment_name} ===")
    print(f"dataset: {dataset_path}")
    print(f"eval examples: {n_examples} | batch_size: {batch_size} | batches: {n_batches}")
    print(
        'evaluation: '
        f"protocol={eval_settings['protocol']} "
        f"max_questions={eval_settings['max_questions']} "
        f"embed_threshold={eval_settings['embed_threshold']} "
        f"brevity_max={eval_settings['brevity_max']} "
        f"enable_nli={eval_settings['enable_nli']}"
    )

    steering_cfg = config.get('steering', {})
    if steering_cfg.get('enabled', False):
        print(
            'steering: '
            f"hookpoint={steering_cfg.get('hookpoint')} "
            f"features={steering_cfg.get('feature_indices')} "
            f"strength={steering_cfg.get('strength')}"
        )
    else:
        print('steering: disabled')


def _run_generation_stage(
    *,
    backend,
    prompt_rows: list[dict[str, Any]],
    prompt_key: str,
    output_key: str,
    batch_size: int,
    console_cfg: dict[str, Any],
    experiment_name: str,
    stage_label: str,
) -> None:
    iterator = range(0, len(prompt_rows), batch_size)
    if console_cfg['show_progress']:
        iterator = tqdm(
            iterator,
            desc=f"{experiment_name} | {stage_label}",
            unit='batch',
            dynamic_ncols=True,
        )

    for start in iterator:
        chunk = prompt_rows[start:start + batch_size]
        prompts = [row[prompt_key] for row in chunk]
        predictions = backend.generate_batch(prompts)
        for row, raw_output in zip(chunk, predictions):
            row[output_key] = raw_output


def _coerce_predicted_ambiguous(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'true', 'false'}:
            return lowered == 'true'
    return None


def _compact_prediction_row(row: dict[str, Any], *, enable_nli: bool) -> dict[str, Any]:
    keep = [
        'id',
        'ambiguity_type',
        'ambiguous_instruction',
        'gold_question',
        'gold_ambiguous',
        'predicted_ambiguous',
        'ambiguity_decision_correct',
        'model_questions',
        'num_questions',
        'asked_question',
        'model_question_first_similarity',
        'model_question_best_similarity',
        'resolved_proxy_first',
        'resolved_proxy_any',
        'json_parsed_output',
        'json_exact_valid',
        'json_schema_valid',
        'json_recoverable_parse',
    ]
    if enable_nli:
        keep.extend([
            'model_question_first_nli_similarity',
            'model_question_best_nli_similarity',
            'resolved_nli_first',
            'resolved_nli_any',
        ])

    compact = {}
    for key in keep:
        if key not in row:
            continue
        value = row[key]
        if value is None:
            continue
        compact[key] = value
    return compact


def _compact_prediction_rows(rows: list[dict[str, Any]], *, enable_nli: bool) -> list[dict[str, Any]]:
    return [_compact_prediction_row(row, enable_nli=enable_nli) for row in rows]


def _select_example_metric_columns(raw_df: pd.DataFrame, *, enable_nli: bool) -> list[str]:
    columns = [
        'id',
        'ambiguity_type',
        'gold_ambiguous',
        'predicted_ambiguous',
        'ambiguity_decision_correct',
        'model_questions',
        'num_questions',
        'asked_question',
        'model_question_first_similarity',
        'model_question_best_similarity',
        'resolved_proxy_first',
        'resolved_proxy_any',
        'json_exact_valid',
        'json_schema_valid',
        'json_recoverable_parse',
    ]
    if enable_nli:
        columns.extend([
            'model_question_first_nli_similarity',
            'model_question_best_nli_similarity',
            'resolved_nli_first',
            'resolved_nli_any',
        ])
    return [column for column in columns if column in raw_df.columns]


def _finalize_prediction_rows(prompt_rows: list[dict[str, Any]], eval_settings: dict[str, Any]) -> list[dict[str, Any]]:
    prediction_rows: list[dict[str, Any]] = []

    for row in prompt_rows:
        prediction_row = {
            'id': row['id'],
            'ambiguity_type': row['ambiguity_type'],
            'environment': row['environment'],
            'ambiguous_instruction': row['ambiguous_instruction'],
            'gold_question': row['gold_question'],
            'gold_answer': row['gold_answer'],
            'gold_plan_for_clear': row['gold_plan_for_clear'],
        }

        raw_output = row.get('raw_model_output', '')
        parsed = parse_model_json(raw_output)
        parsed = parsed if isinstance(parsed, dict) else None

        predicted_ambiguous = _coerce_predicted_ambiguous(parsed.get('ambiguous')) if parsed else None
        model_questions = normalize_questions(parsed.get('question', parsed.get('questions', []))) if parsed else []
        json_metrics = assess_json_output(raw_output)

        prediction_row.update({
            'prompt': row['prompt'],
            'raw_model_output': raw_output,
            'parsed_output': parsed,
            'json_parsed_output': json_metrics['json_parsed_output'],
            'json_exact_valid': json_metrics['json_exact_valid'],
            'json_schema_valid': json_metrics['json_schema_valid'],
            'json_recoverable_parse': json_metrics['json_recoverable_parse'],
        })

        metrics = compute_example_metrics(
            ambiguity_type=row['ambiguity_type'],
            gold_question=row['gold_question'],
            model_questions=model_questions,
            predicted_ambiguous=predicted_ambiguous,
            embed_threshold=eval_settings['embed_threshold'],
            nli_threshold=eval_settings['nli_threshold'],
            enable_nli=eval_settings['enable_nli'],
        )
        prediction_row.update(metrics)
        prediction_rows.append(prediction_row)

    return prediction_rows


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


def run_eval(config: dict) -> dict[str, Any]:
    console_cfg = _configure_console(config)
    eval_settings = _evaluation_settings(config)
    set_seed(int(config.get('seed', 42)))

    experiment_name = config['experiment_name']
    root_dir = Path(config['output']['root_dir'])
    run_dir = ensure_dir(root_dir / experiment_name)
    pred_dir = ensure_dir(run_dir / 'predictions')
    ensure_dir(root_dir / 'logs')

    dataset = load_ambik_clarification_dataset(
        path=config['dataset']['path'],
        limit=config['dataset'].get('limit'),
    )
    prompt_rows = build_prompts(dataset, eval_settings)

    batch_size = int(config.get('batching', {}).get('batch_size', 1))
    _print_run_header(config, n_examples=len(prompt_rows), batch_size=batch_size, eval_settings=eval_settings)

    backend = None
    started_at = time.perf_counter()

    try:
        backend = build_backend(config)

        _run_generation_stage(
            backend=backend,
            prompt_rows=prompt_rows,
            prompt_key='prompt',
            output_key='raw_model_output',
            batch_size=batch_size,
            console_cfg=console_cfg,
            experiment_name=experiment_name,
            stage_label='generating',
        )

        prediction_rows = _finalize_prediction_rows(prompt_rows, eval_settings)

        compact_prediction_rows = _compact_prediction_rows(
            prediction_rows,
            enable_nli=eval_settings['enable_nli'],
        )

        predictions_path = pred_dir / 'predictions.jsonl'
        predictions_full_path = pred_dir / 'predictions_full.jsonl'
        results_path = pred_dir / 'results.json'
        results_full_path = pred_dir / 'results_full.json'
        write_jsonl(predictions_path, compact_prediction_rows)
        write_jsonl(predictions_full_path, prediction_rows)

        run_info = {
            'dataset_csv': config['dataset']['path'],
            'output_json': str(results_path),
            'output_full_json': str(results_full_path),
            'seed': int(config.get('seed', 42)),
            'num_examples': len(prompt_rows),
            'model_name': config['model']['name'],
            'steering_enabled': config.get('steering', {}).get('enabled', False),
            'steering_cfg': config.get('steering') if config.get('steering', {}).get('enabled', False) else None,
            'evaluation': eval_settings,
        }
        if 'run_metadata' in config:
            run_info['run_metadata'] = config['run_metadata']

        write_json(results_path, {'run_info': run_info, 'examples': compact_prediction_rows})
        write_json(results_full_path, {'run_info': run_info, 'examples': prediction_rows})

        raw_df = pd.DataFrame(prediction_rows)
        example_metrics = raw_df[_select_example_metric_columns(raw_df, enable_nli=eval_settings['enable_nli'])].copy()

        aggregate_df, category_df = aggregate_metrics(
            example_metrics,
            embed_threshold=eval_settings['embed_threshold'],
            brevity_max=eval_settings['brevity_max'],
            nli_threshold=eval_settings['nli_threshold'],
            enable_nli=eval_settings['enable_nli'],
        )
        save_metric_tables(example_metrics, aggregate_df, category_df, run_dir)

        example_metrics_path = run_dir / 'metrics' / 'example_metrics.csv'
        aggregate_metrics_path = run_dir / 'tables' / 'aggregate_metrics.csv'
        category_metrics_path = run_dir / 'tables' / 'category_metrics.csv'

        elapsed_sec = time.perf_counter() - started_at
        print(f"completed: {experiment_name} in {elapsed_sec:.1f}s")
        print(f"  predictions: {predictions_path}")
        print(f"  results json: {results_path}")
        print(f"  full results json: {results_full_path}")
        print(f"  example metrics: {example_metrics_path}")
        print(f"  aggregate metrics: {aggregate_metrics_path}")
        print(f"  category metrics: {category_metrics_path}")

        log_payload = {
            'experiment_name': experiment_name,
            'dataset_path': config['dataset']['path'],
            'n_examples': len(prompt_rows),
            'model_name': config['model']['name'],
            'steering_enabled': config.get('steering', {}).get('enabled', False),
            'sae_repo': config.get('steering', {}).get('sae_repo'),
            'hookpoint': config.get('steering', {}).get('hookpoint'),
            'feature_indices': config.get('steering', {}).get('feature_indices'),
            'strength': config.get('steering', {}).get('strength'),
            'predictions_path': str(predictions_path),
            'results_path': str(results_path),
            'results_full_path': str(results_full_path),
            'example_metrics_path': str(example_metrics_path),
            'aggregate_metrics_path': str(aggregate_metrics_path),
            'category_metrics_path': str(category_metrics_path),
            'evaluation': eval_settings,
            'elapsed_sec': elapsed_sec,
        }
        if 'run_metadata' in config:
            log_payload['run_metadata'] = config['run_metadata']

        log_run(root_dir / 'logs' / 'runs.jsonl', log_payload)

        return {
            'experiment_name': experiment_name,
            'predictions_path': str(predictions_path),
            'results_path': str(results_path),
            'results_full_path': str(results_full_path),
            'example_metrics_path': str(example_metrics_path),
            'aggregate_metrics_path': str(aggregate_metrics_path),
            'category_metrics_path': str(category_metrics_path),
            'run_metadata': config.get('run_metadata'),
            'elapsed_sec': elapsed_sec,
        }

    finally:
        _cleanup_backend(backend)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_eval(load_yaml(args.config))