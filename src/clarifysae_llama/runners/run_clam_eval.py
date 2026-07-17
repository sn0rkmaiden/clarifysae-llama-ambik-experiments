from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from tqdm import tqdm

from clarifysae_llama.backends.hf_backend import HFCausalBackend
from clarifysae_llama.config import load_yaml
from clarifysae_llama.data.ambik_loader import load_ambik_selective_dataset
from clarifysae_llama.data.clam_prompting import (
    build_clam_classification_prompt,
    build_clam_question_prompt,
)
from clarifysae_llama.eval.clam_metrics import (
    add_clam_aggregate_metrics,
    clean_single_question,
    classification_summary,
    select_balanced_accuracy_threshold,
)
from clarifysae_llama.eval.metrics import aggregate_metrics, compute_example_metrics
from clarifysae_llama.eval.reporting import save_metric_tables
from clarifysae_llama.utils.io import ensure_dir, write_json, write_jsonl
from clarifysae_llama.utils.seed import set_seed


def _load_demonstrations(path: str | Path) -> list[dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError('CLAM demonstrations file must contain a JSON list.')
    return [dict(item) for item in payload]


def _validate_classification_demonstrations(
    demonstrations: list[dict[str, Any]],
) -> None:
    if not demonstrations:
        raise ValueError('Classification demonstration file is empty.')
    labels = {
        str(item.get('label', '')).strip().upper()
        for item in demonstrations
    }
    invalid = labels.difference({'AMBIGUOUS', 'CLEAR'})
    if invalid:
        raise ValueError(
            f'Unsupported classification demonstration labels: {sorted(invalid)}'
        )
    if not {'AMBIGUOUS', 'CLEAR'}.issubset(labels):
        raise ValueError(
            'Classification demonstrations must contain at least one '
            'AMBIGUOUS and one CLEAR example.'
        )


def _validate_question_demonstrations(
    demonstrations: list[dict[str, Any]],
) -> None:
    if not demonstrations:
        raise ValueError('Question-generation demonstration file is empty.')
    for index, item in enumerate(demonstrations):
        label = str(item.get('label', '')).strip().upper()
        if label != 'AMBIGUOUS':
            raise ValueError(
                'Question-generation demonstrations must all be labeled '
                f'AMBIGUOUS; item {index} has {label!r}.'
            )
        if not str(item.get('question', '')).strip():
            raise ValueError(
                f'Question-generation demonstration {index} has no question.'
            )


def _demonstration_source_ids(
    demonstrations: list[dict[str, Any]],
) -> set[str]:
    return {
        str(item['source_id'])
        for item in demonstrations
        if item.get('source_id') is not None
    }


def _load_classification_cache(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError('Cached CLAM results must contain a JSON object.')
    run_info = dict(payload.get('run_info') or {})
    examples = payload.get('examples')
    if not isinstance(examples, list):
        raise ValueError('Cached CLAM results are missing the examples list.')
    cache: dict[str, dict[str, Any]] = {}
    for item in examples:
        row = dict(item)
        item_id = str(row.get('id', ''))
        if not item_id:
            raise ValueError('Cached CLAM classification example has no id.')
        cache[item_id] = row
    return run_info, cache


def _reuse_classification_results(
    *,
    path: str | Path,
    rows: list[dict[str, Any]],
    model_name: str,
    candidates: list[str],
) -> tuple[float, dict[str, Any] | None, str]:
    run_info, cache = _load_classification_cache(path)
    cached_model = str(run_info.get('model_name', ''))
    if cached_model and cached_model != model_name:
        raise ValueError(
            f'Cached classification model is {cached_model!r}, expected {model_name!r}.'
        )
    cached_candidates = [str(item) for item in run_info.get('candidates', [])]
    if cached_candidates and cached_candidates != candidates:
        raise ValueError(
            f'Cached classification candidates are {cached_candidates}, expected {candidates}.'
        )

    missing_ids = sorted(str(row['id']) for row in rows if str(row['id']) not in cache)
    if missing_ids:
        raise ValueError(
            f'Cached classification results are missing {len(missing_ids)} ids; '
            f'first missing ids: {missing_ids[:5]}'
        )

    fields = (
        'ambiguity_score_ambiguous',
        'ambiguity_score_clear',
        'ambiguity_probability',
        'ambiguity_pair_probability',
        'predicted_ambiguous',
    )
    for row in rows:
        cached = cache[str(row['id'])]
        cached_prompt = str(cached.get('classification_prompt', ''))
        if cached_prompt != str(row['classification_prompt']):
            raise ValueError(
                'Cached classification prompt does not match the current '
                f'prompt for example {row["id"]}. Do not reuse Stage 1 after '
                'changing classification demonstrations or prompt wording.'
            )
        for field in fields:
            if field not in cached:
                raise ValueError(
                    f'Cached classification example {row["id"]} is missing {field}.'
                )
            row[field] = cached[field]

    threshold = float(run_info['decision_threshold'])
    calibration_summary = run_info.get('threshold_calibration')
    return threshold, calibration_summary, f'cache:{path}'


def _load_question_output_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    with open(path, 'r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item_id = str(item.get('id', ''))
            if not item_id:
                raise ValueError(
                    f'Missing id in cached question output at line {line_number}'
                )
            cache[item_id] = dict(item)
    return cache


def _softmax_pair(ambiguous_score: float, clear_score: float) -> float:
    values = torch.tensor([ambiguous_score, clear_score], dtype=torch.float64)
    return float(torch.softmax(values, dim=0)[0].item())


def _build_rows(
    dataset: pd.DataFrame,
    classification_demonstrations: list[dict[str, Any]],
    question_demonstrations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, item in dataset.iterrows():
        environment = str(item['environment_full'])
        task = str(item['task'])
        rows.append({
            **item.to_dict(),
            'classification_prompt': build_clam_classification_prompt(
                environment=environment,
                task=task,
                demonstrations=classification_demonstrations,
            ),
            'question_prompt': build_clam_question_prompt(
                environment=environment,
                task=task,
                demonstrations=question_demonstrations,
            ),
        })
    return rows


def _batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _score_classification_rows(
    *,
    backend: HFCausalBackend,
    rows: list[dict[str, Any]],
    candidates: list[str],
    batch_size: int,
    description: str,
) -> None:
    batches = list(_batched(rows, batch_size))
    for chunk in tqdm(batches, desc=description, unit='batch'):
        prompts = [row['classification_prompt'] for row in chunk]
        scores = backend.score_next_token_candidates_batch(prompts, candidates)
        for row, (ambiguous_score, clear_score) in zip(chunk, scores):
            true_probability = float(torch.exp(torch.tensor(ambiguous_score, dtype=torch.float64)).item())
            pair_probability = _softmax_pair(ambiguous_score, clear_score)
            row['ambiguity_score_ambiguous'] = float(ambiguous_score)
            row['ambiguity_score_clear'] = float(clear_score)
            # CLAM uses the next-token log probability of True as the
            # continuous ambiguity score. The True-vs-False softmax is saved
            # only as an additional diagnostic.
            row['ambiguity_probability'] = true_probability
            row['ambiguity_pair_probability'] = pair_probability


def _apply_threshold(rows: list[dict[str, Any]], threshold: float) -> None:
    for row in rows:
        row['predicted_ambiguous'] = bool(
            float(row['ambiguity_score_ambiguous']) >= threshold
        )


def _calibrate_threshold(
    *,
    backend: HFCausalBackend,
    calibration_config: dict[str, Any],
    classification_demonstrations: list[dict[str, Any]],
    candidates: list[str],
    batch_size: int,
    pred_dir: Path,
) -> tuple[float, dict[str, Any]]:
    objective = str(calibration_config.get('objective', 'balanced_accuracy'))
    if objective != 'balanced_accuracy':
        raise ValueError(
            f'Unsupported CLAM threshold objective: {objective!r}. '
            'Only balanced_accuracy is implemented.'
        )

    calibration_dataset = load_ambik_selective_dataset(
        calibration_config['path'],
        limit_pairs=calibration_config.get('limit'),
        include_unambiguous_pairs=True,
    )
    calibration_rows = _build_rows(
        calibration_dataset,
        classification_demonstrations,
        [],
    )
    eligible_rows = [
        row for row in calibration_rows
        if bool(row.get('classification_eligible', True))
    ]
    if not eligible_rows:
        raise ValueError('No classification-eligible threshold-calibration examples found.')

    _score_classification_rows(
        backend=backend,
        rows=eligible_rows,
        candidates=candidates,
        batch_size=batch_size,
        description='CLAM threshold calibration',
    )
    threshold, selected_summary = select_balanced_accuracy_threshold(
        [bool(row['gold_ambiguous']) for row in eligible_rows],
        [float(row['ambiguity_score_ambiguous']) for row in eligible_rows],
    )
    _apply_threshold(eligible_rows, threshold)
    selected_summary = classification_summary(eligible_rows) | {
        'threshold': threshold,
        'dataset_path': str(calibration_config['path']),
        'n_source_pairs_loaded': int(len(calibration_dataset) // 2),
        'n_excluded_identical_examples': int(len(calibration_rows) - len(eligible_rows)),
    }

    calibration_predictions = [
        {
            'id': row['id'],
            'source_id': row['source_id'],
            'variant': row['variant'],
            'source_ambiguity_type': row.get('source_ambiguity_type'),
            'gold_ambiguous': bool(row['gold_ambiguous']),
            'ambiguity_score_ambiguous': row['ambiguity_score_ambiguous'],
            'ambiguity_score_clear': row['ambiguity_score_clear'],
            'ambiguity_probability': row['ambiguity_probability'],
            'ambiguity_pair_probability': row['ambiguity_pair_probability'],
            'predicted_ambiguous': bool(row['predicted_ambiguous']),
            'classification_prompt': row['classification_prompt'],
        }
        for row in eligible_rows
    ]
    write_jsonl(pred_dir / 'threshold_calibration_predictions.jsonl', calibration_predictions)
    write_json(pred_dir / 'threshold_calibration_summary.json', selected_summary)
    return threshold, selected_summary


def run_clam_eval(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config.get('seed', 42)))
    experiment_name = str(config['experiment_name'])
    root_dir = Path(config['output']['root_dir'])
    run_dir = ensure_dir(root_dir / experiment_name)
    pred_dir = ensure_dir(run_dir / 'predictions')

    clam_cfg = config.get('clam', {})
    legacy_demonstrations_path = clam_cfg.get('demonstrations_path')
    classification_demonstrations_path = (
        clam_cfg.get('classification_demonstrations_path')
        or legacy_demonstrations_path
    )
    question_generation_mode = str(
        clam_cfg.get('question_generation_mode', 'few_shot')
    ).strip().lower()
    if question_generation_mode not in {'few_shot', 'zero_shot'}:
        raise ValueError(
            'clam.question_generation_mode must be either '
            f'"few_shot" or "zero_shot", got {question_generation_mode!r}.'
        )

    configured_question_demonstrations_path = clam_cfg.get(
        'question_demonstrations_path'
    )
    if question_generation_mode == 'few_shot':
        question_demonstrations_path = (
            configured_question_demonstrations_path
            or legacy_demonstrations_path
        )
    else:
        question_demonstrations_path = None

    if not classification_demonstrations_path:
        raise ValueError(
            'Set clam.classification_demonstrations_path. The legacy '
            'clam.demonstrations_path is also accepted for compatibility.'
        )

    classification_demonstrations = _load_demonstrations(
        classification_demonstrations_path
    )
    _validate_classification_demonstrations(classification_demonstrations)

    if question_generation_mode == 'few_shot':
        if not question_demonstrations_path:
            raise ValueError(
                'Set clam.question_demonstrations_path when '
                'clam.question_generation_mode is "few_shot".'
            )
        question_demonstrations = _load_demonstrations(
            question_demonstrations_path
        )
        _validate_question_demonstrations(question_demonstrations)
    else:
        # Zero-shot Stage 2 intentionally uses only QUESTION_HEADER plus the
        # current environment and instruction. This avoids demonstration
        # copying while leaving CLAM's few-shot Stage 1 unchanged.
        question_demonstrations = []

    if (
        question_demonstrations_path is not None
        and Path(classification_demonstrations_path)
        != Path(question_demonstrations_path)
    ):
        shared_source_ids = (
            _demonstration_source_ids(classification_demonstrations)
            & _demonstration_source_ids(question_demonstrations)
        )
        if shared_source_ids:
            raise ValueError(
                'Classification and question-generation demonstrations '
                'must use disjoint source examples. Shared source ids: '
                f'{sorted(shared_source_ids)}'
            )
    include_pairs = bool(config.get('dataset', {}).get('include_unambiguous_pairs', True))
    dataset = load_ambik_selective_dataset(
        config['dataset']['path'],
        limit_pairs=config['dataset'].get('limit'),
        include_unambiguous_pairs=include_pairs,
    )
    rows = _build_rows(
        dataset,
        classification_demonstrations,
        question_demonstrations,
    )

    config = dict(config)
    config['steering'] = {'enabled': False}
    backend = HFCausalBackend(config)

    batch_size = int(config.get('batching', {}).get('batch_size', 1))
    candidate_text = clam_cfg.get('candidate_text', {})
    ambiguous_candidate = str(candidate_text.get('ambiguous', 'True'))
    clear_candidate = str(candidate_text.get('clear', 'False'))
    candidates = [ambiguous_candidate, clear_candidate]
    tokenization = backend.candidate_tokenization(candidates)
    invalid_verbalizers = [item for item in tokenization if item['n_tokens'] != 1]
    if invalid_verbalizers:
        details = '; '.join(
            f"{item['candidate']!r} -> {item['tokens']}"
            for item in invalid_verbalizers
        )
        raise ValueError(
            'The configured CLAM True/False verbalizers are not single tokens '
            f'for {config["model"]["name"]}: {details}. Choose two single-token '
            'verbalizers and use them consistently in the prompt builder.'
        )

    classification_cache_path = clam_cfg.get('reuse_classification_from')
    calibration_cfg = dict(clam_cfg.get('threshold_calibration') or {})
    if classification_cache_path:
        threshold, calibration_summary, threshold_source = (
            _reuse_classification_results(
                path=classification_cache_path,
                rows=rows,
                model_name=str(config['model']['name']),
                candidates=candidates,
            )
        )
        classification_output_source = f'cache:{classification_cache_path}'
    elif calibration_cfg.get('path'):
        threshold, calibration_summary = _calibrate_threshold(
            backend=backend,
            calibration_config=calibration_cfg,
            classification_demonstrations=classification_demonstrations,
            candidates=candidates,
            batch_size=batch_size,
            pred_dir=pred_dir,
        )
        threshold_source = 'held_out_calibration'
        classification_output_source = 'generated'
    else:
        configured_threshold = clam_cfg.get('decision_threshold')
        if configured_threshold is None:
            raise ValueError(
                'Set clam.threshold_calibration.path for held-out threshold '
                'selection, provide clam.reuse_classification_from, or set an '
                'explicit clam.decision_threshold.'
            )
        threshold = float(configured_threshold)
        calibration_summary = None
        threshold_source = 'fixed_config'
        classification_output_source = 'generated'

    print(f'\n=== run_clam_eval :: {experiment_name} ===')
    print(f'pairs: {len(dataset) // 2 if include_pairs else len(dataset)} | examples: {len(dataset)}')
    print(f'candidates: {candidates}')
    print(
        'classification demonstrations: '
        f'{classification_demonstrations_path} '
        f'({len(classification_demonstrations)})'
    )
    if question_generation_mode == 'zero_shot':
        print('question generation: zero-shot (0 demonstrations)')
    else:
        print(
            'question demonstrations: '
            f'{question_demonstrations_path} ({len(question_demonstrations)})'
        )
    print(f'candidate tokenization: {tokenization}')
    print(f'decision threshold on log p(True): {threshold:.8f} ({threshold_source})')
    print(f'classification outputs: {classification_output_source}')

    if not classification_cache_path:
        _score_classification_rows(
            backend=backend,
            rows=rows,
            candidates=candidates,
            batch_size=batch_size,
            description='CLAM stage 1: classify',
        )
        _apply_threshold(rows, threshold)

    # Generate a question for every gold-ambiguous example to preserve the
    # oracle-gated diagnostic. For clear examples, generate only when the
    # predicted gate fires and the pair has non-contradictory labels.
    question_rows = [
        row for row in rows
        if bool(row['gold_ambiguous'])
        or (
            bool(row['predicted_ambiguous'])
            and bool(row.get('classification_eligible', True))
        )
    ]
    question_row_ids = {str(row['id']) for row in question_rows}
    cached_outputs_path = clam_cfg.get('reuse_question_outputs_from')
    if cached_outputs_path:
        question_cache = _load_question_output_cache(cached_outputs_path)
        missing_ids = sorted(question_row_ids.difference(question_cache))
        if missing_ids:
            raise ValueError(
                f'Cached CLAM outputs are missing {len(missing_ids)} required ids; '
                f'first missing ids: {missing_ids[:5]}'
            )
        question_chunks = [question_rows]
        question_output_source = f'cache:{cached_outputs_path}'
    else:
        question_cache = {}
        question_chunks = list(_batched(question_rows, batch_size))
        question_output_source = 'generated'

    for chunk in tqdm(question_chunks, desc='CLAM stage 2: ask', unit='batch'):
        if cached_outputs_path:
            raw_outputs = []
            for row in chunk:
                cached = question_cache[str(row['id'])]
                if bool(row['gold_ambiguous']):
                    raw_output = (
                        cached.get('raw_oracle_question_output')
                        or cached.get('raw_question_output')
                        or ''
                    )
                else:
                    raw_output = cached.get('raw_question_output') or ''
                if not str(raw_output).strip():
                    raise ValueError(
                        f'Cached CLAM output for {row["id"]} has no reusable raw question.'
                    )
                raw_outputs.append(str(raw_output))
        else:
            raw_outputs = backend.generate_batch([row['question_prompt'] for row in chunk])

        for row, raw_output in zip(chunk, raw_outputs):
            cleaned = clean_single_question(raw_output)
            row['raw_oracle_question_output'] = raw_output if bool(row['gold_ambiguous']) else ''
            row['oracle_generated_question'] = cleaned if bool(row['gold_ambiguous']) else ''
            use_end_to_end = bool(
                row['predicted_ambiguous']
                and (
                    bool(row['gold_ambiguous'])
                    or bool(row.get('classification_eligible', True))
                )
            )
            row['raw_question_output'] = raw_output if use_end_to_end else ''
            row['generated_question'] = cleaned if use_end_to_end else ''

    for row in rows:
        row.setdefault('raw_question_output', '')
        row.setdefault('generated_question', '')
        row.setdefault('raw_oracle_question_output', '')
        row.setdefault('oracle_generated_question', '')

    embed_threshold = float(config.get('evaluation', {}).get('embed_threshold', 0.75))
    nli_threshold = config.get('evaluation', {}).get('nli_threshold')
    enable_nli = bool(config.get('evaluation', {}).get('enable_nli', False))
    brevity_max = int(config.get('evaluation', {}).get('brevity_max', 1))

    prediction_rows: list[dict[str, Any]] = []
    for row in rows:
        questions = [row['generated_question']] if row['generated_question'] else []
        metrics = compute_example_metrics(
            ambiguity_type=str(row['ambiguity_type']),
            gold_question=str(row['gold_question']),
            model_questions=questions,
            predicted_ambiguous=bool(row['predicted_ambiguous']),
            embed_threshold=embed_threshold,
            nli_threshold=nli_threshold,
            enable_nli=enable_nli,
        )
        metrics['gold_ambiguous'] = bool(row['gold_ambiguous'])
        metrics['ambiguity_decision_correct'] = bool(
            row['predicted_ambiguous'] == row['gold_ambiguous']
        )

        oracle_questions = [row['oracle_generated_question']] if row['oracle_generated_question'] else []
        oracle_metrics = compute_example_metrics(
            ambiguity_type=str(row['ambiguity_type']),
            gold_question=str(row['gold_question']),
            model_questions=oracle_questions,
            predicted_ambiguous=True if bool(row['gold_ambiguous']) else False,
            embed_threshold=embed_threshold,
            nli_threshold=nli_threshold,
            enable_nli=enable_nli,
        )

        prediction_rows.append({
            'id': row['id'],
            'source_id': row['source_id'],
            'variant': row['variant'],
            'ambiguity_type': row['ambiguity_type'],
            'source_ambiguity_type': row.get('source_ambiguity_type'),
            'classification_eligible': bool(row.get('classification_eligible', True)),
            'pair_texts_identical': bool(row.get('pair_texts_identical', False)),
            'environment': row['environment_full'],
            'instruction': row['task'],
            'gold_question': row['gold_question'],
            'gold_ambiguous': bool(row['gold_ambiguous']),
            'classification_prompt': row['classification_prompt'],
            'question_prompt': row['question_prompt'] if str(row['id']) in question_row_ids else None,
            'ambiguity_score_ambiguous': row['ambiguity_score_ambiguous'],
            'ambiguity_score_clear': row['ambiguity_score_clear'],
            'ambiguity_probability': row['ambiguity_probability'],
            'ambiguity_pair_probability': row['ambiguity_pair_probability'],
            'predicted_ambiguous': bool(row['predicted_ambiguous']),
            'raw_question_output': row['raw_question_output'],
            'generated_question': row['generated_question'],
            'raw_oracle_question_output': row['raw_oracle_question_output'],
            'oracle_generated_question': row['oracle_generated_question'],
            'oracle_gate_question_similarity': oracle_metrics['model_question_best_similarity'],
            'oracle_gate_resolved_proxy_any': oracle_metrics['resolved_proxy_any'],
            **metrics,
        })

    predictions_path = pred_dir / 'predictions.jsonl'
    results_path = pred_dir / 'results.json'
    write_jsonl(predictions_path, prediction_rows)
    write_json(results_path, {
        'run_info': {
            'experiment_name': experiment_name,
            'model_name': config['model']['name'],
            'dataset_path': config['dataset']['path'],
            'include_unambiguous_pairs': include_pairs,
            'decision_threshold': threshold,
            'threshold_source': threshold_source,
            'threshold_calibration': calibration_summary,
            'candidates': candidates,
            'candidate_tokenization': tokenization,
            'classification_demonstrations_path': str(
                classification_demonstrations_path
            ),
            'question_generation_mode': question_generation_mode,
            'question_demonstrations_path': (
                str(question_demonstrations_path)
                if question_demonstrations_path is not None
                else None
            ),
            'n_classification_demonstrations': len(
                classification_demonstrations
            ),
            'n_question_demonstrations': len(question_demonstrations),
            'classification_output_source': classification_output_source,
            'question_output_source': question_output_source,
            'model_config': dict(config.get('model', {})),
            'generation_config': dict(config.get('generation', {})),
            'prompting_config': dict(config.get('prompting', {})),
        },
        'examples': prediction_rows,
    })

    example_metrics = pd.DataFrame(prediction_rows)
    aggregate_df, category_df = aggregate_metrics(
        example_metrics,
        embed_threshold=embed_threshold,
        brevity_max=brevity_max,
        nli_threshold=nli_threshold,
        enable_nli=enable_nli,
    )
    aggregate_df = add_clam_aggregate_metrics(aggregate_df, example_metrics)
    save_metric_tables(example_metrics, aggregate_df, category_df, run_dir)

    print(f'completed: {experiment_name}')
    print(f'  predictions: {predictions_path}')
    print(f'  aggregate metrics: {run_dir / "tables" / "aggregate_metrics.csv"}')
    return {
        'experiment_name': experiment_name,
        'predictions_path': str(predictions_path),
        'results_path': str(results_path),
        'aggregate_metrics_path': str(run_dir / 'tables' / 'aggregate_metrics.csv'),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_clam_eval(load_yaml(args.config))
