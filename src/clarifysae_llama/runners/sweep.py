from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import re
import shutil
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from clarifysae_llama.config import dump_yaml, load_yaml, set_by_dotted_path
from clarifysae_llama.runners.run_eval import run_eval
from clarifysae_llama.runners.run_clarq_eval import run_clarq_eval
from clarifysae_llama.utils.io import ensure_dir, write_csv, write_jsonl


LEGACY_MANIFEST_COLUMNS = [
    'run_name',
    'parameter',
    'value',
    'config_path',
    'predictions_path',
    'results_path',
    'example_metrics_path',
    'aggregate_metrics_path',
    'category_metrics_path',
]

SINGLE_FEATURE_MANIFEST_COLUMNS = [
    'run_name',
    'vocab',
    'hookpoint',
    'module_path',
    'sae_file',
    'sae_repo',
    'sae_id',
    'feature_index',
    'strength',
    'config_path',
    'predictions_path',
    'results_path',
    'example_metrics_path',
    'aggregate_metrics_path',
    'category_metrics_path',
]

CLARQ_SINGLE_FEATURE_MANIFEST_COLUMNS = [
    'run_name',
    'vocab',
    'hookpoint',
    'module_path',
    'sae_file',
    'sae_repo',
    'sae_id',
    'feature_index',
    'strength',
    'config_path',
    'results_path',
    'metrics_path',
    'summary_path',
    'report_path',
]

CLARQ_MULTI_FEATURE_MANIFEST_COLUMNS = [
    'run_name',
    'vocab',
    'hookpoint',
    'module_path',
    'sae_file',
    'sae_repo',
    'sae_id',
    'feature_set_label',
    'feature_indices',
    'feature_count',
    'feature_weights',
    'normalize_each',
    'norm_cap',
    'strength',
    'config_path',
    'results_path',
    'metrics_path',
    'summary_path',
    'report_path',
]

CLARQ_COMPACT_SUMMARY_COLUMNS = [
    'run_name',
    'feature_index',
    'feature_set_label',
    'feature_indices',
    'feature_count',
    'strength',
    'success_rate',
    'step_recall',
    'ClarQ_count',
    'ClarQ_rate',
    'ClarQ_depth',
    'Goodbye_rate',
    'ARL',
    'AQD',
    'report_path',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to sweep YAML config')
    return parser.parse_args()


def _sanitize_token(value: Any) -> str:
    token = str(value).strip()
    token = token.replace(' ', '')
    token = token.replace('[', '')
    token = token.replace(']', '')
    token = token.replace(',', '-')
    token = token.replace('.', 'p')
    token = re.sub(r'[^A-Za-z0-9_\-]+', '_', token)
    return token.strip('_') or 'value'


def _artifact_file_stem(value: Any, max_len: int = 120) -> str:
    token = _sanitize_token(value)
    if len(token) <= max_len:
        return token
    digest = hashlib.md5(token.encode('utf-8')).hexdigest()[:10]
    keep = max(1, max_len - len(digest) - 2)
    return f'{token[:keep].rstrip("_")}__{digest}'


def _short_hookpoint(hookpoint: str) -> str:
    match = re.fullmatch(r'layers\.(\d+)\.(.+)', hookpoint)
    if match:
        layer_idx, suffix = match.groups()
        return f"l{layer_idx}_{suffix.replace('.', '_')}"
    return _sanitize_token(hookpoint)


def _build_legacy_run_name(experiment_prefix: str, parameter: str, value: Any) -> str:
    suffix = _sanitize_token(value)
    return f"{experiment_prefix}__{parameter.replace('.', '_')}__{suffix}"


def _build_single_feature_run_name(
    experiment_prefix: str,
    vocab: str | None,
    hookpoint: str,
    feature_index: int,
    strength: Any,
) -> str:
    parts = [experiment_prefix]
    if vocab:
        parts.append(_sanitize_token(vocab))
    parts.extend([
        _short_hookpoint(hookpoint),
        f'feat{feature_index}',
        f'str{_sanitize_token(strength)}',
    ])
    return '__'.join(parts)


def _features_token(feature_indices: list[int], *, max_items: int = 6) -> str:
    values = [str(int(x)) for x in feature_indices]
    if len(values) > max_items:
        digest = hashlib.md5(','.join(values).encode('utf-8')).hexdigest()[:8]
        shown = '-'.join(values[:max_items])
        return f'{shown}-plus{len(values) - max_items}-{digest}'
    return '-'.join(values)


def _feature_set_label(feature_set: dict[str, Any], set_index: int) -> str:
    for key in ('label', 'name', 'cluster_label'):
        if feature_set.get(key) not in (None, ''):
            return str(feature_set[key])
    if feature_set.get('cluster_id') not in (None, ''):
        return f"cluster{feature_set['cluster_id']}"
    if feature_set.get('cluster') not in (None, ''):
        return f"cluster{feature_set['cluster']}"
    return f'set{set_index}'


def _normalize_feature_set_entry(raw_entry: Any, set_index: int) -> dict[str, Any]:
    if isinstance(raw_entry, dict):
        entry = dict(raw_entry)
    elif isinstance(raw_entry, list):
        entry = {'features': raw_entry}
    else:
        raise ValueError(
            'Each feature set must be either a mapping with a features list or a bare list of feature ids. '
            f'Got {type(raw_entry)!r} at index {set_index}.'
        )

    if 'features' not in entry:
        raise ValueError(f'feature_sets[{set_index}] is missing features.')
    if not isinstance(entry['features'], list) or not entry['features']:
        raise ValueError(f'feature_sets[{set_index}].features must be a non-empty list.')
    entry['features'] = [int(x) for x in entry['features']]
    entry['label'] = _feature_set_label(entry, set_index)

    if entry.get('weights') is not None:
        if not isinstance(entry['weights'], list) or len(entry['weights']) != len(entry['features']):
            raise ValueError(
                f"feature_sets[{set_index}].weights must be a list with the same length as features "
                f"({len(entry.get('weights') or [])} != {len(entry['features'])})."
            )
        entry['weights'] = [float(x) for x in entry['weights']]
    return entry


def _feature_sets_from_group(group: dict[str, Any]) -> list[dict[str, Any]]:
    raw_sets = group.get('feature_sets')
    if raw_sets is None:
        raw_sets = group.get('clusters')
    if raw_sets is None:
        # Convenience: a group-level features list becomes one combined feature set.
        raw_sets = [{'label': group.get('label', 'combined'), 'features': group.get('features')}]
    if not isinstance(raw_sets, list) or not raw_sets:
        raise ValueError('multi_feature_strength groups require a non-empty feature_sets list.')
    return [_normalize_feature_set_entry(entry, idx) for idx, entry in enumerate(raw_sets)]


def _build_feature_set_run_name(
    experiment_prefix: str,
    vocab: str | None,
    hookpoint: str,
    feature_set_label: str,
    feature_indices: list[int],
    strength: Any,
) -> str:
    parts = [experiment_prefix]
    if vocab:
        parts.append(_sanitize_token(vocab))
    parts.extend([
        _short_hookpoint(hookpoint),
        _sanitize_token(feature_set_label),
        f'multi{len(feature_indices)}',
        f'feats{_sanitize_token(_features_token(feature_indices))}',
        f'str{_sanitize_token(strength)}',
    ])
    return '__'.join(parts)


def _storage_defaults() -> dict[str, Any]:
    return {
        'layout': 'flat',
        'keep_generated_configs': False,
        'keep_predictions': False,
        'keep_predictions_full': False,
        'keep_results': True,
        'keep_results_full': False,
        'keep_example_metrics': False,
        'keep_aggregate_metrics': False,
        'keep_category_metrics': False,
        'keep_clarq_metrics': True,
        'keep_clarq_summary': True,
        'keep_clarq_report': True,
        'keep_clarq_feature_dashboards': True,
        'cleanup_tmp_run_dirs': True,
    }


def _storage_cfg(sweep_cfg: dict[str, Any]) -> dict[str, Any]:
    storage = copy.deepcopy(_storage_defaults())
    storage.update(sweep_cfg.get('sweep', {}).get('storage', {}))
    return storage


def _prepare_sweep_dirs(
    sweep_cfg: dict[str, Any],
    base_cfg: dict[str, Any],
) -> tuple[str, Path, Path | None, Path, dict[str, Any]]:
    sweep_name = str(sweep_cfg.get('experiment_name') or f"{base_cfg['experiment_name']}__sweep")
    root_dir = Path(base_cfg['output']['root_dir'])
    sweep_dir = ensure_dir(root_dir / 'sweeps' / sweep_name)
    storage = _storage_cfg(sweep_cfg)

    generated_cfg_dir: Path | None
    if storage.get('keep_generated_configs', False):
        generated_cfg_dir = ensure_dir(sweep_dir / 'generated_configs')
    else:
        generated_cfg_dir = None

    tmp_root = ensure_dir(sweep_dir / '_tmp_run_eval')

    dump_yaml(sweep_dir / 'source_sweep_config.yaml', sweep_cfg)
    return sweep_name, sweep_dir, generated_cfg_dir, tmp_root, storage


def _validate_legacy_sweep_config(sweep_cfg: dict[str, Any]) -> None:
    sweep_section = sweep_cfg.get('sweep', {})
    if 'parameter' not in sweep_section or 'values' not in sweep_section:
        raise ValueError('Legacy sweep configs must define sweep.parameter and sweep.values.')
    if not isinstance(sweep_section['values'], list) or not sweep_section['values']:
        raise ValueError('sweep.values must be a non-empty list.')


def _validate_single_feature_sweep_config(sweep_cfg: dict[str, Any]) -> None:
    sweep_section = sweep_cfg.get('sweep', {})
    strengths = sweep_section.get('strengths')
    groups = sweep_section.get('groups')

    if not isinstance(strengths, list) or not strengths:
        raise ValueError('single_feature_strength sweep requires a non-empty sweep.strengths list.')
    if not isinstance(groups, list) or not groups:
        raise ValueError('single_feature_strength sweep requires a non-empty sweep.groups list.')

    for group_idx, group in enumerate(groups):
        if 'hookpoint' not in group:
            raise ValueError(f'sweep.groups[{group_idx}] is missing hookpoint.')
        if 'features' not in group:
            raise ValueError(f'sweep.groups[{group_idx}] is missing features.')
        if not isinstance(group['features'], list) or not group['features']:
            raise ValueError(f'sweep.groups[{group_idx}].features must be a non-empty list.')

        if 'module_path' in group and not isinstance(group['module_path'], str):
            raise ValueError(f'sweep.groups[{group_idx}].module_path must be a string if provided.')
        if 'sae_file' in group and not isinstance(group['sae_file'], str):
            raise ValueError(f'sweep.groups[{group_idx}].sae_file must be a string if provided.')
        if 'sae_repo' in group and not isinstance(group['sae_repo'], str):
            raise ValueError(f'sweep.groups[{group_idx}].sae_repo must be a string if provided.')
        if 'sae_id' in group and not isinstance(group['sae_id'], str):
            raise ValueError(f'sweep.groups[{group_idx}].sae_id must be a string if provided.')


def _validate_multi_feature_sweep_config(sweep_cfg: dict[str, Any]) -> None:
    sweep_section = sweep_cfg.get('sweep', {})
    strengths = sweep_section.get('strengths')
    groups = sweep_section.get('groups')

    if not isinstance(strengths, list) or not strengths:
        raise ValueError('multi_feature_strength sweep requires a non-empty sweep.strengths list.')
    if not isinstance(groups, list) or not groups:
        raise ValueError('multi_feature_strength sweep requires a non-empty sweep.groups list.')

    for group_idx, group in enumerate(groups):
        if 'hookpoint' not in group:
            raise ValueError(f'sweep.groups[{group_idx}] is missing hookpoint.')
        if 'module_path' in group and not isinstance(group['module_path'], str):
            raise ValueError(f'sweep.groups[{group_idx}].module_path must be a string if provided.')
        if 'sae_file' in group and not isinstance(group['sae_file'], str):
            raise ValueError(f'sweep.groups[{group_idx}].sae_file must be a string if provided.')
        if 'sae_repo' in group and not isinstance(group['sae_repo'], str):
            raise ValueError(f'sweep.groups[{group_idx}].sae_repo must be a string if provided.')
        if 'sae_id' in group and not isinstance(group['sae_id'], str):
            raise ValueError(f'sweep.groups[{group_idx}].sae_id must be a string if provided.')
        _feature_sets_from_group(group)


def _emit_run_start(
    run_idx: int,
    total_runs: int,
    *,
    run_name: str,
    vocab: str | None = None,
    hookpoint: str | None = None,
    feature_index: int | str | None = None,
    feature_indices: list[int] | None = None,
    feature_set_label: str | None = None,
    strength: Any | None = None,
    config_path: Path | None = None,
) -> None:
    print(f"\n[{run_idx}/{total_runs}] Running {run_name}")
    details = []
    if vocab is not None:
        details.append(f'vocab={vocab}')
    if hookpoint is not None:
        details.append(f'hookpoint={hookpoint}')
    if feature_index is not None:
        details.append(f'feature={feature_index}')
    if feature_set_label is not None:
        details.append(f'feature_set={feature_set_label}')
    if feature_indices is not None:
        details.append(f'features={feature_indices}')
    if strength is not None:
        details.append(f'strength={strength}')
    if details:
        print('  ' + ' '.join(details))
    if config_path is not None:
        print(f'  config: {config_path}')


def _replace_file(src: str | Path | None, dst: Path | None) -> str | None:
    if src is None or dst is None:
        return None
    src = Path(src)
    if not src.exists():
        return None
    ensure_dir(dst.parent)
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))
    return str(dst)


def _safe_rmtree(path: str | Path | None) -> None:
    if path is None:
        return
    path = Path(path)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _safe_unlink(path: str | Path | None) -> None:
    if path is None:
        return
    path = Path(path)
    if path.exists():
        path.unlink(missing_ok=True)


def _load_single_row_csv(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def _load_multi_row_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_clarq_payload(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload_path = Path(path)
    if not payload_path.exists():
        return {}
    return json.loads(payload_path.read_text())


def _strength_sort_key(value: Any) -> tuple[int, Any]:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _is_missing(value: Any) -> bool:
    return pd.isna(value) if not isinstance(value, (list, dict, tuple, set)) else False


def _display_value(value: Any, *, digits: int = 4) -> str:
    if value is None or _is_missing(value):
        return '—'
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f'{value:.{digits}f}'.rstrip('0').rstrip('.')
    return str(value)


def _dialogue_metric_lookup(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        try:
            key = (int(row.get('task_type_index', 0)), int(row.get('dialogue_index', 0)))
        except (TypeError, ValueError):
            continue
        lookup[key] = row
    return lookup


def _extract_clarq_dialogues(results_path: str | Path | None, run_metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = _load_clarq_payload(results_path)
    if not payload:
        return []
    lookup = _dialogue_metric_lookup(run_metric_rows)
    dialogues: list[dict[str, Any]] = []
    for task_type_index, one_type in enumerate(payload.get('data', [])):
        for dialogue_index, conv in enumerate(one_type):
            background_splitted = conv.get('background_splitted') or []
            task_name = background_splitted[1] if len(background_splitted) > 1 else ''
            transcript = ((conv.get('l2l') or [[]])[0]) or []
            dialogues.append({
                'task_type_index': task_type_index,
                'dialogue_index': dialogue_index,
                'task_name': task_name,
                'background': conv.get('background', ''),
                'gold_hints': conv.get('all_response_exaplain') or conv.get('all_response_explain') or [],
                'gold_structure': conv.get('gold_structure') or [],
                'transcript': transcript,
                'metrics': lookup.get((task_type_index, dialogue_index), {}),
            })
    return dialogues


def _feature_group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get('vocab'),
        row.get('hookpoint'),
        row.get('module_path'),
        row.get('sae_file'),
        row.get('sae_id'),
        row.get('feature_set_label') or row.get('feature_index'),
        str(row.get('feature_indices') or ''),
    )


def _feature_dashboard_filename(row: dict[str, Any]) -> str:
    parts = []
    if row.get('vocab') not in (None, ''):
        parts.append(_sanitize_token(row['vocab']))
    feature_label = row.get('feature_set_label') or row.get('feature_index', 'na')
    parts.extend([_short_hookpoint(str(row.get('hookpoint', 'hook'))), f"feat{_sanitize_token(feature_label)}"])
    digest = hashlib.md5(repr(_feature_group_key(row)).encode('utf-8')).hexdigest()[:8]
    parts.append(digest)
    return '__'.join(parts) + '.html'


def _transcript_html(transcript: list[str]) -> str:
    blocks = []
    for idx, utterance in enumerate(transcript):
        speaker = 'Jax' if idx % 2 == 0 else 'Seeker'
        role_class = 'jax' if idx % 2 == 0 else 'seeker'
        blocks.append(
            f"<div class='turn {role_class}'><div class='speaker'>{escape(speaker)}</div><div class='utterance'>{escape(str(utterance)).replace(chr(10), '<br>')}</div></div>"
        )
    return ''.join(blocks) or "<div class='empty'>No transcript available.</div>"


def _dialogue_summary_table(dialogues: list[dict[str, Any]]) -> str:
    rows = []
    for dialogue in dialogues:
        metrics = dialogue.get('metrics', {})
        rows.append(
            '<tr>'
            f"<td>{dialogue['task_type_index']}</td>"
            f"<td>{dialogue['dialogue_index']}</td>"
            f"<td>{escape(dialogue.get('task_name') or '—')}</td>"
            f"<td>{_display_value(metrics.get('success'))}</td>"
            f"<td>{_display_value(metrics.get('step_recall'))}</td>"
            f"<td>{_display_value(metrics.get('ClarQ_count'))}</td>"
            f"<td>{_display_value(metrics.get('Goodbye'))}</td>"
            '</tr>'
        )
    return ''.join(rows)


def _dialogues_html(dialogues: list[dict[str, Any]]) -> str:
    parts = []
    for dialogue in dialogues:
        metrics = dialogue.get('metrics', {})
        gold_hints = dialogue.get('gold_hints') or []
        gold_structure = dialogue.get('gold_structure') or []
        hints_html = ''.join(f'<li>{escape(str(item))}</li>' for item in gold_hints) or '<li>—</li>'
        structure_html = ', '.join(escape(str(item)) for item in gold_structure) or '—'
        summary_bits = [
            f"task_type={dialogue['task_type_index']}",
            f"dialogue={dialogue['dialogue_index']}",
            f"success={_display_value(metrics.get('success'))}",
            f"step_recall={_display_value(metrics.get('step_recall'))}",
            f"ClarQ_count={_display_value(metrics.get('ClarQ_count'))}",
            f"Goodbye={_display_value(metrics.get('Goodbye'))}",
        ]
        parts.append(
            '<details class="dialogue">'
            f"<summary>{escape(dialogue.get('task_name') or 'Dialogue')} <span class='meta'>{escape(' • '.join(summary_bits))}</span></summary>"
            "<div class='dialogue-body'>"
            f"<div class='dialogue-section'><div class='section-title'>Gold structure</div><div>{structure_html}</div></div>"
            f"<div class='dialogue-section'><div class='section-title'>Gold clarification targets</div><ul>{hints_html}</ul></div>"
            f"<div class='dialogue-section'><div class='section-title'>Transcript</div>{_transcript_html(dialogue.get('transcript') or [])}</div>"
            '</div>'
            '</details>'
        )
    return ''.join(parts) or "<div class='empty'>No dialogues available.</div>"


def _build_clarq_feature_dashboard_html(
    *,
    sweep_name: str,
    group_rows: list[dict[str, Any]],
    metrics_rows_by_run: dict[str, list[dict[str, Any]]],
) -> str:
    ordered_rows = sorted(group_rows, key=lambda row: _strength_sort_key(row.get('strength')))
    first_row = ordered_rows[0]

    overview_rows = []
    panels = []
    selector_options = ["<option value='all'>All strengths</option>"]

    for idx, row in enumerate(ordered_rows):
        strength = row.get('strength')
        strength_token = _sanitize_token(strength)
        run_name = row['run_name']
        dialogues = _extract_clarq_dialogues(row.get('results_path'), metrics_rows_by_run.get(run_name, []))

        selector_options.append(
            f"<option value='{escape(strength_token)}'>{escape(_display_value(strength))}</option>"
        )
        overview_rows.append(
            '<tr>'
            f"<td><button class='link-button' data-strength-target='{escape(strength_token)}'>{escape(_display_value(strength))}</button></td>"
            f"<td>{_display_value(row.get('success_rate'))}</td>"
            f"<td>{_display_value(row.get('step_recall'))}</td>"
            f"<td>{_display_value(row.get('ClarQ_count'))}</td>"
            f"<td>{_display_value(row.get('ClarQ_rate'))}</td>"
            f"<td>{_display_value(row.get('ClarQ_depth'))}</td>"
            f"<td>{_display_value(row.get('Goodbye_rate'))}</td>"
            f"<td>{_display_value(row.get('ARL'))}</td>"
            f"<td>{_display_value(row.get('AQD'))}</td>"
            '</tr>'
        )

        summary_cards = ''.join([
            f"<div class='card'><div class='label'>Strength</div><div class='value'>{escape(_display_value(strength))}</div></div>",
            f"<div class='card'><div class='label'>Success rate</div><div class='value'>{_display_value(row.get('success_rate'))}</div></div>",
            f"<div class='card'><div class='label'>Step recall</div><div class='value'>{_display_value(row.get('step_recall'))}</div></div>",
            f"<div class='card'><div class='label'>ClarQ count</div><div class='value'>{_display_value(row.get('ClarQ_count'))}</div></div>",
            f"<div class='card'><div class='label'>ClarQ rate</div><div class='value'>{_display_value(row.get('ClarQ_rate'))}</div></div>",
            f"<div class='card'><div class='label'>ClarQ depth</div><div class='value'>{_display_value(row.get('ClarQ_depth'))}</div></div>",
            f"<div class='card'><div class='label'>Goodbye rate</div><div class='value'>{_display_value(row.get('Goodbye_rate'))}</div></div>",
            f"<div class='card'><div class='label'>ARL</div><div class='value'>{_display_value(row.get('ARL'))}</div></div>",
            f"<div class='card'><div class='label'>AQD</div><div class='value'>{_display_value(row.get('AQD'))}</div></div>",
        ])
        panels.append(
            f"<section class='strength-panel{' visible' if idx == 0 else ''}' data-strength='{escape(strength_token)}'>"
            f"<h2>Strength {escape(_display_value(strength))}</h2>"
            f"<div class='cards'>{summary_cards}</div>"
            "<h3>Dialogue overview</h3>"
            "<table><thead><tr><th>Task type</th><th>Dialogue</th><th>Task</th><th>Success</th><th>Step recall</th><th>ClarQ count</th><th>Goodbye</th></tr></thead>"
            f"<tbody>{_dialogue_summary_table(dialogues)}</tbody></table>"
            "<h3>Dialogues</h3>"
            f"{_dialogues_html(dialogues)}"
            '</section>'
        )

    run_type = 'Steered' if first_row.get('strength') not in (None, '') else 'Baseline'
    if first_row.get('feature_set_label') not in (None, ''):
        feature_title = f"Feature set {escape(str(first_row.get('feature_set_label')))}"
    else:
        feature_title = f"Feature {escape(_display_value(first_row.get('feature_index')))}"
    subtitle_bits = [
        f"Sweep: {escape(sweep_name)}",
        f"Run type: {escape(run_type)}",
        f"Hookpoint: {escape(str(first_row.get('hookpoint') or '—'))}",
    ]
    if first_row.get('vocab') not in (None, ''):
        subtitle_bits.append(f"Vocab: {escape(str(first_row.get('vocab')))}")
    if first_row.get('feature_indices') not in (None, ''):
        subtitle_bits.append(f"Features: {escape(str(first_row.get('feature_indices')))}")
    if first_row.get('module_path') not in (None, ''):
        subtitle_bits.append(f"Module: {escape(str(first_row.get('module_path')))}")
    if first_row.get('sae_file') not in (None, ''):
        subtitle_bits.append(f"SAE file: {escape(str(first_row.get('sae_file')))}")
    if first_row.get('sae_id') not in (None, ''):
        subtitle_bits.append(f"SAE id: {escape(str(first_row.get('sae_id')))}")

    return f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>{feature_title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; background: #f8fafc; }}
    h1, h2, h3 {{ margin-bottom: 0.4rem; }}
    .subtitle {{ color: #475569; margin-bottom: 18px; }}
    .toolbar {{ display: flex; gap: 12px; align-items: center; margin: 18px 0 24px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .card {{ background: white; border-radius: 12px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .label {{ color: #64748b; font-size: 0.9rem; margin-bottom: 6px; }}
    .value {{ font-size: 1.2rem; font-weight: 600; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 18px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e2e8f0; text-align: left; vertical-align: top; }}
    th {{ background: #e2e8f0; }}
    .strength-panel {{ display: none; }}
    .strength-panel.visible {{ display: block; }}
    .dialogue {{ background: white; border-radius: 12px; padding: 0; margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    .dialogue > summary {{ cursor: pointer; padding: 14px; font-weight: 600; }}
    .dialogue-body {{ padding: 0 14px 14px; }}
    .meta {{ color: #64748b; font-weight: 400; font-size: 0.92rem; }}
    .dialogue-section {{ margin: 14px 0; }}
    .section-title {{ font-weight: 600; margin-bottom: 8px; }}
    .turn {{ border-radius: 10px; padding: 10px 12px; margin: 8px 0; }}
    .turn.jax {{ background: #dbeafe; }}
    .turn.seeker {{ background: #ede9fe; }}
    .speaker {{ font-weight: 700; margin-bottom: 6px; }}
    .link-button {{ background: none; border: none; color: #2563eb; cursor: pointer; padding: 0; font: inherit; text-decoration: underline; }}
    .empty {{ color: #64748b; }}
  </style>
</head>
<body>
  <h1>{feature_title}</h1>
  <div class='subtitle'>{escape(' • '.join(subtitle_bits))}</div>
  <div class='toolbar'>
    <label for='strength-select'><strong>Choose strength:</strong></label>
    <select id='strength-select'>
      {''.join(selector_options)}
    </select>
  </div>
  <h2>Strength comparison</h2>
  <table>
    <thead>
      <tr><th>Strength</th><th>Success rate</th><th>Step recall</th><th>ClarQ count</th><th>ClarQ rate</th><th>ClarQ depth</th><th>Goodbye rate</th><th>ARL</th><th>AQD</th></tr>
    </thead>
    <tbody>{''.join(overview_rows)}</tbody>
  </table>
  {''.join(panels)}
  <script>
    const select = document.getElementById('strength-select');
    const panels = Array.from(document.querySelectorAll('.strength-panel'));
    const buttons = Array.from(document.querySelectorAll('[data-strength-target]'));
    function setStrength(value) {{
      if (value === 'all') {{
        panels.forEach(panel => panel.classList.add('visible'));
        return;
      }}
      panels.forEach(panel => panel.classList.toggle('visible', panel.dataset.strength === value));
    }}
    select.addEventListener('change', (event) => setStrength(event.target.value));
    buttons.forEach(button => button.addEventListener('click', () => {{
      select.value = button.dataset.strengthTarget;
      setStrength(button.dataset.strengthTarget);
    }}));
    setStrength(panels[0] ? panels[0].dataset.strength : 'all');
    if (panels[0]) {{ select.value = panels[0].dataset.strength; }}
  </script>
</body>
</html>"""


def _write_clarq_feature_dashboards(
    *,
    sweep_dir: Path,
    sweep_name: str,
    summary_rows: list[dict[str, Any]],
    metrics_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    feature_reports_dir = ensure_dir(sweep_dir / 'feature_reports')
    metrics_rows_by_run: dict[str, list[dict[str, Any]]] = {}
    for row in metrics_rows:
        metrics_rows_by_run.setdefault(str(row.get('run_name')), []).append(row)

    grouped_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped_rows.setdefault(_feature_group_key(row), []).append(row)

    manifest: list[dict[str, Any]] = []
    for _, group_rows in sorted(grouped_rows.items(), key=lambda item: (str(item[0][1]), item[0][4])):
        dashboard_filename = _feature_dashboard_filename(group_rows[0])
        dashboard_path = feature_reports_dir / dashboard_filename
        dashboard_html = _build_clarq_feature_dashboard_html(
            sweep_name=sweep_name,
            group_rows=group_rows,
            metrics_rows_by_run=metrics_rows_by_run,
        )
        dashboard_path.write_text(dashboard_html, encoding='utf-8')
        ordered_rows = sorted(group_rows, key=lambda row: _strength_sort_key(row.get('strength')))
        manifest.append({
            'feature_index': group_rows[0].get('feature_index'),
            'feature_set_label': group_rows[0].get('feature_set_label'),
            'feature_indices': group_rows[0].get('feature_indices'),
            'vocab': group_rows[0].get('vocab'),
            'hookpoint': group_rows[0].get('hookpoint'),
            'module_path': group_rows[0].get('module_path'),
            'sae_file': group_rows[0].get('sae_file'),
            'sae_id': group_rows[0].get('sae_id'),
            'strengths': ', '.join(_display_value(row.get('strength')) for row in ordered_rows),
            'dashboard_path': str(dashboard_path),
        })

    if manifest:
        index_rows = []
        for row in manifest:
            rel_path = Path(row['dashboard_path']).relative_to(sweep_dir)
            index_rows.append(
                '<tr>'
                f"<td>{escape(_display_value(row.get('feature_set_label') or row.get('feature_index')))}</td>"
                f"<td>{escape(str(row.get('hookpoint') or '—'))}</td>"
                f"<td>{escape(str(row.get('vocab') or '—'))}</td>"
                f"<td>{escape(str(row.get('strengths') or '—'))}</td>"
                f"<td><a href='{escape(rel_path.as_posix())}'>{escape(Path(row['dashboard_path']).name)}</a></td>"
                '</tr>'
            )
        index_html = f"""<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>{escape(sweep_name)} feature dashboards</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; background: #f8fafc; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e2e8f0; text-align: left; }}
    th {{ background: #e2e8f0; }}
  </style>
</head>
<body>
  <h1>{escape(sweep_name)} feature dashboards</h1>
  <p>One HTML per steering feature or feature set. Each dashboard lets you switch between strengths on a single page.</p>
  <table>
    <thead><tr><th>Feature / set</th><th>Hookpoint</th><th>Vocab</th><th>Strengths</th><th>Dashboard</th></tr></thead>
    <tbody>{''.join(index_rows)}</tbody>
  </table>
</body>
</html>"""
        (sweep_dir / 'clarq_feature_dashboards.html').write_text(index_html, encoding='utf-8')
        write_csv(sweep_dir / 'feature_dashboards.csv', pd.DataFrame(manifest))

    return manifest


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _flatten_run_artifacts(
    *,
    sweep_dir: Path,
    run_name: str,
    result: dict[str, Any],
    storage: dict[str, Any],
) -> dict[str, Any]:
    if storage.get('layout', 'flat') != 'flat':
        return {
            'predictions_path': result.get('predictions_path'),
            'results_path': result.get('results_path'),
            'example_metrics_path': result.get('example_metrics_path'),
            'aggregate_metrics_path': result.get('aggregate_metrics_path'),
            'category_metrics_path': result.get('category_metrics_path'),
        }

    predictions_dir = ensure_dir(sweep_dir / 'predictions')
    results_dir = ensure_dir(sweep_dir / 'results')
    metrics_dir = ensure_dir(sweep_dir / 'metrics')

    kept_paths = {
        'predictions_path': None,
        'results_path': None,
        'example_metrics_path': None,
        'aggregate_metrics_path': None,
        'category_metrics_path': None,
    }

    predictions_path = result.get('predictions_path')
    results_path = result.get('results_path')
    example_metrics_path = result.get('example_metrics_path')
    aggregate_metrics_path = result.get('aggregate_metrics_path')
    category_metrics_path = result.get('category_metrics_path')
    results_full_path = result.get('results_full_path')

    predictions_full_path = None
    if predictions_path is not None:
        pred_dir = Path(predictions_path).parent
        candidate = pred_dir / 'predictions_full.jsonl'
        if candidate.exists():
            predictions_full_path = candidate

    if storage.get('keep_predictions', False):
        kept_paths['predictions_path'] = _replace_file(
            predictions_path,
            predictions_dir / f'{_artifact_file_stem(run_name)}__predictions.jsonl',
        )
    else:
        _safe_unlink(predictions_path)

    if storage.get('keep_predictions_full', False):
        _replace_file(
            predictions_full_path,
            predictions_dir / f'{run_name}__predictions_full.jsonl',
        )
    else:
        _safe_unlink(predictions_full_path)

    if storage.get('keep_results', True):
        kept_paths['results_path'] = _replace_file(
            results_path,
            results_dir / f'{run_name}__results.json',
        )
    else:
        _safe_unlink(results_path)

    if storage.get('keep_results_full', False):
        _replace_file(
            results_full_path,
            results_dir / f'{run_name}__results_full.json',
        )
    else:
        _safe_unlink(results_full_path)

    if storage.get('keep_example_metrics', False):
        kept_paths['example_metrics_path'] = _replace_file(
            example_metrics_path,
            metrics_dir / f'{run_name}__example_metrics.csv',
        )
    else:
        _safe_unlink(example_metrics_path)

    if storage.get('keep_aggregate_metrics', False):
        kept_paths['aggregate_metrics_path'] = _replace_file(
            aggregate_metrics_path,
            metrics_dir / f'{_artifact_file_stem(run_name)}__aggregate_metrics.csv',
        )
    else:
        _safe_unlink(aggregate_metrics_path)

    if storage.get('keep_category_metrics', False):
        kept_paths['category_metrics_path'] = _replace_file(
            category_metrics_path,
            metrics_dir / f'{_artifact_file_stem(run_name)}__category_metrics.csv',
        )
    else:
        _safe_unlink(category_metrics_path)

    if storage.get('cleanup_tmp_run_dirs', True):
        if predictions_path is not None:
            _safe_rmtree(Path(predictions_path).parent)
        if example_metrics_path is not None:
            _safe_rmtree(Path(example_metrics_path).parents[1])

    return kept_paths


def _flatten_clarq_run_artifacts(
    *,
    sweep_dir: Path,
    run_name: str,
    result: dict[str, Any],
    storage: dict[str, Any],
) -> dict[str, Any]:
    if storage.get('layout', 'flat') != 'flat':
        return {
            'results_path': result.get('results_path'),
            'metrics_path': result.get('metrics_path'),
            'summary_path': result.get('summary_path'),
            'report_path': result.get('report_path'),
        }

    results_dir = ensure_dir(sweep_dir / 'results')
    metrics_dir = ensure_dir(sweep_dir / 'metrics')
    summaries_dir = ensure_dir(sweep_dir / 'summaries')
    reports_dir = ensure_dir(sweep_dir / 'reports')

    kept_paths = {
        'results_path': None,
        'metrics_path': None,
        'summary_path': None,
        'report_path': None,
    }

    results_path = result.get('results_path')
    metrics_path = result.get('metrics_path')
    summary_path = result.get('summary_path')
    report_path = result.get('report_path')

    if storage.get('keep_results', True):
        kept_paths['results_path'] = _replace_file(
            results_path,
            results_dir / f'{_artifact_file_stem(run_name)}__clarq_results.json',
        )
    else:
        _safe_unlink(results_path)

    if storage.get('keep_clarq_metrics', True):
        kept_paths['metrics_path'] = _replace_file(
            metrics_path,
            metrics_dir / f'{_artifact_file_stem(run_name)}__clarq_metrics.csv',
        )
    else:
        _safe_unlink(metrics_path)

    if storage.get('keep_clarq_summary', True):
        kept_paths['summary_path'] = _replace_file(
            summary_path,
            summaries_dir / f'{_artifact_file_stem(run_name)}__clarq_summary.csv',
        )
    else:
        _safe_unlink(summary_path)

    if storage.get('keep_clarq_report', True):
        kept_paths['report_path'] = _replace_file(
            report_path,
            reports_dir / f'{_artifact_file_stem(run_name)}__clarq_report.html',
        )
    else:
        _safe_unlink(report_path)

    if storage.get('cleanup_tmp_run_dirs', True) and results_path is not None:
        _safe_rmtree(Path(results_path).parent)

    return kept_paths


def _merge_metadata(row: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    merged = dict(row)
    merged.update(metrics)
    return merged


def _run_legacy_sweep(sweep_cfg: dict[str, Any], base_cfg: dict[str, Any]) -> None:
    _validate_legacy_sweep_config(sweep_cfg)
    sweep_name, sweep_dir, generated_cfg_dir, tmp_root, storage = _prepare_sweep_dirs(sweep_cfg, base_cfg)
    parameter = sweep_cfg['sweep']['parameter']
    values = sweep_cfg['sweep']['values']

    print(f"=== sweep :: {sweep_name} ===")
    print(f'legacy parameter sweep over {parameter}')
    print(f'total runs: {len(values)}')
    print(f"storage layout: {storage.get('layout', 'flat')}")

    manifest_rows: list[dict[str, Any]] = []
    aggregate_summary_rows: list[dict[str, Any]] = []
    category_summary_rows: list[dict[str, Any]] = []

    for run_idx, value in enumerate(values, start=1):
        run_cfg = copy.deepcopy(base_cfg)
        run_cfg['output']['root_dir'] = str(tmp_root)
        set_by_dotted_path(run_cfg, parameter, value)
        run_name = _build_legacy_run_name(sweep_name, parameter, value)
        run_cfg['experiment_name'] = run_name

        cfg_path = None
        if generated_cfg_dir is not None:
            cfg_path = generated_cfg_dir / f'{_artifact_file_stem(run_name)}.yaml'
            dump_yaml(cfg_path, run_cfg)

        _emit_run_start(run_idx, len(values), run_name=run_name, config_path=cfg_path)
        try:
            result = run_eval(run_cfg)
            print(f"  finished: {run_name}")
        finally:
            _release_cuda_memory()

        aggregate_metrics = _load_single_row_csv(result.get('aggregate_metrics_path'))
        category_metrics = _load_multi_row_csv(result.get('category_metrics_path'))

        flattened_paths = _flatten_run_artifacts(
            sweep_dir=sweep_dir,
            run_name=run_name,
            result=result,
            storage=storage,
        )

        manifest_row = {
            'run_name': run_name,
            'parameter': parameter,
            'value': value,
            'config_path': str(cfg_path) if cfg_path is not None else None,
            'predictions_path': flattened_paths['predictions_path'],
            'results_path': flattened_paths['results_path'],
            'example_metrics_path': flattened_paths['example_metrics_path'],
            'aggregate_metrics_path': flattened_paths['aggregate_metrics_path'],
            'category_metrics_path': flattened_paths['category_metrics_path'],
        }
        manifest_rows.append(manifest_row)

        aggregate_summary_rows.append(_merge_metadata(manifest_row, aggregate_metrics))
        if not category_metrics.empty:
            for _, cat_row in category_metrics.iterrows():
                category_summary_rows.append(_merge_metadata(manifest_row, cat_row.to_dict()))

    manifest_jsonl = sweep_dir / 'manifest.jsonl'
    manifest_csv = sweep_dir / 'manifest.csv'
    write_jsonl(manifest_jsonl, manifest_rows)
    write_csv(manifest_csv, pd.DataFrame(manifest_rows, columns=LEGACY_MANIFEST_COLUMNS))

    if aggregate_summary_rows:
        write_csv(sweep_dir / 'aggregate_summary.csv', pd.DataFrame(aggregate_summary_rows))
    if category_summary_rows:
        write_csv(sweep_dir / 'category_summary.csv', pd.DataFrame(category_summary_rows))

    if storage.get('cleanup_tmp_run_dirs', True):
        _safe_rmtree(tmp_root)

    print(f"\nSweep finished: {sweep_name}")
    print(f'  manifest: {manifest_csv}')
    if aggregate_summary_rows:
        print(f'  aggregate summary: {sweep_dir / "aggregate_summary.csv"}')
    if category_summary_rows:
        print(f'  category summary: {sweep_dir / "category_summary.csv"}')


def _run_single_feature_strength_sweep(sweep_cfg: dict[str, Any], base_cfg: dict[str, Any]) -> None:
    _validate_single_feature_sweep_config(sweep_cfg)
    sweep_name, sweep_dir, generated_cfg_dir, tmp_root, storage = _prepare_sweep_dirs(sweep_cfg, base_cfg)
    strengths = sweep_cfg['sweep']['strengths']
    groups = sweep_cfg['sweep']['groups']

    total_runs = sum(len(group['features']) * len(strengths) for group in groups)
    print(f"=== sweep :: {sweep_name} ===")
    print('mode: single_feature_strength')
    print(f'total runs: {total_runs}')
    print(f"storage layout: {storage.get('layout', 'flat')}")

    manifest_rows: list[dict[str, Any]] = []
    aggregate_summary_rows: list[dict[str, Any]] = []
    category_summary_rows: list[dict[str, Any]] = []

    seen_run_names: set[str] = set()
    run_counter = 0

    for group_idx, group in enumerate(groups):
        vocab = group.get('vocab')
        hookpoint = str(group['hookpoint'])
        module_path = group.get('module_path')
        sae_file = group.get('sae_file')
        sae_repo = group.get('sae_repo')
        sae_id = group.get('sae_id')
        features = group['features']

        for feature_index in features:
            feature_index = int(feature_index)
            for strength in strengths:
                run_counter += 1
                run_cfg = copy.deepcopy(base_cfg)
                run_cfg['output']['root_dir'] = str(tmp_root)
                set_by_dotted_path(run_cfg, 'steering.hookpoint', hookpoint)
                set_by_dotted_path(run_cfg, 'steering.feature_indices', [feature_index])
                set_by_dotted_path(run_cfg, 'steering.strength', strength)

                if module_path is not None:
                    set_by_dotted_path(run_cfg, 'steering.module_path', module_path)
                if sae_file is not None:
                    set_by_dotted_path(run_cfg, 'steering.sae_file', sae_file)
                if sae_repo is not None:
                    set_by_dotted_path(run_cfg, 'steering.sae_repo', sae_repo)
                if sae_id is not None:
                    set_by_dotted_path(run_cfg, 'steering.sae_id', sae_id)

                run_name = _build_single_feature_run_name(
                    experiment_prefix=sweep_name,
                    vocab=None if vocab is None else str(vocab),
                    hookpoint=hookpoint,
                    feature_index=feature_index,
                    strength=strength,
                )
                if run_name in seen_run_names:
                    raise ValueError(
                        'Generated duplicate run name. Add a vocab label or adjust your sweep config: '
                        f'{run_name}'
                    )
                seen_run_names.add(run_name)

                artifact_run_name = _artifact_file_stem(run_name)
                run_cfg['experiment_name'] = artifact_run_name
                run_cfg['run_metadata'] = {
                    'sweep_name': sweep_name,
                    'sweep_mode': 'single_feature_strength',
                    'group_index': group_idx,
                    'vocab': vocab,
                    'hookpoint': hookpoint,
                    'module_path': module_path,
                    'sae_file': sae_file,
                    'sae_repo': sae_repo,
                    'sae_id': sae_id,
                    'feature_index': feature_index,
                    'strength': strength,
                }

                cfg_path = None
                if generated_cfg_dir is not None:
                    cfg_path = generated_cfg_dir / f'{_artifact_file_stem(run_name)}.yaml'
                    dump_yaml(cfg_path, run_cfg)

                _emit_run_start(
                    run_counter,
                    total_runs,
                    run_name=run_name,
                    vocab=None if vocab is None else str(vocab),
                    hookpoint=hookpoint,
                    feature_index=feature_index,
                    strength=strength,
                    config_path=cfg_path,
                )
                try:
                    result = run_eval(run_cfg)
                    print(f"  finished: {run_name}")
                finally:
                    _release_cuda_memory()

                aggregate_metrics = _load_single_row_csv(result.get('aggregate_metrics_path'))
                category_metrics = _load_multi_row_csv(result.get('category_metrics_path'))

                flattened_paths = _flatten_run_artifacts(
                    sweep_dir=sweep_dir,
                    run_name=run_name,
                    result=result,
                    storage=storage,
                )

                manifest_row = {
                    'run_name': run_name,
                    'vocab': vocab,
                    'hookpoint': hookpoint,
                    'module_path': module_path,
                    'sae_file': sae_file,
                    'sae_repo': sae_repo,
                    'sae_id': sae_id,
                    'feature_index': feature_index,
                    'strength': strength,
                    'config_path': str(cfg_path) if cfg_path is not None else None,
                    'predictions_path': flattened_paths['predictions_path'],
                    'results_path': flattened_paths['results_path'],
                    'example_metrics_path': flattened_paths['example_metrics_path'],
                    'aggregate_metrics_path': flattened_paths['aggregate_metrics_path'],
                    'category_metrics_path': flattened_paths['category_metrics_path'],
                }
                manifest_rows.append(manifest_row)

                aggregate_summary_rows.append(_merge_metadata(manifest_row, aggregate_metrics))
                if not category_metrics.empty:
                    for _, cat_row in category_metrics.iterrows():
                        category_summary_rows.append(_merge_metadata(manifest_row, cat_row.to_dict()))

    manifest_df = pd.DataFrame(manifest_rows, columns=SINGLE_FEATURE_MANIFEST_COLUMNS)
    manifest_jsonl = sweep_dir / 'manifest.jsonl'
    manifest_csv = sweep_dir / 'manifest.csv'
    write_jsonl(manifest_jsonl, manifest_rows)
    write_csv(manifest_csv, manifest_df)

    if aggregate_summary_rows:
        write_csv(sweep_dir / 'aggregate_summary.csv', pd.DataFrame(aggregate_summary_rows))
    if category_summary_rows:
        write_csv(sweep_dir / 'category_summary.csv', pd.DataFrame(category_summary_rows))

    if storage.get('cleanup_tmp_run_dirs', True):
        _safe_rmtree(tmp_root)

    print(f"\nSweep finished: {sweep_name}")
    print(f'  manifest: {manifest_csv}')
    if aggregate_summary_rows:
        print(f'  aggregate summary: {sweep_dir / "aggregate_summary.csv"}')
    if category_summary_rows:
        print(f'  category summary: {sweep_dir / "category_summary.csv"}')



def _run_clarq_single_feature_strength_sweep(sweep_cfg: dict[str, Any], base_cfg: dict[str, Any]) -> None:
    _validate_single_feature_sweep_config(sweep_cfg)
    sweep_name, sweep_dir, generated_cfg_dir, tmp_root, storage = _prepare_sweep_dirs(sweep_cfg, base_cfg)
    strengths = sweep_cfg['sweep']['strengths']
    groups = sweep_cfg['sweep']['groups']

    total_runs = sum(len(group['features']) * len(strengths) for group in groups)
    print(f"=== sweep :: {sweep_name} ===")
    print('mode: clarq single_feature_strength')
    print(f'total runs: {total_runs}')
    print(f"storage layout: {storage.get('layout', 'flat')}")

    manifest_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []

    seen_run_names: set[str] = set()
    run_counter = 0

    for group_idx, group in enumerate(groups):
        vocab = group.get('vocab')
        hookpoint = str(group['hookpoint'])
        module_path = group.get('module_path')
        sae_file = group.get('sae_file')
        sae_repo = group.get('sae_repo')
        sae_id = group.get('sae_id')
        features = group['features']

        for feature_index in features:
            feature_index = int(feature_index)
            for strength in strengths:
                run_counter += 1
                run_cfg = copy.deepcopy(base_cfg)
                run_cfg['output']['root_dir'] = str(tmp_root)

                # Turn steering on explicitly for ClarQ sweeps
                if 'steering' not in run_cfg:
                    run_cfg['steering'] = {}
                set_by_dotted_path(run_cfg, 'steering.enabled', True)
                set_by_dotted_path(run_cfg, 'steering.hookpoint', hookpoint)
                set_by_dotted_path(run_cfg, 'steering.feature_indices', [feature_index])
                set_by_dotted_path(run_cfg, 'steering.strength', strength)

                if module_path is not None:
                    set_by_dotted_path(run_cfg, 'steering.module_path', module_path)
                if sae_file is not None:
                    set_by_dotted_path(run_cfg, 'steering.sae_file', sae_file)
                if sae_repo is not None:
                    set_by_dotted_path(run_cfg, 'steering.sae_repo', sae_repo)
                if sae_id is not None:
                    set_by_dotted_path(run_cfg, 'steering.sae_id', sae_id)

                run_name = _build_single_feature_run_name(
                    experiment_prefix=sweep_name,
                    vocab=None if vocab is None else str(vocab),
                    hookpoint=hookpoint,
                    feature_index=feature_index,
                    strength=strength,
                )
                if run_name in seen_run_names:
                    raise ValueError(
                        'Generated duplicate run name. Add a vocab label or adjust your sweep config: '
                        f'{run_name}'
                    )
                seen_run_names.add(run_name)

                artifact_run_name = _artifact_file_stem(run_name)
                run_cfg['experiment_name'] = artifact_run_name
                run_cfg['run_metadata'] = {
                    'sweep_name': sweep_name,
                    'sweep_mode': 'clarq_single_feature_strength',
                    'group_index': group_idx,
                    'vocab': vocab,
                    'hookpoint': hookpoint,
                    'module_path': module_path,
                    'sae_file': sae_file,
                    'sae_repo': sae_repo,
                    'sae_id': sae_id,
                    'feature_index': feature_index,
                    'strength': strength,
                }

                cfg_path = None
                if generated_cfg_dir is not None:
                    cfg_path = generated_cfg_dir / f'{_artifact_file_stem(run_name)}.yaml'
                    dump_yaml(cfg_path, run_cfg)

                _emit_run_start(
                    run_counter,
                    total_runs,
                    run_name=run_name,
                    vocab=None if vocab is None else str(vocab),
                    hookpoint=hookpoint,
                    feature_index=feature_index,
                    strength=strength,
                    config_path=cfg_path,
                )
                try:
                    result = run_clarq_eval(run_cfg)
                    print(f"  finished: {run_name}")
                finally:
                    _release_cuda_memory()

                summary_metrics = _load_single_row_csv(result.get('summary_path'))
                run_metrics_df = _load_multi_row_csv(result.get('metrics_path'))

                flattened_paths = _flatten_clarq_run_artifacts(
                    sweep_dir=sweep_dir,
                    run_name=run_name,
                    result=result,
                    storage=storage,
                )

                manifest_row = {
                    'run_name': run_name,
                    'vocab': vocab,
                    'hookpoint': hookpoint,
                    'module_path': module_path,
                    'sae_file': sae_file,
                    'sae_repo': sae_repo,
                    'sae_id': sae_id,
                    'feature_index': feature_index,
                    'strength': strength,
                    'config_path': str(cfg_path) if cfg_path is not None else None,
                    'results_path': flattened_paths['results_path'],
                    'metrics_path': flattened_paths['metrics_path'],
                    'summary_path': flattened_paths['summary_path'],
                    'report_path': flattened_paths['report_path'],
                }
                manifest_rows.append(manifest_row)

                if summary_metrics:
                    summary_rows.append(_merge_metadata(manifest_row, summary_metrics))
                if not run_metrics_df.empty:
                    for _, metric_row in run_metrics_df.iterrows():
                        metrics_rows.append(_merge_metadata(manifest_row, metric_row.to_dict()))

    manifest_df = pd.DataFrame(manifest_rows, columns=CLARQ_SINGLE_FEATURE_MANIFEST_COLUMNS)
    manifest_jsonl = sweep_dir / 'manifest.jsonl'
    manifest_csv = sweep_dir / 'manifest.csv'
    write_jsonl(manifest_jsonl, manifest_rows)
    write_csv(manifest_csv, manifest_df)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        available_summary_cols = [c for c in CLARQ_COMPACT_SUMMARY_COLUMNS if c in summary_df.columns]
        if available_summary_cols:
            summary_df = summary_df.loc[:, available_summary_cols]
        write_csv(sweep_dir / 'clarq_summary.csv', summary_df)
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        write_csv(sweep_dir / 'clarq_metrics.csv', metrics_df)

    feature_dashboard_manifest: list[dict[str, Any]] = []
    if storage.get('keep_clarq_feature_dashboards', True) and summary_rows:
        feature_dashboard_manifest = _write_clarq_feature_dashboards(
            sweep_dir=sweep_dir,
            sweep_name=sweep_name,
            summary_rows=summary_rows,
            metrics_rows=metrics_rows,
        )

    if storage.get('cleanup_tmp_run_dirs', True):
        _safe_rmtree(tmp_root)

    print(f"\nSweep finished: {sweep_name}")
    print(f'  manifest: {manifest_csv}')
    if summary_rows:
        print(f'  summary: {sweep_dir / "clarq_summary.csv"}')
    if metrics_rows:
        print(f'  metrics: {sweep_dir / "clarq_metrics.csv"}')
    if feature_dashboard_manifest:
        print(f'  feature dashboards: {sweep_dir / "clarq_feature_dashboards.html"}')
    reports_dir = sweep_dir / 'reports'
    if reports_dir.exists() and any(reports_dir.glob('*.html')):
        print(f'  per-run reports: {reports_dir}')


def _run_clarq_multi_feature_strength_sweep(sweep_cfg: dict[str, Any], base_cfg: dict[str, Any]) -> None:
    _validate_multi_feature_sweep_config(sweep_cfg)
    sweep_name, sweep_dir, generated_cfg_dir, tmp_root, storage = _prepare_sweep_dirs(sweep_cfg, base_cfg)
    strengths = sweep_cfg['sweep']['strengths']
    groups = sweep_cfg['sweep']['groups']

    feature_sets_by_group = [_feature_sets_from_group(group) for group in groups]
    total_runs = sum(len(feature_sets) * len(strengths) for feature_sets in feature_sets_by_group)
    print(f"=== sweep :: {sweep_name} ===")
    print('mode: clarq multi_feature_strength')
    print(f'total runs: {total_runs}')
    print(f"storage layout: {storage.get('layout', 'flat')}")

    manifest_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []

    seen_run_names: set[str] = set()
    run_counter = 0

    for group_idx, (group, feature_sets) in enumerate(zip(groups, feature_sets_by_group)):
        vocab = group.get('vocab')
        hookpoint = str(group['hookpoint'])
        module_path = group.get('module_path')
        sae_file = group.get('sae_file')
        sae_repo = group.get('sae_repo')
        sae_id = group.get('sae_id')
        group_weights = group.get('weights')
        group_normalize_each = bool(group.get('normalize_each', False))
        group_norm_cap = group.get('norm_cap')

        for feature_set_idx, feature_set in enumerate(feature_sets):
            feature_indices = [int(x) for x in feature_set['features']]
            feature_set_label = str(feature_set['label'])
            feature_weights = feature_set.get('weights', group_weights)
            normalize_each = bool(feature_set.get('normalize_each', group_normalize_each))
            norm_cap = feature_set.get('norm_cap', group_norm_cap)

            if feature_weights is not None:
                feature_weights = [float(x) for x in feature_weights]
                if len(feature_weights) != len(feature_indices):
                    raise ValueError(
                        f"Weights for feature set {feature_set_label!r} must match feature count "
                        f"({len(feature_weights)} != {len(feature_indices)})."
                    )

            for strength in strengths:
                run_counter += 1
                run_cfg = copy.deepcopy(base_cfg)
                run_cfg['output']['root_dir'] = str(tmp_root)

                if 'steering' not in run_cfg:
                    run_cfg['steering'] = {}
                set_by_dotted_path(run_cfg, 'steering.enabled', True)
                set_by_dotted_path(run_cfg, 'steering.hookpoint', hookpoint)
                set_by_dotted_path(run_cfg, 'steering.feature_indices', feature_indices)
                set_by_dotted_path(run_cfg, 'steering.feature_weights', feature_weights)
                set_by_dotted_path(run_cfg, 'steering.normalize_each', normalize_each)
                set_by_dotted_path(run_cfg, 'steering.norm_cap', norm_cap)
                set_by_dotted_path(run_cfg, 'steering.strength', strength)

                if module_path is not None:
                    set_by_dotted_path(run_cfg, 'steering.module_path', module_path)
                if sae_file is not None:
                    set_by_dotted_path(run_cfg, 'steering.sae_file', sae_file)
                if sae_repo is not None:
                    set_by_dotted_path(run_cfg, 'steering.sae_repo', sae_repo)
                if sae_id is not None:
                    set_by_dotted_path(run_cfg, 'steering.sae_id', sae_id)

                run_name = _build_feature_set_run_name(
                    experiment_prefix=sweep_name,
                    vocab=None if vocab is None else str(vocab),
                    hookpoint=hookpoint,
                    feature_set_label=feature_set_label,
                    feature_indices=feature_indices,
                    strength=strength,
                )
                if run_name in seen_run_names:
                    raise ValueError(
                        'Generated duplicate run name. Add a unique label to each feature set: '
                        f'{run_name}'
                    )
                seen_run_names.add(run_name)

                artifact_run_name = _artifact_file_stem(run_name)
                run_cfg['experiment_name'] = artifact_run_name
                run_cfg['run_metadata'] = {
                    'sweep_name': sweep_name,
                    'sweep_mode': 'clarq_multi_feature_strength',
                    'group_index': group_idx,
                    'feature_set_index': feature_set_idx,
                    'vocab': vocab,
                    'hookpoint': hookpoint,
                    'module_path': module_path,
                    'sae_file': sae_file,
                    'sae_repo': sae_repo,
                    'sae_id': sae_id,
                    'feature_set_label': feature_set_label,
                    'feature_indices': feature_indices,
                    'feature_count': len(feature_indices),
                    'feature_weights': feature_weights,
                    'normalize_each': normalize_each,
                    'norm_cap': norm_cap,
                    'strength': strength,
                }

                cfg_path = None
                if generated_cfg_dir is not None:
                    cfg_path = generated_cfg_dir / f'{_artifact_file_stem(run_name)}.yaml'
                    dump_yaml(cfg_path, run_cfg)

                _emit_run_start(
                    run_counter,
                    total_runs,
                    run_name=run_name,
                    vocab=None if vocab is None else str(vocab),
                    hookpoint=hookpoint,
                    feature_set_label=feature_set_label,
                    feature_indices=feature_indices,
                    strength=strength,
                    config_path=cfg_path,
                )
                try:
                    result = run_clarq_eval(run_cfg)
                    print(f"  finished: {run_name}")
                finally:
                    _release_cuda_memory()

                summary_metrics = _load_single_row_csv(result.get('summary_path'))
                run_metrics_df = _load_multi_row_csv(result.get('metrics_path'))

                flattened_paths = _flatten_clarq_run_artifacts(
                    sweep_dir=sweep_dir,
                    run_name=run_name,
                    result=result,
                    storage=storage,
                )

                manifest_row = {
                    'run_name': run_name,
                    'vocab': vocab,
                    'hookpoint': hookpoint,
                    'module_path': module_path,
                    'sae_file': sae_file,
                    'sae_repo': sae_repo,
                    'sae_id': sae_id,
                    'feature_set_label': feature_set_label,
                    'feature_indices': feature_indices,
                    'feature_count': len(feature_indices),
                    'feature_weights': feature_weights,
                    'normalize_each': normalize_each,
                    'norm_cap': norm_cap,
                    'strength': strength,
                    'config_path': str(cfg_path) if cfg_path is not None else None,
                    'results_path': flattened_paths['results_path'],
                    'metrics_path': flattened_paths['metrics_path'],
                    'summary_path': flattened_paths['summary_path'],
                    'report_path': flattened_paths['report_path'],
                }
                manifest_rows.append(manifest_row)

                # Keep the existing dashboard/summary code working: for a multi-feature
                # run, feature_index is a human-readable set label.
                summary_metadata = dict(manifest_row)
                summary_metadata['feature_index'] = feature_set_label
                if summary_metrics:
                    summary_rows.append(_merge_metadata(summary_metadata, summary_metrics))
                if not run_metrics_df.empty:
                    for _, metric_row in run_metrics_df.iterrows():
                        metrics_rows.append(_merge_metadata(summary_metadata, metric_row.to_dict()))

    manifest_df = pd.DataFrame(manifest_rows, columns=CLARQ_MULTI_FEATURE_MANIFEST_COLUMNS)
    manifest_jsonl = sweep_dir / 'manifest.jsonl'
    manifest_csv = sweep_dir / 'manifest.csv'
    write_jsonl(manifest_jsonl, manifest_rows)
    write_csv(manifest_csv, manifest_df)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        available_summary_cols = [c for c in CLARQ_COMPACT_SUMMARY_COLUMNS if c in summary_df.columns]
        if available_summary_cols:
            summary_df = summary_df.loc[:, available_summary_cols]
        write_csv(sweep_dir / 'clarq_summary.csv', summary_df)
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        write_csv(sweep_dir / 'clarq_metrics.csv', metrics_df)

    feature_dashboard_manifest: list[dict[str, Any]] = []
    if storage.get('keep_clarq_feature_dashboards', True) and summary_rows:
        feature_dashboard_manifest = _write_clarq_feature_dashboards(
            sweep_dir=sweep_dir,
            sweep_name=sweep_name,
            summary_rows=summary_rows,
            metrics_rows=metrics_rows,
        )

    if storage.get('cleanup_tmp_run_dirs', True):
        _safe_rmtree(tmp_root)

    print(f"\nSweep finished: {sweep_name}")
    print(f'  manifest: {manifest_csv}')
    if summary_rows:
        print(f'  summary: {sweep_dir / "clarq_summary.csv"}')
    if metrics_rows:
        print(f'  metrics: {sweep_dir / "clarq_metrics.csv"}')
    if feature_dashboard_manifest:
        print(f'  feature dashboards: {sweep_dir / "clarq_feature_dashboards.html"}')
    reports_dir = sweep_dir / 'reports'
    if reports_dir.exists() and any(reports_dir.glob('*.html')):
        print(f'  per-run reports: {reports_dir}')


if __name__ == '__main__':
    args = parse_args()
    sweep_cfg = load_yaml(args.config)
    base_cfg = load_yaml(sweep_cfg['base_config'])

    sweep_section = sweep_cfg.get('sweep', {})
    sweep_mode = sweep_section.get('mode', 'legacy')
    is_clarq_base = 'clarq' in base_cfg

    if 'parameter' in sweep_section and 'values' in sweep_section:
        _run_legacy_sweep(sweep_cfg, base_cfg)
    elif sweep_mode == 'single_feature_strength':
        if is_clarq_base:
            _run_clarq_single_feature_strength_sweep(sweep_cfg, base_cfg)
        else:
            _run_single_feature_strength_sweep(sweep_cfg, base_cfg)
    elif sweep_mode in {'multi_feature_strength', 'feature_set_strength', 'cluster_feature_strength'}:
        if not is_clarq_base:
            raise ValueError('multi_feature_strength is currently implemented for ClarQ base configs.')
        _run_clarq_multi_feature_strength_sweep(sweep_cfg, base_cfg)
    else:
        raise ValueError(
            'Unsupported sweep config. Use either legacy sweep.parameter/sweep.values, '
            'sweep.mode=single_feature_strength, or sweep.mode=multi_feature_strength.'
        )
