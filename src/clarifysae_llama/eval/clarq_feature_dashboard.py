from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from clarifysae_llama.eval.clarq_html_report import _extract_dialogues, _fmt_metric, _load_optional_csv
from clarifysae_llama.utils.io import ensure_dir


_FEATURE_CSS = """
:root {
  color-scheme: light dark;
  --bg: #0b1020;
  --panel: #121933;
  --panel-2: #182241;
  --text: #eaf0ff;
  --muted: #aab7d9;
  --accent: #82aaff;
  --accent-2: #c792ea;
  --good: #4caf50;
  --bad: #ef5350;
  --warn: #ffb74d;
  --border: rgba(255,255,255,0.12);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.container { max-width: 1500px; margin: 0 auto; padding: 24px; }
.header h1 { margin: 0 0 8px; font-size: 30px; }
.header p { margin: 0; color: var(--muted); }
.grid { display: grid; gap: 12px; }
.meta-grid, .summary-grid { grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin-top: 20px; }
.card {
  background: linear-gradient(180deg, var(--panel), var(--panel-2));
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 16px;
}
.card h3 { margin: 0 0 8px; font-size: 14px; color: var(--muted); font-weight: 600; }
.card .value { font-size: 24px; font-weight: 700; }
.section-title { margin: 28px 0 12px; font-size: 22px; }
.small { font-size: 13px; }
.muted { color: var(--muted); }
.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
  margin-bottom: 12px;
}
.controls select {
  background: var(--panel);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 12px;
}
.table-wrap {
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 16px;
  background: var(--panel);
}
table { width: 100%; border-collapse: collapse; min-width: 900px; }
th, td { padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }
th { background: #10172f; white-space: nowrap; }
tr:hover td { background: rgba(255,255,255,0.03); }
.badge { display: inline-block; min-width: 28px; text-align: center; padding: 3px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }
.badge.good { background: rgba(76, 175, 80, 0.18); color: #9be7a3; }
.badge.bad { background: rgba(239, 83, 80, 0.18); color: #ffb3b0; }
.badge.warn { background: rgba(255, 183, 77, 0.18); color: #ffd79f; }
.strength-section.hidden { display: none; }
.strength-section { margin-top: 16px; }
.dialogue { margin-top: 16px; border: 1px solid var(--border); border-radius: 16px; overflow: hidden; background: linear-gradient(180deg, var(--panel), var(--panel-2)); }
.dialogue summary {
  cursor: pointer;
  list-style: none;
  padding: 14px 16px;
  display: flex;
  gap: 12px;
  justify-content: space-between;
  align-items: center;
}
.dialogue summary::-webkit-details-marker { display: none; }
.dialogue-body { padding: 16px; display: grid; grid-template-columns: 1.4fr 1fr; gap: 16px; border-top: 1px solid var(--border); }
@media (max-width: 1100px) { .dialogue-body { grid-template-columns: 1fr; } }
.transcript, .gold-box, .background-box {
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px;
}
.turn { margin: 0 0 12px; white-space: pre-wrap; }
.turn:last-child { margin-bottom: 0; }
.turn .speaker { font-weight: 700; color: var(--accent); }
.turn.seeker .speaker { color: var(--accent-2); }
.list { margin: 0; padding-left: 18px; }
.footer { margin-top: 28px; color: var(--muted); font-size: 13px; }
.index-table td, .index-table th { vertical-align: middle; }
.codeish { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
"""


_FEATURE_JS = """
function initStrengthSelector(selectId, sectionClass) {
  const select = document.getElementById(selectId);
  const sections = Array.from(document.querySelectorAll(`.${sectionClass}`));
  function apply() {
    const value = select.value;
    sections.forEach((section) => {
      section.classList.toggle('hidden', section.dataset.strength !== value);
    });
  }
  select.addEventListener('change', apply);
  apply();
}
"""


def _slugify(value: str) -> str:
    token = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value).strip())
    token = re.sub(r'_+', '_', token).strip('._-')
    return token or 'value'


def _feature_slug(vocab: Any, hookpoint: Any, feature_index: Any) -> str:
    vocab_part = _slugify(vocab) if vocab not in (None, '') else 'na'
    return f"feature__{vocab_part}__{_slugify(hookpoint)}__feat{_slugify(feature_index)}"


def _as_badge(value: Any) -> str:
    if value == 1:
        return "<span class='badge good'>yes</span>"
    if value == 0:
        return "<span class='badge bad'>no</span>"
    return "<span class='badge warn'>—</span>"


def _safe_text(value: Any) -> str:
    if value is None:
        return '—'
    try:
        if isinstance(value, float) and math.isnan(value):
            return '—'
    except Exception:
        pass
    return str(value)


def _load_payload(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def _dialogues_for_run(results_path: str | Path | None, metrics_path: str | Path | None) -> list[dict[str, Any]]:
    payload = _load_payload(results_path)
    if payload is None:
        return []
    metrics_df = _load_optional_csv(metrics_path)
    return _extract_dialogues(payload, metrics_df)


def _strength_sort_key(value: Any) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except Exception:
        return (1, str(value))


def _build_feature_page(
    *,
    page_path: Path,
    sweep_name: str,
    feature_meta: dict[str, Any],
    run_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    reports_dir: Path,
    results_dir: Path,
) -> None:
    strengths = sorted([row.get('strength') for row in run_rows], key=_strength_sort_key)

    meta_cards = [
        ('Sweep', sweep_name),
        ('Feature', feature_meta.get('feature_index')),
        ('Vocab', feature_meta.get('vocab')),
        ('Hookpoint', feature_meta.get('hookpoint')),
        ('Module path', feature_meta.get('module_path')),
        ('SAE file', feature_meta.get('sae_file')),
        ('Strengths', ', '.join(_safe_text(x) for x in strengths)),
    ]
    meta_html = ''.join(
        f"<div class='card'><h3>{html.escape(label)}</h3><div class='small'>{html.escape(_safe_text(value))}</div></div>"
        for label, value in meta_cards
    )

    summary_rows_sorted = sorted(summary_rows, key=lambda row: _strength_sort_key(row.get('strength')))
    summary_table_rows = []
    best_success = None
    best_recall = None
    for row in summary_rows_sorted:
        success = row.get('success_rate')
        recall = row.get('step_recall')
        try:
            if success is not None and (best_success is None or float(success) > float(best_success[1])):
                best_success = (row.get('strength'), success)
        except Exception:
            pass
        try:
            if recall is not None and (best_recall is None or float(recall) > float(best_recall[1])):
                best_recall = (row.get('strength'), recall)
        except Exception:
            pass
        summary_table_rows.append(
            f"""
            <tr>
              <td>{html.escape(_safe_text(row.get('strength')))}</td>
              <td>{html.escape(_fmt_metric(row.get('success_rate')))}</td>
              <td>{html.escape(_fmt_metric(row.get('step_recall')))}</td>
              <td>{html.escape(_fmt_metric(row.get('ClarQ_count')))}</td>
              <td>{html.escape(_fmt_metric(row.get('ClarQ_rate')))}</td>
              <td>{html.escape(_fmt_metric(row.get('ClarQ_depth')))}</td>
              <td>{html.escape(_fmt_metric(row.get('Goodbye_rate')))}</td>
              <td>{html.escape(_fmt_metric(row.get('ARL')))}</td>
              <td>{html.escape(_fmt_metric(row.get('AQD')))}</td>
            </tr>
            """
        )

    best_cards = [
        ('Best success strength', best_success[0] if best_success else None),
        ('Best success rate', best_success[1] if best_success else None),
        ('Best recall strength', best_recall[0] if best_recall else None),
        ('Best step recall', best_recall[1] if best_recall else None),
    ]
    best_cards_html = ''.join(
        f"<div class='card'><h3>{html.escape(label)}</h3><div class='value'>{html.escape(_fmt_metric(value))}</div></div>"
        for label, value in best_cards
    )

    strength_options = ''.join(
        f"<option value='{html.escape(_safe_text(strength))}'>{html.escape(_safe_text(strength))}</option>"
        for strength in strengths
    )

    by_strength = {_safe_text(row.get('strength')): row for row in run_rows}
    strength_sections = []
    for strength in strengths:
        run_row = by_strength.get(_safe_text(strength))
        if run_row is None:
            continue
        run_summary = next((row for row in summary_rows_sorted if _safe_text(row.get('strength')) == _safe_text(strength)), {})
        dialogues = _dialogues_for_run(run_row.get('results_path'), run_row.get('metrics_path'))
        report_path = Path(run_row['report_path']) if run_row.get('report_path') else None
        results_path = Path(run_row['results_path']) if run_row.get('results_path') else None

        metric_cards = [
            ('Success rate', run_summary.get('success_rate')),
            ('Step recall', run_summary.get('step_recall')),
            ('ClarQ count', run_summary.get('ClarQ_count')),
            ('ClarQ rate', run_summary.get('ClarQ_rate')),
            ('ClarQ depth', run_summary.get('ClarQ_depth')),
            ('Goodbye rate', run_summary.get('Goodbye_rate')),
            ('ARL', run_summary.get('ARL')),
            ('AQD', run_summary.get('AQD')),
        ]
        metric_cards_html = ''.join(
            f"<div class='card'><h3>{html.escape(label)}</h3><div class='value'>{html.escape(_fmt_metric(value))}</div></div>"
            for label, value in metric_cards
        )

        dialogue_rows = []
        dialogue_cards = []
        for d in dialogues:
            dialogue_rows.append(
                f"""
                <tr>
                  <td>{d['task_type_index']}</td>
                  <td>{d['dialogue_index']}</td>
                  <td>{html.escape(_safe_text(d['task_name']))}</td>
                  <td>{_as_badge(d.get('success'))}</td>
                  <td>{html.escape(_fmt_metric(d.get('step_recall')))}</td>
                  <td>{html.escape(_fmt_metric(d.get('ClarQ_count')))}</td>
                  <td>{html.escape(_fmt_metric(d.get('ClarQ_rate')))}</td>
                  <td>{html.escape(_fmt_metric(d.get('ClarQ_depth')))}</td>
                  <td>{_as_badge(d.get('Goodbye'))}</td>
                  <td>{html.escape(_fmt_metric(d.get('total_turns')))}</td>
                </tr>
                """
            )
            transcript_parts = []
            for idx, turn in enumerate(d.get('transcript', [])):
                speaker = 'Jax' if idx % 2 == 0 else 'Seeker'
                speaker_class = 'helper' if idx % 2 == 0 else 'seeker'
                transcript_parts.append(
                    f"<div class='turn {speaker_class}'><div class='speaker'>{speaker}</div><div>{html.escape(str(turn))}</div></div>"
                )
            gold_steps = ''.join(
                f"<li><span class='codeish'>{html.escape(str(gs))}</span>: {html.escape(str(expl))}</li>"
                for gs, expl in zip(d.get('gold_structure', []), d.get('gold_clarifications', []))
            )
            background_list = ''.join(
                f"<li>{html.escape(str(item))}</li>" for item in d.get('background_splitted', [])
            )
            dialogue_cards.append(
                f"""
                <details class='dialogue'>
                  <summary>
                    <div><strong>Task type {d['task_type_index']} · dialogue {d['dialogue_index']}</strong> · {html.escape(_safe_text(d['task_name']))}</div>
                    <div class='small muted'>success: {html.escape(_fmt_metric(d.get('success')))} · step_recall: {html.escape(_fmt_metric(d.get('step_recall')))} · goodbye: {html.escape(_fmt_metric(d.get('Goodbye')))}</div>
                  </summary>
                  <div class='dialogue-body'>
                    <div>
                      <div class='transcript'>
                        <h3>Transcript</h3>
                        {''.join(transcript_parts) or '<div class="muted">No transcript available.</div>'}
                      </div>
                      <div class='background-box' style='margin-top:16px;'>
                        <h3>Task background</h3>
                        <div class='small' style='white-space: pre-wrap;'>{html.escape(str(d.get('background', '')))}</div>
                      </div>
                    </div>
                    <div>
                      <div class='gold-box'>
                        <h3>Gold clarification targets</h3>
                        <ol class='list small'>{gold_steps or '<li>No gold clarifications available.</li>'}</ol>
                      </div>
                      <div class='gold-box' style='margin-top:16px;'>
                        <h3>Gold full provider response</h3>
                        <div class='small' style='white-space: pre-wrap;'>{html.escape(str(d.get('all_response', '')))}</div>
                      </div>
                      <div class='gold-box' style='margin-top:16px;'>
                        <h3>Background slots</h3>
                        <ul class='list small'>{background_list or '<li>No background slots available.</li>'}</ul>
                      </div>
                    </div>
                  </div>
                </details>
                """
            )

        links = []
        if report_path is not None and report_path.exists() and reports_dir in report_path.parents:
            links.append(f"<a href='../reports/{html.escape(report_path.name)}'>per-run report</a>")
        if results_path is not None and results_path.exists() and results_dir in results_path.parents:
            links.append(f"<a href='../results/{html.escape(results_path.name)}'>results json</a>")
        link_html = ' · '.join(links) if links else '—'

        strength_sections.append(
            f"""
            <section class='strength-section feature-strength-section' data-strength='{html.escape(_safe_text(strength))}'>
              <div class='summary-grid grid'>{metric_cards_html}</div>
              <div class='small muted' style='margin-top:12px;'>Artifacts: {link_html}</div>
              <h3 class='section-title'>Per-dialogue overview · strength {html.escape(_safe_text(strength))}</h3>
              <div class='table-wrap'>
                <table>
                  <thead>
                    <tr>
                      <th>task type</th>
                      <th>dialogue</th>
                      <th>task</th>
                      <th>success</th>
                      <th>step recall</th>
                      <th>ClarQ count</th>
                      <th>ClarQ rate</th>
                      <th>ClarQ depth</th>
                      <th>goodbye</th>
                      <th>total turns</th>
                    </tr>
                  </thead>
                  <tbody>
                    {''.join(dialogue_rows) or "<tr><td colspan='10'>No dialogue rows available.</td></tr>"}
                  </tbody>
                </table>
              </div>
              <h3 class='section-title'>Transcripts · strength {html.escape(_safe_text(strength))}</h3>
              {''.join(dialogue_cards) or "<div class='card small'>No transcript content available for this run.</div>"}
            </section>
            """
        )

    html_doc = f"""
    <!doctype html>
    <html lang='en'>
      <head>
        <meta charset='utf-8' />
        <meta name='viewport' content='width=device-width, initial-scale=1' />
        <title>ClarQ feature dashboard</title>
        <style>{_FEATURE_CSS}</style>
      </head>
      <body>
        <div class='container'>
          <div class='header'>
            <h1>ClarQ feature dashboard</h1>
            <p><a href='index.html'>back to feature index</a></p>
          </div>

          <div class='meta-grid grid'>{meta_html}</div>

          <h2 class='section-title'>Best strengths</h2>
          <div class='summary-grid grid'>{best_cards_html}</div>

          <h2 class='section-title'>Strength comparison</h2>
          <div class='table-wrap'>
            <table>
              <thead>
                <tr>
                  <th>strength</th>
                  <th>success rate</th>
                  <th>step recall</th>
                  <th>ClarQ count</th>
                  <th>ClarQ rate</th>
                  <th>ClarQ depth</th>
                  <th>Goodbye rate</th>
                  <th>ARL</th>
                  <th>AQD</th>
                </tr>
              </thead>
              <tbody>
                {''.join(summary_table_rows) or "<tr><td colspan='9'>No summary rows available.</td></tr>"}
              </tbody>
            </table>
          </div>

          <h2 class='section-title'>Inspect one strength</h2>
          <div class='controls'>
            <label for='strengthPicker'>Strength</label>
            <select id='strengthPicker'>{strength_options}</select>
          </div>

          {''.join(strength_sections)}

          <div class='footer'>Feature-level ClarQ dashboard generated automatically during the sweep.</div>
        </div>
        <script>{_FEATURE_JS}\ninitStrengthSelector('strengthPicker', 'feature-strength-section');</script>
      </body>
    </html>
    """
    page_path.write_text(html_doc, encoding='utf-8')


def _build_index_page(
    *,
    index_path: Path,
    sweep_name: str,
    feature_rows: list[dict[str, Any]],
) -> None:
    table_rows = []
    for row in feature_rows:
        table_rows.append(
            f"""
            <tr>
              <td><a href='{html.escape(row['page_name'])}'>{html.escape(_safe_text(row.get('feature_index')))}</a></td>
              <td>{html.escape(_safe_text(row.get('vocab')))}</td>
              <td>{html.escape(_safe_text(row.get('hookpoint')))}</td>
              <td>{html.escape(_safe_text(row.get('module_path')))}</td>
              <td>{html.escape(_safe_text(row.get('sae_file')))}</td>
              <td>{html.escape(_safe_text(row.get('num_strengths')))}</td>
              <td>{html.escape(_safe_text(row.get('best_success_strength')))}</td>
              <td>{html.escape(_fmt_metric(row.get('best_success_rate')))}</td>
              <td>{html.escape(_safe_text(row.get('best_recall_strength')))}</td>
              <td>{html.escape(_fmt_metric(row.get('best_step_recall')))}</td>
            </tr>
            """
        )

    html_doc = f"""
    <!doctype html>
    <html lang='en'>
      <head>
        <meta charset='utf-8' />
        <meta name='viewport' content='width=device-width, initial-scale=1' />
        <title>ClarQ feature dashboards</title>
        <style>{_FEATURE_CSS}</style>
      </head>
      <body>
        <div class='container'>
          <div class='header'>
            <h1>ClarQ feature dashboards</h1>
            <p>Sweep: {html.escape(sweep_name)}</p>
          </div>
          <h2 class='section-title'>Features</h2>
          <div class='table-wrap'>
            <table class='index-table'>
              <thead>
                <tr>
                  <th>feature</th>
                  <th>vocab</th>
                  <th>hookpoint</th>
                  <th>module path</th>
                  <th>SAE file</th>
                  <th># strengths</th>
                  <th>best success strength</th>
                  <th>best success rate</th>
                  <th>best recall strength</th>
                  <th>best step recall</th>
                </tr>
              </thead>
              <tbody>
                {''.join(table_rows) or "<tr><td colspan='10'>No feature dashboards available.</td></tr>"}
              </tbody>
            </table>
          </div>
        </div>
      </body>
    </html>
    """
    index_path.write_text(html_doc, encoding='utf-8')


def build_clarq_feature_dashboards(
    *,
    sweep_name: str,
    sweep_dir: str | Path,
    manifest_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> dict[str, str]:
    sweep_dir = Path(sweep_dir)
    feature_dir = ensure_dir(sweep_dir / 'feature_reports')
    if not manifest_rows:
        return {}

    manifest_df = pd.DataFrame(manifest_rows)
    summary_df = pd.DataFrame(summary_rows)
    feature_dashboard_paths: dict[str, str] = {}
    feature_rows_for_index: list[dict[str, Any]] = []
    reports_dir = sweep_dir / 'reports'
    results_dir = sweep_dir / 'results'

    group_cols = ['vocab', 'hookpoint', 'module_path', 'sae_file', 'feature_index']
    for feature_values, feature_df in manifest_df.groupby(group_cols, dropna=False, sort=False):
        vocab, hookpoint, module_path, sae_file, feature_index = feature_values
        feature_meta = {
            'vocab': vocab,
            'hookpoint': hookpoint,
            'module_path': module_path,
            'sae_file': sae_file,
            'feature_index': feature_index,
        }
        run_rows = feature_df.to_dict(orient='records')
        run_names = set(feature_df['run_name'].tolist())
        feature_summary_rows = summary_df[summary_df['run_name'].isin(run_names)].to_dict(orient='records') if not summary_df.empty else []

        slug = _feature_slug(vocab, hookpoint, feature_index)
        page_path = feature_dir / f'{slug}.html'
        _build_feature_page(
            page_path=page_path,
            sweep_name=sweep_name,
            feature_meta=feature_meta,
            run_rows=run_rows,
            summary_rows=feature_summary_rows,
            reports_dir=reports_dir,
            results_dir=results_dir,
        )

        best_success_strength = None
        best_success_rate = None
        best_recall_strength = None
        best_step_recall = None
        for row in feature_summary_rows:
            try:
                success = row.get('success_rate')
                if success is not None and (best_success_rate is None or float(success) > float(best_success_rate)):
                    best_success_strength = row.get('strength')
                    best_success_rate = success
            except Exception:
                pass
            try:
                recall = row.get('step_recall')
                if recall is not None and (best_step_recall is None or float(recall) > float(best_step_recall)):
                    best_recall_strength = row.get('strength')
                    best_step_recall = recall
            except Exception:
                pass

        page_rel = str(page_path.relative_to(sweep_dir))
        feature_rows_for_index.append(
            {
                **feature_meta,
                'page_name': page_path.name,
                'page_rel': page_rel,
                'num_strengths': len({row.get('strength') for row in feature_summary_rows}) or len({row.get('strength') for row in run_rows}),
                'best_success_strength': best_success_strength,
                'best_success_rate': best_success_rate,
                'best_recall_strength': best_recall_strength,
                'best_step_recall': best_step_recall,
            }
        )
        for run_name in run_names:
            feature_dashboard_paths[run_name] = page_rel

    index_path = feature_dir / 'index.html'
    _build_index_page(index_path=index_path, sweep_name=sweep_name, feature_rows=feature_rows_for_index)
    return feature_dashboard_paths
