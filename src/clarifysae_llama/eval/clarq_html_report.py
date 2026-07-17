from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from clarifysae_llama.utils.io import ensure_dir


def _load_optional_csv(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return int(value)
    except Exception:
        return None


def _fmt_metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return '—'
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return '—'
    except Exception:
        pass
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, int):
        return str(value)
    try:
        fv = float(value)
    except Exception:
        return str(value)
    if fv.is_integer():
        return str(int(fv))
    return f'{fv:.{digits}f}'


def _build_fallback_summary(dialogues: list[dict[str, Any]]) -> dict[str, Any]:
    num_dialogues = len(dialogues)
    if num_dialogues == 0:
        return {
            'num_dialogues': 0,
            'success_rate': None,
            'Goodbye_rate': None,
            'avg_total_turns': None,
            'avg_helper_turns': None,
            'avg_seeker_turns': None,
        }

    success_vals = [d.get('success') for d in dialogues if d.get('success') is not None]
    goodbye_vals = [d.get('Goodbye') for d in dialogues if d.get('Goodbye') is not None]
    total_turns = [d.get('total_turns') for d in dialogues if d.get('total_turns') is not None]
    helper_turns = [d.get('helper_turns') for d in dialogues if d.get('helper_turns') is not None]
    seeker_turns = [d.get('seeker_turns') for d in dialogues if d.get('seeker_turns') is not None]

    def _avg(values: list[Any]) -> float | None:
        if not values:
            return None
        return float(sum(values) / len(values))

    return {
        'num_dialogues': num_dialogues,
        'success_rate': _avg(success_vals),
        'Goodbye_rate': _avg(goodbye_vals),
        'avg_total_turns': _avg(total_turns),
        'avg_helper_turns': _avg(helper_turns),
        'avg_seeker_turns': _avg(seeker_turns),
    }


def _extract_dialogues(payload: dict[str, Any], metrics_df: pd.DataFrame | None) -> list[dict[str, Any]]:
    meta = payload.get('meta', {})
    eval_set = set(meta.get('evaluation_set', []))
    data = payload.get('data', [])

    metrics_by_key: dict[tuple[int, int, int], dict[str, Any]] = {}
    if metrics_df is not None and not metrics_df.empty:
        for _, row in metrics_df.iterrows():
            key = (
                _safe_int(row.get('task_type_index')) or 0,
                _safe_int(row.get('dialogue_index')) or 0,
                _safe_int(row.get('dialogue_slot')) or 0,
            )
            metrics_by_key[key] = {col: row[col] for col in metrics_df.columns}

    dialogues: list[dict[str, Any]] = []
    for task_type_index, one_type in enumerate(data):
        if eval_set and task_type_index not in eval_set:
            continue
        for dialogue_index, conv in enumerate(one_type):
            dialogue_slot = 0
            transcript = list(((conv.get('l2l') or [[[]]])[0] if conv.get('l2l') else []))
            metrics = metrics_by_key.get((task_type_index, dialogue_index, dialogue_slot), {})
            helper_turns = _safe_int(metrics.get('helper_turns'))
            seeker_turns = _safe_int(metrics.get('seeker_turns'))
            success = _safe_int(metrics.get('success'))
            goodbye = _safe_int(metrics.get('Goodbye'))
            step_recall = _safe_float(metrics.get('step_recall'))
            clarq_count = _safe_float(metrics.get('ClarQ_count'))
            clarq_rate = _safe_float(metrics.get('ClarQ_rate'))
            clarq_depth = _safe_float(metrics.get('ClarQ_depth'))
            arl = _safe_float(metrics.get('ARL'))
            aqd = _safe_float(metrics.get('AQD'))
            total_turns = len(transcript)
            if helper_turns is None:
                helper_turns = (total_turns + 1) // 2 if total_turns else None
            if seeker_turns is None:
                seeker_turns = total_turns // 2 if total_turns else None
            if goodbye is None and transcript:
                goodbye = 1 if 'goodbye' in str(transcript[-1]).lower() else 0
            
            dialogues.append(
                {
                    'key': f'{task_type_index}-{dialogue_index}-{dialogue_slot}',
                    'anchor': f'dlg-{task_type_index}-{dialogue_index}-{dialogue_slot}',
                    'task_type_index': task_type_index,
                    'dialogue_index': dialogue_index,
                    'dialogue_slot': dialogue_slot,
                    'task_name': (conv.get('background_splitted') or ['Unknown', 'Unknown'])[1] if conv.get('background_splitted') else 'Unknown',
                    'transcript': transcript,
                    'background': conv.get('background', ''),
                    'background_splitted': conv.get('background_splitted', []),
                    'gold_structure': conv.get('gold_structure', []),
                    'gold_clarifications': conv.get('all_response_exaplain', []),
                    'all_response': conv.get('all_response', ''),
                    'helper_turns': helper_turns,
                    'seeker_turns': seeker_turns,
                    'total_turns': total_turns,
                    'success': success,
                    'Goodbye': goodbye,
                    'step_recall': step_recall,
                    'ClarQ_count': clarq_count,
                    'ClarQ_rate': clarq_rate,
                    'ClarQ_depth': clarq_depth,
                    'ARL': arl,
                    'AQD': aqd,
                }
            )
    return dialogues


CSS = """
:root {
  color-scheme: light dark;
  --bg: #0b1020;
  --panel: #121933;
  --panel-2: #182241;
  --text: #eaf0ff;
  --muted: #aab7d9;
  --accent: #82aaff;
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
.container {
  max-width: 1500px;
  margin: 0 auto;
  padding: 24px;
}
.header h1 { margin: 0 0 8px; font-size: 32px; }
.header p { margin: 0; color: var(--muted); }
.meta-grid, .summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
  margin-top: 20px;
}
.card {
  background: linear-gradient(180deg, var(--panel), var(--panel-2));
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}
.card h3 { margin: 0 0 8px; font-size: 14px; color: var(--muted); font-weight: 600; }
.card .value { font-size: 24px; font-weight: 700; }
.section-title { margin: 28px 0 12px; font-size: 22px; }
.controls {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  align-items: center;
  margin-bottom: 12px;
}
.controls input, .controls select {
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
table {
  width: 100%;
  border-collapse: collapse;
  min-width: 1000px;
}
th, td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  text-align: left;
  vertical-align: top;
}
th {
  position: sticky;
  top: 0;
  background: #10172f;
  cursor: pointer;
  white-space: nowrap;
}
tr:hover td { background: rgba(255,255,255,0.03); }
.badge {
  display: inline-block;
  min-width: 28px;
  text-align: center;
  padding: 3px 8px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
}
.badge.good { background: rgba(76, 175, 80, 0.18); color: #9be7a3; }
.badge.bad { background: rgba(239, 83, 80, 0.18); color: #ffb3b0; }
.badge.warn { background: rgba(255, 183, 77, 0.18); color: #ffd79f; }
.dialogue {
  margin-top: 16px;
  border: 1px solid var(--border);
  border-radius: 16px;
  overflow: hidden;
  background: linear-gradient(180deg, var(--panel), var(--panel-2));
}
.dialogue-header {
  padding: 14px 16px;
  display: flex;
  gap: 12px;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid var(--border);
}
.dialogue-header .title {
  font-weight: 700;
}
.dialogue-body {
  padding: 16px;
  display: grid;
  grid-template-columns: 1.5fr 1fr;
  gap: 16px;
}
@media (max-width: 1100px) {
  .dialogue-body { grid-template-columns: 1fr; }
}
.transcript, .gold-box, .background-box {
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px;
}
.turn { margin: 0 0 12px; white-space: pre-wrap; }
.turn:last-child { margin-bottom: 0; }
.turn .speaker { font-weight: 700; color: var(--accent); }
.turn.seeker .speaker { color: #c792ea; }
.muted { color: var(--muted); }
.small { font-size: 13px; }
.list { margin: 0; padding-left: 18px; }
.codeish { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.hidden { display: none !important; }
.footer { margin-top: 28px; color: var(--muted); font-size: 13px; }
"""


JS = """
const rows = Array.from(document.querySelectorAll('#metrics-body tr'));

function parseCellValue(value) {
  if (value === null || value === undefined) return '';
  const n = Number(value);
  return Number.isNaN(n) ? String(value).toLowerCase() : n;
}

function sortTable(colKey) {
  const body = document.getElementById('metrics-body');
  const currentKey = body.dataset.sortKey;
  const currentDir = body.dataset.sortDir || 'asc';
  const nextDir = currentKey === colKey && currentDir === 'asc' ? 'desc' : 'asc';
  rows.sort((a, b) => {
    const av = parseCellValue(a.dataset[colKey]);
    const bv = parseCellValue(b.dataset[colKey]);
    if (av < bv) return nextDir === 'asc' ? -1 : 1;
    if (av > bv) return nextDir === 'asc' ? 1 : -1;
    return 0;
  });
  rows.forEach((row) => body.appendChild(row));
  body.dataset.sortKey = colKey;
  body.dataset.sortDir = nextDir;
}

function applyFilters() {
  const query = document.getElementById('searchBox').value.trim().toLowerCase();
  const outcome = document.getElementById('outcomeFilter').value;
  const onlyNoGoodbye = document.getElementById('noGoodbyeFilter').checked;
  const onlyLowRecall = document.getElementById('lowRecallFilter').checked;
  const cards = Array.from(document.querySelectorAll('.dialogue'));

  rows.forEach((row) => {
    const text = row.dataset.search || '';
    const success = row.dataset.success;
    const goodbye = row.dataset.goodbye;
    const stepRecall = Number(row.dataset.steprecall || 'NaN');
    let show = true;
    if (query && !text.includes(query)) show = false;
    if (outcome === 'success' && success !== '1') show = false;
    if (outcome === 'failure' && success !== '0') show = false;
    if (onlyNoGoodbye && goodbye !== '0') show = false;
    if (onlyLowRecall && !(stepRecall < 1.0)) show = false;
    row.classList.toggle('hidden', !show);
    const anchor = row.dataset.anchor;
    const card = document.getElementById(anchor);
    if (card) card.classList.toggle('hidden', !show);
  });
}

document.querySelectorAll('th[data-sort-key]').forEach((th) => {
  th.addEventListener('click', () => sortTable(th.dataset.sortKey));
});

document.getElementById('searchBox').addEventListener('input', applyFilters);
document.getElementById('outcomeFilter').addEventListener('change', applyFilters);
document.getElementById('noGoodbyeFilter').addEventListener('change', applyFilters);
document.getElementById('lowRecallFilter').addEventListener('change', applyFilters);

sortTable('task');
applyFilters();
"""


def build_clarq_html_report(
    payload: dict[str, Any],
    output_path: str | Path,
    metrics_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    metrics_df = _load_optional_csv(metrics_path)
    summary_df = _load_optional_csv(summary_path)
    dialogues = _extract_dialogues(payload, metrics_df)
    fallback_summary = _build_fallback_summary(dialogues)
    meta = payload.get('meta', {})

    summary_row = {}
    if summary_df is not None and not summary_df.empty:
        summary_row = summary_df.iloc[0].to_dict()

    summary_cards = [
        ('Dialogues', summary_row.get('num_dialogues', fallback_summary['num_dialogues'])),
        ('Success rate', summary_row.get('success_rate', fallback_summary['success_rate'])),
        ('Goodbye rate', summary_row.get('Goodbye_rate', fallback_summary['Goodbye_rate'])),
        ('Step recall', summary_row.get('step_recall')),
        ('ClarQ count', summary_row.get('ClarQ_count')),
        ('ClarQ rate', summary_row.get('ClarQ_rate')),
        ('ClarQ depth', summary_row.get('ClarQ_depth')),
        ('ARL', summary_row.get('ARL', fallback_summary['avg_total_turns'])),
    ]

    meta_items = [
        ('Dataset', meta.get('task_data_path', '—')),
        ('Mode', meta.get('mode', '—')),
        ('Language', meta.get('language', '—')),
        ('Evaluation set', ', '.join(str(x) for x in meta.get('evaluation_set', [])) or '—'),
        ('Seeker model', meta.get('seeker_agent_llm', '—')),
        ('Provider model', meta.get('provider_agent_llm', '—')),
        ('Judge model', meta.get('judge_model', '—')),
    ]

    summary_html = ''.join(
        f"<div class='card'><h3>{html.escape(label)}</h3><div class='value'>{html.escape(_fmt_metric(value))}</div></div>"
        for label, value in summary_cards
    )
    meta_html = ''.join(
        f"<div class='card'><h3>{html.escape(label)}</h3><div class='small'>{html.escape(str(value))}</div></div>"
        for label, value in meta_items
    )

    table_rows = []
    dialogue_cards = []
    for d in dialogues:
        success = d.get('success')
        goodbye = d.get('Goodbye')
        success_badge = (
            "<span class='badge good'>yes</span>" if success == 1 else
            "<span class='badge bad'>no</span>" if success == 0 else
            "<span class='badge warn'>—</span>"
        )
        goodbye_badge = (
            "<span class='badge good'>yes</span>" if goodbye == 1 else
            "<span class='badge bad'>no</span>" if goodbye == 0 else
            "<span class='badge warn'>—</span>"
        )
        search_text = ' '.join(
            str(x) for x in [
                d['task_name'],
                d['task_type_index'],
                d['dialogue_index'],
                d['background'],
                ' '.join(d['transcript']),
            ]
        ).lower()
        table_rows.append(
            f"""
            <tr data-task="{d['task_type_index']}" data-dialogue="{d['dialogue_index']}" data-success="{'' if success is None else success}"
                data-steprecall="{'' if d.get('step_recall') is None else d.get('step_recall')}"
                data-goodbye="{'' if goodbye is None else goodbye}"
                data-search="{html.escape(search_text)}" data-anchor="{d['anchor']}">
              <td>{d['task_type_index']}</td>
              <td>{d['dialogue_index']}</td>
              <td>{html.escape(str(d['task_name']))}</td>
              <td>{success_badge}</td>
              <td>{html.escape(_fmt_metric(d.get('step_recall')))}</td>
              <td>{html.escape(_fmt_metric(d.get('ClarQ_count')))}</td>
              <td>{html.escape(_fmt_metric(d.get('ClarQ_rate')))}</td>
              <td>{html.escape(_fmt_metric(d.get('ClarQ_depth')))}</td>
              <td>{goodbye_badge}</td>
              <td>{html.escape(_fmt_metric(d.get('helper_turns')))}</td>
              <td>{html.escape(_fmt_metric(d.get('seeker_turns')))}</td>
              <td>{html.escape(_fmt_metric(d.get('total_turns')))}</td>
              <td><a href="#{d['anchor']}">open</a></td>
            </tr>
            """
        )

        transcript_parts = []
        for idx, turn in enumerate(d['transcript']):
            speaker = 'Jax' if idx % 2 == 0 else 'Seeker'
            speaker_class = 'helper' if idx % 2 == 0 else 'seeker'
            transcript_parts.append(
                f"<div class='turn {speaker_class}'><div class='speaker'>{speaker}</div><div>{html.escape(str(turn))}</div></div>"
            )
        gold_steps = ''.join(
            f"<li><span class='codeish'>{html.escape(str(gs))}</span>: {html.escape(str(expl))}</li>"
            for gs, expl in zip(d.get('gold_structure', []), d.get('gold_clarifications', []))
        )
        background_split = d.get('background_splitted', [])
        background_list = ''.join(f'<li>{html.escape(str(item))}</li>' for item in background_split)
        dialogue_cards.append(
            f"""
            <section class="dialogue" id="{d['anchor']}">
              <div class="dialogue-header">
                <div>
                  <div class="title">Task type {d['task_type_index']} · dialogue {d['dialogue_index']} · {html.escape(str(d['task_name']))}</div>
                  <div class="small muted">success: {html.escape(_fmt_metric(success))} · step_recall: {html.escape(_fmt_metric(d.get('step_recall')))} · goodbye: {html.escape(_fmt_metric(goodbye))}</div>
                </div>
                <div class="small"><a href="#top">back to top</a></div>
              </div>
              <div class="dialogue-body">
                <div>
                  <div class="transcript">
                    <h3>Transcript</h3>
                    {''.join(transcript_parts) or '<div class="muted">No transcript available.</div>'}
                  </div>
                  <div class="background-box" style="margin-top:16px;">
                    <h3>Task background</h3>
                    <div class="small" style="white-space: pre-wrap;">{html.escape(str(d.get('background', '')))}</div>
                  </div>
                </div>
                <div>
                  <div class="gold-box">
                    <h3>Gold clarification targets</h3>
                    <ol class="list small">{gold_steps or '<li>No gold clarifications available.</li>'}</ol>
                  </div>
                  <div class="gold-box" style="margin-top:16px;">
                    <h3>Gold full provider response</h3>
                    <div class="small" style="white-space: pre-wrap;">{html.escape(str(d.get('all_response', '')))}</div>
                  </div>
                  <div class="gold-box" style="margin-top:16px;">
                    <h3>Background slots</h3>
                    <ul class="list small">{background_list or '<li>No background slots available.</li>'}</ul>
                  </div>
                </div>
              </div>
            </section>
            """
        )

    html_doc = f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>ClarQ evaluation report</title>
        <style>{CSS}</style>
      </head>
      <body>
        <div class="container" id="top">
          <div class="header">
            <h1>ClarQ evaluation report</h1>
            <p>Generated from clarq_results.json and optional metrics/summary tables.</p>
          </div>

          <div class="meta-grid">{meta_html}</div>

          <h2 class="section-title">Summary</h2>
          <div class="summary-grid">{summary_html}</div>

          <h2 class="section-title">Per-dialogue overview</h2>
          <div class="controls">
            <input id="searchBox" type="search" placeholder="Search task, transcript, or slot..." />
            <select id="outcomeFilter">
              <option value="all">all outcomes</option>
              <option value="success">success only</option>
              <option value="failure">failure only</option>
            </select>
            <label><input id="noGoodbyeFilter" type="checkbox" /> no goodbye</label>
            <label><input id="lowRecallFilter" type="checkbox" /> step_recall &lt; 1</label>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th data-sort-key="task">task type</th>
                  <th data-sort-key="dialogue">dialogue</th>
                  <th>task</th>
                  <th data-sort-key="success">success</th>
                  <th data-sort-key="steprecall">step recall</th>
                  <th data-sort-key="clarqcount">ClarQ count</th>
                  <th data-sort-key="clarqrate">ClarQ rate</th>
                  <th data-sort-key="clarqdepth">ClarQ depth</th>
                  <th data-sort-key="goodbye">goodbye</th>
                  <th data-sort-key="helperturns">helper turns</th>
                  <th data-sort-key="seekerturns">seeker turns</th>
                  <th data-sort-key="totalturns">total turns</th>
                  <th>transcript</th>
                </tr>
              </thead>
              <tbody id="metrics-body">
                {''.join(table_rows)}
              </tbody>
            </table>
          </div>

          <h2 class="section-title">Transcripts</h2>
          {''.join(dialogue_cards)}

          <div class="footer">Report generated automatically during ClarQ evaluation.</div>
        </div>
        <script>
          document.querySelectorAll('#metrics-body tr').forEach((row) => {{
            const cells = row.children;
            row.dataset.clarqcount = cells[5].innerText.trim();
            row.dataset.clarqrate = cells[6].innerText.trim();
            row.dataset.clarqdepth = cells[7].innerText.trim();
            row.dataset.helperturns = cells[9].innerText.trim();
            row.dataset.seekerturns = cells[10].innerText.trim();
            row.dataset.totalturns = cells[11].innerText.trim();
          }});
          {JS}
        </script>
      </body>
    </html>
    """

    output_path.write_text(html_doc, encoding='utf-8')
    return output_path
