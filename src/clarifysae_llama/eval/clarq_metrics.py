from __future__ import annotations

import json
from typing import Any

import pandas as pd

from clarifysae_llama.clarq_legacy.utils import detect_language


def add_punctuation(sentence: str, chinese: bool = False) -> str:
    if sentence and sentence[-1] not in ['.', '?', '!', ',', '。', '？', '！', '，']:
        return sentence + ('。' if chinese else '.')
    return sentence


def data2prompt_mini(gold: list[str], gold_explain: list[str], predict: list[str]) -> str:
    if not gold:
        return "Gold information is empty. Return match: true in JSON."

    if detect_language(gold[0]) == "Chinese":
        g_name = '正确信息及其信息用途'
        p_name = '生成信息'
        start = (
            "下面展示了两段文字。第一段是：{0}，第二段是：{1}。第一段的信息用途可以帮助你理解正确信息的用途。"
            "{1}缺少这些用途的说明，你需要自行分析{1}的用途，并判断其是否包含了正确信息的用途。"
            "\n判断方法是先分析{1}的用途，然后检查是否可以在{1}中找到与正确信息用途一致的内容。"
        ).format(g_name, p_name)

        numbered_gold = [
            f"{i+1}. {add_punctuation(line, True)} 信息用途：{add_punctuation(line_ex, True)}"
            for (i, line), line_ex in zip(enumerate(gold), gold_explain)
        ]
        numbered_predict = [f"{i+1}. {add_punctuation(line, True)}" for i, line in enumerate(predict)]
        middle = f"{g_name}：\n" + "\n".join(numbered_gold) + f"\n\n{p_name}：\n" + "\n".join(numbered_predict)
        end = "仔细判断第一段是否被第二段提及。返回格式为：{ 'analysis': '...', 'match': true/false }"
        return "\n\n".join([start, middle, end])

    g_name = 'Gold Information and its Purpose'
    p_name = 'Generated Information'
    start = (
        "Below are two passages of text. First: {0}, second: {1}. Analyze whether the second contains "
        "the purpose of the gold information."
    ).format(g_name, p_name)
    numbered_gold = [
        f"{i+1}. {add_punctuation(line)}  Purpose: {add_punctuation(line_ex)}"
        for (i, line), line_ex in zip(enumerate(gold), gold_explain)
    ]
    numbered_predict = [f"{i+1}. {add_punctuation(line)}" for i, line in enumerate(predict)]
    middle = f"{g_name}:\n" + "\n".join(numbered_gold) + f"\n\n{p_name}:\n" + "\n".join(numbered_predict)
    end = "Return JSON with fields { 'analysis': '...', 'match': true/false }."
    return "\n\n".join([start, middle, end])


def evaluate_one_multi(gold: list[str], gold_explain: list[str], predict: list[str], judge_llm) -> int:
    gold = [s[4:].strip() if s.lower().startswith("jax:") else s.strip() for s in gold[1:]]
    predict = [s[4:].strip() if s.lower().startswith("jax:") else s.strip() for s in predict]

    if not gold:
        return 0

    if detect_language(gold[0]) != "Chinese":
        gold_norm = [g.lower() for g in gold]
        pred_norm = [p.lower() for p in predict]
    else:
        gold_norm = gold[:]
        pred_norm = predict[:]

    if gold_norm == pred_norm:
        return 1
    if all(g in pred_norm for g in gold_norm):
        return 1

    gold_clean = [g.rstrip('，。？！,.?!') for g in gold_norm]
    pred_clean = [p.rstrip('，。？！,.?!') for p in pred_norm]

    contained = set()
    explain_map = {g: gold_explain[i] for i, g in enumerate(gold_clean)}

    for p in pred_clean:
        for g in gold_clean:
            if g in p:
                contained.add(g)

    gold_diff = [g for g in gold_clean if g not in contained]
    pred_diff = [p for p in pred_clean if p not in contained]

    if not gold_diff:
        return 1
    if not pred_diff:
        return 0
    if judge_llm is None:
        return 0

    for gd in gold_diff:
        gde = explain_map[gd]
        prompt = data2prompt_mini([gd], [gde], pred_diff)
        response, _ = judge_llm.request(prompt, None, previous_message=None, json_format=True)
        try:
            parsed = json.loads(response)
        except Exception:
            return 0
        if not parsed.get('match', False):
            return 0
    return 1


def parse_evaluation_set(raw: str) -> list[int]:
    raw = raw.strip()
    if "," in raw:
        return [int(x) for x in raw.split(",") if x]
    if "-" in raw:
        a, b = raw.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(raw)]


def _normalize_jax_lines(lines: list[str]) -> list[str]:
    cleaned = [s[4:].strip() if s.lower().startswith("jax:") else s.strip() for s in lines]
    if cleaned and detect_language(cleaned[0]) != "Chinese":
        cleaned = [x.lower() for x in cleaned]
    return [x.rstrip('，。？！,.?!') for x in cleaned]


def _extract_dialogue_turns(h2l: list[str]) -> tuple[list[str], list[str]]:
    """
    Returns:
      helper: provider/Jax responses after the initial greeting
      seeker: seeker utterances
    """
    helper: list[str] = []
    seeker: list[str] = []

    for k, sent in enumerate(h2l[1:]):
        if k % 2 == 1 and k != 1:
            helper.append(sent)
        elif k % 2 == 0:
            seeker.append(sent.strip())

    return helper, seeker


def _select_dialogue(conv: dict[str, Any]) -> tuple[int, list[str]]:
    """
    Prefer the first non-empty l2l slot. If all are empty, return slot 0 and [].
    """
    l2l = conv.get('l2l', []) or []
    for idx, h2l in enumerate(l2l):
        if h2l:
            return idx, h2l
    return 0, []


def _compute_dialogue_row(
    *,
    meta: dict[str, Any],
    task_type_index: int,
    dialogue_index: int,
    dialogue_slot: int,
    conv: dict[str, Any],
    h2l: list[str],
    judge_llm,
) -> dict[str, Any]:
    gold_r = conv['all_response'].split("\n")
    gold_helper_lines = gold_r[1:]
    gold_explain = conv.get('all_response_exaplain', [])
    gold_structure = conv.get('gold_structure', [])

    helper, seeker = _extract_dialogue_turns(h2l)

    gold_clean = _normalize_jax_lines(gold_helper_lines)
    pred_clean = _normalize_jax_lines(helper)

    if helper:
        strict = evaluate_one_multi(gold_r, gold_explain, helper, judge_llm)
    else:
        strict = 0

    aqd = (len(helper) + 1 - len(gold_r)) if gold_r else 0.0

    if seeker:
        if detect_language(seeker[0]) != 'Chinese':
            arl = sum(s.count(' ') for s in seeker) / len(seeker)
        else:
            arl = sum(len(s) for s in seeker) / len(seeker)
    else:
        arl = 0.0

    covered = sum(any(g in p for p in pred_clean) for g in gold_clean) if gold_clean else 0
    step_recall = (covered / len(gold_clean)) if gold_clean else 0.0

    num_turns = len(seeker)
    num_q = sum('?' in s for s in seeker)
    clarq_rate = (num_q / num_turns) if num_turns > 0 else 0.0

    last_q = -1
    for idx, s in enumerate(seeker):
        if '?' in s:
            last_q = idx
    clarq_depth = (last_q + 1) if last_q >= 0 else 0

    goodbye = 1 if (seeker and 'goodbye' in seeker[-1].lower()) else 0

    q_lengths = [s.count(' ') + 1 for s in seeker if '?' in s]
    clarq_len = (sum(q_lengths) / len(q_lengths)) if q_lengths else 0.0

    return {
        # run metadata
        'seeker_agent_llm': meta.get('seeker_agent_llm'),
        'provider_agent_llm': meta.get('provider_agent_llm'),
        'judge_model': meta.get('judge_model'),
        'mode': meta.get('mode'),
        'language': meta.get('language'),
        'evaluation_set': meta.get('evaluation_set_arg') or meta.get('evaluation_set'),
        'steering_feature': (meta.get('steering') or {}).get('feature'),
        'steering_features': (meta.get('steering') or {}).get('feature_indices'),
        'steering_feature_set_label': (meta.get('steering') or {}).get('feature_set_label'),
        'steering_feature_count': (meta.get('steering') or {}).get('feature_count'),
        'steering_strength': (meta.get('steering') or {}).get('strength'),

        # dialogue identifiers
        'task_type_index': task_type_index,
        'dialogue_index': dialogue_index,
        'dialogue_slot': dialogue_slot,

        # debugging / bookkeeping
        'gold_steps': len(gold_clean),
        'helper_turns': len(helper),
        'seeker_turns': len(seeker),
        'dialogue_present': 1 if h2l else 0,

        # main metrics
        'success': strict,
        'AQD': aqd,
        'ARL': arl,
        'step_recall': step_recall,
        'ClarQ_count': num_q,
        'ClarQ_rate': clarq_rate,
        'ClarQ_depth': clarq_depth,
        'Goodbye': goodbye,
        'ClarQ_len': clarq_len,
    }


def compute_metrics_for_payload(
    payload: dict[str, Any] | list[Any],
    judge_llm,
    evaluation_set: list[int],
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    all_conv = payload
    if isinstance(payload, dict) and 'data' in payload:
        meta = payload.get('meta', {}) or {}
        all_conv = payload['data']

    rows: list[dict[str, Any]] = []

    for task_type_index, one_type in enumerate(all_conv):
        if task_type_index not in evaluation_set:
            continue

        for dialogue_index, conv in enumerate(one_type):
            dialogue_slot, h2l = _select_dialogue(conv)
            row = _compute_dialogue_row(
                meta=meta,
                task_type_index=task_type_index,
                dialogue_index=dialogue_index,
                dialogue_slot=dialogue_slot,
                conv=conv,
                h2l=h2l,
                judge_llm=judge_llm,
            )
            rows.append(row)

    denom = len(rows) if rows else 1

    summary = {
        'seeker_agent_llm': meta.get('seeker_agent_llm'),
        'provider_agent_llm': meta.get('provider_agent_llm'),
        'judge_model': meta.get('judge_model'),
        'mode': meta.get('mode'),
        'language': meta.get('language'),
        'evaluation_set': meta.get('evaluation_set_arg') or meta.get('evaluation_set'),
        'steering_feature': (meta.get('steering') or {}).get('feature'),
        'steering_features': (meta.get('steering') or {}).get('feature_indices'),
        'steering_feature_set_label': (meta.get('steering') or {}).get('feature_set_label'),
        'steering_feature_count': (meta.get('steering') or {}).get('feature_count'),
        'steering_strength': (meta.get('steering') or {}).get('strength'),
        'num_dialogues': denom,
        'success_rate': sum(r['success'] for r in rows) / denom,
        'AQD': sum(r['AQD'] for r in rows) / denom,
        'ARL': sum(r['ARL'] for r in rows) / denom,
        'step_recall': sum(r['step_recall'] for r in rows) / denom,
        'ClarQ_count': sum(r['ClarQ_count'] for r in rows) / denom,
        'ClarQ_rate': sum(r['ClarQ_rate'] for r in rows) / denom,
        'ClarQ_depth': sum(r['ClarQ_depth'] for r in rows) / denom,
        'Goodbye_rate': sum(r['Goodbye'] for r in rows) / denom,
        'ClarQ_len': sum(r['ClarQ_len'] for r in rows) / denom,
    }

    return {
        'rows': rows,
        'summary': summary,
    }


def metrics_to_dataframes(metrics: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = metrics.get('rows', [])
    summary = metrics.get('summary', {})

    metrics_df = pd.DataFrame(rows)

    summary_cols = [
        'success_rate',
        'AQD',
        'ARL',
        'step_recall',
        'ClarQ_count',
        'ClarQ_rate',
        'ClarQ_depth',
        'Goodbye_rate',
        'ClarQ_len',
        'num_dialogues',
        'steering_feature',
        'steering_features',
        'steering_feature_set_label',
        'steering_feature_count',
        'steering_strength',
        'language',
        'evaluation_set',
    ]
    summary_df = pd.DataFrame([summary])
    summary_df = summary_df[[c for c in summary_cols if c in summary_df.columns]].copy()

    return metrics_df, summary_df