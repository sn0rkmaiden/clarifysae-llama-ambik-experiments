from __future__ import annotations

import re
from typing import Any

import pandas as pd


QUESTION_PATTERN = re.compile(
    r"\b(what|which|where|when|why|who|how|should|can|could|would|will|"
    r"do|does|did|is|are|am|was|were|may)\b[^?]*\?",
    flags=re.IGNORECASE | re.DOTALL,
)


def binary_auroc(labels: list[bool], scores: list[float]) -> float | None:
    """Compute AUROC from ranks without adding a scikit-learn dependency."""
    if len(labels) != len(scores):
        raise ValueError('labels and scores must have the same length')
    n_pos = sum(bool(label) for label in labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    order = sorted(range(len(scores)), key=lambda idx: scores[idx])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        for pos in range(start, end):
            ranks[order[pos]] = avg_rank
        start = end

    sum_pos_ranks = sum(rank for rank, label in zip(ranks, labels) if label)
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _classification_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    tp = sum(1 for row in rows if bool(row['gold_ambiguous']) and bool(row['predicted_ambiguous']))
    tn = sum(1 for row in rows if not bool(row['gold_ambiguous']) and not bool(row['predicted_ambiguous']))
    fp = sum(1 for row in rows if not bool(row['gold_ambiguous']) and bool(row['predicted_ambiguous']))
    fn = sum(1 for row in rows if bool(row['gold_ambiguous']) and not bool(row['predicted_ambiguous']))
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def classification_summary(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Compute binary metrics on already filtered classification rows."""
    if not rows:
        return {
            'n_examples': 0,
            'n_ambiguous': 0,
            'n_clear': 0,
            'accuracy': None,
            'balanced_accuracy': None,
            'precision': None,
            'recall': None,
            'specificity': None,
            'auroc': None,
        }

    counts = _classification_counts(rows)
    tp, tn, fp, fn = counts['tp'], counts['tn'], counts['fp'], counts['fn']
    n_pos = tp + fn
    n_neg = tn + fp
    recall = tp / n_pos if n_pos else None
    specificity = tn / n_neg if n_neg else None
    precision = tp / (tp + fp) if (tp + fp) else None
    balanced_accuracy = (
        (float(recall) + float(specificity)) / 2.0
        if recall is not None and specificity is not None
        else None
    )

    return {
        'n_examples': len(rows),
        'n_ambiguous': n_pos,
        'n_clear': n_neg,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': (tp + tn) / len(rows),
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'auroc': binary_auroc(
            [bool(row['gold_ambiguous']) for row in rows],
            [float(row['ambiguity_probability']) for row in rows],
        ),
    }


def select_balanced_accuracy_threshold(
    labels: list[bool],
    scores: list[float],
) -> tuple[float, dict[str, float | int | None]]:
    """Choose a threshold that maximizes balanced accuracy.

    Candidate thresholds are the score midpoints plus values immediately below
    the minimum and above the maximum. Ties prefer higher accuracy, a closer
    balance between recall and specificity, and finally the more conservative
    (higher) threshold.
    """
    if len(labels) != len(scores) or not labels:
        raise ValueError('labels and scores must be non-empty and have equal length')
    if len(set(bool(label) for label in labels)) < 2:
        raise ValueError('Threshold calibration requires both ambiguous and clear examples')

    unique_scores = sorted(set(float(score) for score in scores))
    eps = 1e-12
    candidates = [unique_scores[0] - eps]
    candidates.extend(
        (left + right) / 2.0
        for left, right in zip(unique_scores, unique_scores[1:])
    )
    candidates.append(unique_scores[-1] + eps)

    best_threshold: float | None = None
    best_summary: dict[str, float | int | None] | None = None
    best_key: tuple[float, float, float, float] | None = None

    for threshold in candidates:
        rows = [
            {
                'gold_ambiguous': bool(label),
                'predicted_ambiguous': bool(score >= threshold),
                'ambiguity_probability': float(score),
            }
            for label, score in zip(labels, scores)
        ]
        summary = classification_summary(rows)
        balanced = float(summary['balanced_accuracy'] or 0.0)
        accuracy = float(summary['accuracy'] or 0.0)
        recall = float(summary['recall'] or 0.0)
        specificity = float(summary['specificity'] or 0.0)
        key = (balanced, accuracy, -abs(recall - specificity), threshold)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_summary = summary

    assert best_threshold is not None and best_summary is not None
    best_summary = dict(best_summary)
    best_summary['threshold'] = best_threshold
    return best_threshold, best_summary


def add_clam_aggregate_metrics(
    aggregate_df: pd.DataFrame,
    example_metrics: pd.DataFrame,
) -> pd.DataFrame:
    if aggregate_df.empty or example_metrics.empty:
        return aggregate_df

    rows = example_metrics.to_dict(orient='records')
    ambiguous_rows = [row for row in rows if bool(row['gold_ambiguous'])]
    clear_rows = [row for row in rows if not bool(row['gold_ambiguous'])]
    eligible_rows = [row for row in rows if bool(row.get('classification_eligible', True))]
    eligible_ambiguous = [row for row in eligible_rows if bool(row['gold_ambiguous'])]
    eligible_clear = [row for row in eligible_rows if not bool(row['gold_ambiguous'])]

    classification = classification_summary(eligible_rows)
    selective_success_values: list[float] = []
    for row in eligible_rows:
        if bool(row['gold_ambiguous']):
            selective_success_values.append(1.0 if bool(row['resolved_proxy_any']) else 0.0)
        else:
            selective_success_values.append(1.0 if not bool(row['asked_question']) else 0.0)

    result = aggregate_df.copy()

    # Replace the generic classifier metrics, which would otherwise include
    # contradictory AmbiK pairs whose ambiguous and clear texts are identical.
    result.loc[0, 'ambiguity_decision_accuracy'] = classification['accuracy']
    result.loc[0, 'ambiguity_precision'] = classification['precision']
    result.loc[0, 'ambiguity_recall'] = classification['recall']
    result.loc[0, 'ambiguity_specificity'] = classification['specificity']
    result.loc[0, 'classification_balanced_accuracy'] = classification['balanced_accuracy']
    result.loc[0, 'classification_auroc'] = classification['auroc']
    result.loc[0, 'classification_n_examples'] = classification['n_examples']
    result.loc[0, 'classification_n_ambiguous'] = classification['n_ambiguous']
    result.loc[0, 'classification_n_clear'] = classification['n_clear']
    result.loc[0, 'classification_excluded_examples'] = len(rows) - len(eligible_rows)
    result.loc[0, 'classification_excluded_source_pairs'] = len({
        str(row['source_id'])
        for row in rows
        if not bool(row.get('classification_eligible', True))
    })

    result.loc[0, 'asking_rate_ambiguous'] = (
        sum(1 for row in eligible_ambiguous if bool(row['asked_question'])) / len(eligible_ambiguous)
        if eligible_ambiguous else None
    )
    result.loc[0, 'overasking_rate_clear'] = (
        sum(1 for row in eligible_clear if bool(row['asked_question'])) / len(eligible_clear)
        if eligible_clear else None
    )
    result.loc[0, 'clam_selective_success'] = (
        sum(selective_success_values) / len(selective_success_values)
        if selective_success_values else None
    )

    # End-to-end and oracle-gated question metrics are computed on all 100 gold
    # ambiguous examples, including rows excluded only from the paired
    # classification analysis.
    result.loc[0, 'resolved_proxy_rate_ambiguous'] = (
        sum(1 for row in ambiguous_rows if bool(row['resolved_proxy_any'])) / len(ambiguous_rows)
        if ambiguous_rows else None
    )
    result.loc[0, 'resolved_proxy_rate_ambiguous_classification_eligible'] = (
        sum(1 for row in eligible_ambiguous if bool(row['resolved_proxy_any'])) / len(eligible_ambiguous)
        if eligible_ambiguous else None
    )
    result.loc[0, 'question_similarity_asked_ambiguous'] = (
        sum(float(row['model_question_best_similarity']) for row in ambiguous_rows if bool(row['asked_question']))
        / sum(1 for row in ambiguous_rows if bool(row['asked_question']))
        if any(bool(row['asked_question']) for row in ambiguous_rows) else None
    )
    result.loc[0, 'oracle_gate_resolved_proxy_rate_ambiguous'] = (
        sum(1 for row in ambiguous_rows if bool(row.get('oracle_gate_resolved_proxy_any'))) / len(ambiguous_rows)
        if ambiguous_rows else None
    )
    result.loc[0, 'oracle_gate_question_similarity_ambiguous'] = (
        sum(float(row.get('oracle_gate_question_similarity', 0.0)) for row in ambiguous_rows) / len(ambiguous_rows)
        if ambiguous_rows else None
    )
    result.loc[0, 'n_ambiguous'] = len(ambiguous_rows)
    result.loc[0, 'n_clear'] = len(clear_rows)
    return result


def clean_single_question(raw_output: Any) -> str:
    """Extract the first actual question from a model generation.

    Small instruction-tuned models sometimes emit a preamble such as
    ``Sure, here is the clarification question:`` on the first line. The old
    parser returned that preamble and discarded the real question below it.
    """
    text = str(raw_output or '').strip()
    if not text:
        return ''

    text = text.replace('```text', '').replace('```', '')
    text = text.replace('**', '').replace('__', '')

    marker_match = re.search(
        r'clarification\s+question\s*:',
        text,
        flags=re.IGNORECASE,
    )
    search_text = text[marker_match.end():] if marker_match else text

    match = QUESTION_PATTERN.search(search_text)
    if match is None and search_text != text:
        match = QUESTION_PATTERN.search(text)
    if match is not None:
        return ' '.join(match.group(0).split()).strip(' `"\'')

    for line in text.splitlines():
        cleaned = line.strip().lstrip('-*•0123456789.): ')
        if '?' in cleaned:
            return cleaned[: cleaned.index('?') + 1].strip(' `"\'')

    return ''
