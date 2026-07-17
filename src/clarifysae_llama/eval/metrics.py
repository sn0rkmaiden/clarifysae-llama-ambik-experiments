from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from clarifysae_llama.eval.text_matching import best_match_score, nli_question_similarity


PREFERENCE_CATEGORY = 'preferences'



def normalize_questions(questions: Any) -> list[str]:
    if isinstance(questions, list):
        return [str(q).strip() for q in questions if str(q).strip()]
    if isinstance(questions, str):
        return [questions.strip()] if questions.strip() else []
    return []



def compute_example_metrics(
    *,
    ambiguity_type: str,
    gold_question: str,
    model_questions: list[str],
    predicted_ambiguous: bool | None,
    embed_threshold: float = 0.75,
    nli_threshold: float | None = None,
    enable_nli: bool = False,
) -> dict[str, Any]:
    if nli_threshold is None:
        nli_threshold = embed_threshold

    questions = normalize_questions(model_questions)
    gold_question = str(gold_question or '').strip()

    first_similarity = 0.0
    best_similarity = 0.0
    if questions:
        first_similarity = float(best_match_score(questions[0], gold_question, threshold=embed_threshold))
        for question in questions:
            score = float(best_match_score(question, gold_question, threshold=embed_threshold))
            if score > best_similarity:
                best_similarity = score

    if enable_nli:
        first_nli_similarity = 0.0
        best_nli_similarity = 0.0
        if questions:
            first_nli_similarity = float(nli_question_similarity(questions[0], gold_question))
            for question in questions:
                score = float(nli_question_similarity(question, gold_question))
                if score > best_nli_similarity:
                    best_nli_similarity = score
        resolved_nli_first = bool(first_nli_similarity >= float(nli_threshold))
        resolved_nli_any = bool(best_nli_similarity >= float(nli_threshold))
    else:
        first_nli_similarity = None
        best_nli_similarity = None
        resolved_nli_first = None
        resolved_nli_any = None

    gold_ambiguous = ambiguity_type != 'unambiguous_direct'
    ambiguity_decision_correct = None
    if predicted_ambiguous is not None:
        ambiguity_decision_correct = bool(predicted_ambiguous == gold_ambiguous)

    return {
        'ambiguity_type': ambiguity_type,
        'gold_ambiguous': gold_ambiguous,
        'predicted_ambiguous': predicted_ambiguous,
        'ambiguity_decision_correct': ambiguity_decision_correct,
        'model_questions': questions,
        'num_questions': len(questions),
        'asked_question': bool(len(questions) > 0),
        'model_question_first_similarity': first_similarity,
        'model_question_best_similarity': best_similarity,
        'resolved_proxy_first': bool(first_similarity >= float(embed_threshold)),
        'resolved_proxy_any': bool(best_similarity >= float(embed_threshold)),
        # Backward-compatible aliases.
        'resolved_proxy': bool(best_similarity >= float(embed_threshold)),
        'model_question_first_nli_similarity': first_nli_similarity,
        'model_question_best_nli_similarity': best_nli_similarity,
        'resolved_nli_first': resolved_nli_first,
        'resolved_nli_any': resolved_nli_any,
        'resolved_nli': resolved_nli_any,
    }



def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))



def _asked_question(row: dict[str, Any]) -> bool:
    if 'asked_question' in row:
        return bool(row['asked_question'])
    return int(row.get('num_questions', 0)) > 0



def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None



def aggregate_metrics(
    example_metrics: pd.DataFrame,
    *,
    embed_threshold: float = 0.75,
    brevity_max: int = 1,
    nli_threshold: float | None = None,
    enable_nli: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if nli_threshold is None:
        nli_threshold = embed_threshold

    if example_metrics.empty:
        overall = {
            'total_examples': 0,
            'embed_threshold': embed_threshold,
            'nli_threshold': nli_threshold,
            'enable_nli': enable_nli,
        }
        return pd.DataFrame([overall]), pd.DataFrame(columns=['ambiguity_type'])

    rows = example_metrics.to_dict(orient='records')
    counts = Counter(row['ambiguity_type'] for row in rows)
    question_hist = Counter(int(row['num_questions']) for row in rows)

    total_examples = len(rows)
    avg_num_questions = float(sum(int(row['num_questions']) for row in rows) / total_examples)
    brevity_score = float(sum(1 for row in rows if int(row['num_questions']) <= brevity_max) / total_examples)

    asked_rows = [row for row in rows if int(row['num_questions']) > 0]
    overall_first_similarity = _mean_or_none([
        float(row.get('model_question_first_similarity', 0.0)) for row in asked_rows
    ])
    overall_best_similarity = _mean_or_none([
        float(row.get('model_question_best_similarity', 0.0)) for row in asked_rows
    ])
    resolved_proxy_first_rate = float(sum(1 for row in rows if bool(row.get('resolved_proxy_first'))) / total_examples)
    resolved_proxy_any_rate = float(sum(1 for row in rows if bool(row.get('resolved_proxy_any', row.get('resolved_proxy')))) / total_examples)

    ambiguity_rows = [row for row in rows if row.get('ambiguity_decision_correct') is not None]
    ambiguity_decision_accuracy = _mean_or_none([1.0 if row['ambiguity_decision_correct'] else 0.0 for row in ambiguity_rows])

    gold_ambiguous_rows = [row for row in rows if bool(row.get('gold_ambiguous'))]
    gold_clear_rows = [row for row in rows if not bool(row.get('gold_ambiguous'))]
    predicted_ambiguous_rows = [row for row in rows if row.get('predicted_ambiguous') is True]
    predicted_clear_rows = [row for row in rows if row.get('predicted_ambiguous') is False]

    ambiguity_precision = (
        float(sum(1 for row in predicted_ambiguous_rows if bool(row.get('gold_ambiguous'))) / len(predicted_ambiguous_rows))
        if predicted_ambiguous_rows else None
    )
    ambiguity_recall = (
        float(sum(1 for row in gold_ambiguous_rows if row.get('predicted_ambiguous') is True) / len(gold_ambiguous_rows))
        if gold_ambiguous_rows else None
    )
    ambiguity_specificity = (
        float(sum(1 for row in gold_clear_rows if row.get('predicted_ambiguous') is False) / len(gold_clear_rows))
        if gold_clear_rows else None
    )

    overasking_rate_clear = (
        float(sum(1 for row in gold_clear_rows if _asked_question(row)) / len(gold_clear_rows))
        if gold_clear_rows else None
    )
    asking_rate_ambiguous = (
        float(sum(1 for row in gold_ambiguous_rows if _asked_question(row)) / len(gold_ambiguous_rows))
        if gold_ambiguous_rows else None
    )

    necessity_tp = sum(1 for row in rows if row['ambiguity_type'] == PREFERENCE_CATEGORY and _asked_question(row))
    necessity_fp = sum(1 for row in rows if row['ambiguity_type'] != PREFERENCE_CATEGORY and _asked_question(row))
    necessity_fn = sum(1 for row in rows if row['ambiguity_type'] == PREFERENCE_CATEGORY and not _asked_question(row))
    n_preferences = sum(1 for row in rows if row['ambiguity_type'] == PREFERENCE_CATEGORY)

    necessity_precision = (
        float(necessity_tp / (necessity_tp + necessity_fp))
        if (necessity_tp + necessity_fp) > 0 else None
    )
    necessity_recall = (
        float(necessity_tp / (necessity_tp + necessity_fn))
        if (necessity_tp + necessity_fn) > 0 else None
    )
    necessity_score = float(necessity_tp / n_preferences) if n_preferences > 0 else 0.0
    overall_weighted_score = 0.5 * necessity_score + 0.4 * float(overall_best_similarity or 0.0) + 0.1 * brevity_score

    json_exact_valid_rate = None
    if 'json_exact_valid' in example_metrics.columns:
        json_exact_valid_rate = float(sum(1 for row in rows if bool(row.get('json_exact_valid'))) / total_examples)
    json_schema_valid_rate = None
    if 'json_schema_valid' in example_metrics.columns:
        json_schema_valid_rate = float(sum(1 for row in rows if bool(row.get('json_schema_valid'))) / total_examples)
    json_recoverable_parse_rate = None
    if 'json_recoverable_parse' in example_metrics.columns:
        json_recoverable_parse_rate = float(sum(1 for row in rows if bool(row.get('json_recoverable_parse'))) / total_examples)

    overall: dict[str, Any] = {
        'total_examples': total_examples,
        'embed_threshold': embed_threshold,
        'nli_threshold': nli_threshold,
        'enable_nli': enable_nli,
        'avg_num_questions': avg_num_questions,
        'brevity_score': brevity_score,
        'overall_first_similarity_asked': overall_first_similarity,
        'overall_best_similarity_asked': overall_best_similarity,
        'overall_similarity_asked': overall_best_similarity,
        'resolved_proxy_first_rate': resolved_proxy_first_rate,
        'resolved_proxy_any_rate': resolved_proxy_any_rate,
        'resolved_proxy_rate': resolved_proxy_any_rate,
        'asking_rate_ambiguous': asking_rate_ambiguous,
        'overasking_rate_clear': overasking_rate_clear,
        'necessity_precision': necessity_precision,
        'necessity_recall': necessity_recall,
        'overall_weighted_score': overall_weighted_score,
        'ambiguity_decision_accuracy': ambiguity_decision_accuracy,
        'ambiguity_precision': ambiguity_precision,
        'ambiguity_recall': ambiguity_recall,
        'ambiguity_specificity': ambiguity_specificity,
    }

    if json_exact_valid_rate is not None:
        overall['json_exact_valid_rate'] = json_exact_valid_rate
    if json_schema_valid_rate is not None:
        overall['json_schema_valid_rate'] = json_schema_valid_rate
    if json_recoverable_parse_rate is not None:
        overall['json_recoverable_parse_rate'] = json_recoverable_parse_rate

    if enable_nli:
        overall_first_similarity_nli = _mean_or_none([
            float(row['model_question_first_nli_similarity'])
            for row in asked_rows
            if row.get('model_question_first_nli_similarity') is not None
        ])
        overall_best_similarity_nli = _mean_or_none([
            float(row['model_question_best_nli_similarity'])
            for row in asked_rows
            if row.get('model_question_best_nli_similarity') is not None
        ])
        resolved_nli_first_rate = float(sum(1 for row in rows if row.get('resolved_nli_first') is True) / total_examples)
        resolved_nli_any_rate = float(sum(1 for row in rows if row.get('resolved_nli_any') is True) / total_examples)
        overall_weighted_score_nli = 0.5 * necessity_score + 0.4 * float(overall_best_similarity_nli or 0.0) + 0.1 * brevity_score
        overall['overall_first_similarity_nli_asked'] = overall_first_similarity_nli
        overall['overall_best_similarity_nli_asked'] = overall_best_similarity_nli
        overall['overall_similarity_nli_asked'] = overall_best_similarity_nli
        overall['resolved_nli_first_rate'] = resolved_nli_first_rate
        overall['resolved_nli_any_rate'] = resolved_nli_any_rate
        overall['resolved_nli_rate'] = resolved_nli_any_rate
        overall['overall_weighted_score_nli'] = overall_weighted_score_nli

    for category, count in sorted(counts.items()):
        overall[f'count__{category}'] = int(count)

    for n_questions, count in sorted(question_hist.items()):
        overall[f'num_questions_hist__{n_questions}'] = int(count)

    category_rows: list[dict[str, Any]] = []
    for category in sorted(counts.keys()):
        cat_rows = [row for row in rows if row['ambiguity_type'] == category]
        cat_asked = [row for row in cat_rows if int(row['num_questions']) > 0]
        asked_rate = float(sum(1 for row in cat_rows if _asked_question(row)) / len(cat_rows))
        category_row: dict[str, Any] = {
            'ambiguity_type': category,
            'n_examples': len(cat_rows),
            'asked_rate': asked_rate,
            'avg_num_questions': float(sum(int(row['num_questions']) for row in cat_rows) / len(cat_rows)),
            'mean_first_similarity_over_asked': _mean_or_none([
                float(row.get('model_question_first_similarity', 0.0)) for row in cat_asked
            ]),
            'mean_best_similarity_over_asked': _mean_or_none([
                float(row.get('model_question_best_similarity', 0.0)) for row in cat_asked
            ]),
            'resolved_proxy_first_rate': float(sum(1 for row in cat_rows if bool(row.get('resolved_proxy_first'))) / len(cat_rows)),
            'resolved_proxy_any_rate': float(sum(1 for row in cat_rows if bool(row.get('resolved_proxy_any', row.get('resolved_proxy')))) / len(cat_rows)),
            'resolved_proxy_rate': float(sum(1 for row in cat_rows if bool(row.get('resolved_proxy_any', row.get('resolved_proxy')))) / len(cat_rows)),
            'ambiguity_decision_accuracy': _mean_or_none([
                1.0 if row['ambiguity_decision_correct'] else 0.0
                for row in cat_rows
                if row.get('ambiguity_decision_correct') is not None
            ]),
        }
        if 'json_exact_valid' in example_metrics.columns:
            category_row['json_exact_valid_rate'] = float(sum(1 for row in cat_rows if bool(row.get('json_exact_valid'))) / len(cat_rows))
        if 'json_schema_valid' in example_metrics.columns:
            category_row['json_schema_valid_rate'] = float(sum(1 for row in cat_rows if bool(row.get('json_schema_valid'))) / len(cat_rows))
        if 'json_recoverable_parse' in example_metrics.columns:
            category_row['json_recoverable_parse_rate'] = float(sum(1 for row in cat_rows if bool(row.get('json_recoverable_parse'))) / len(cat_rows))

        if enable_nli:
            category_row['mean_first_similarity_nli_over_asked'] = _mean_or_none([
                float(row['model_question_first_nli_similarity'])
                for row in cat_asked
                if row.get('model_question_first_nli_similarity') is not None
            ])
            category_row['mean_best_similarity_nli_over_asked'] = _mean_or_none([
                float(row['model_question_best_nli_similarity'])
                for row in cat_asked
                if row.get('model_question_best_nli_similarity') is not None
            ])
            category_row['resolved_nli_first_rate'] = float(
                sum(1 for row in cat_rows if row.get('resolved_nli_first') is True) / len(cat_rows)
            )
            category_row['resolved_nli_any_rate'] = float(
                sum(1 for row in cat_rows if row.get('resolved_nli_any') is True) / len(cat_rows)
            )
            category_row['resolved_nli_rate'] = category_row['resolved_nli_any_rate']
        category_rows.append(category_row)

        overall[f'asked_rate__{category}'] = category_row['asked_rate']
        overall[f'mean_first_similarity_over_asked__{category}'] = category_row['mean_first_similarity_over_asked']
        overall[f'mean_best_similarity_over_asked__{category}'] = category_row['mean_best_similarity_over_asked']
        overall[f'resolved_proxy_first_rate__{category}'] = category_row['resolved_proxy_first_rate']
        overall[f'resolved_proxy_any_rate__{category}'] = category_row['resolved_proxy_any_rate']
        overall[f'resolved_proxy_rate__{category}'] = category_row['resolved_proxy_rate']
        overall[f'ambiguity_decision_accuracy__{category}'] = category_row['ambiguity_decision_accuracy']
        if 'json_exact_valid_rate' in category_row:
            overall[f'json_exact_valid_rate__{category}'] = category_row['json_exact_valid_rate']
        if 'json_schema_valid_rate' in category_row:
            overall[f'json_schema_valid_rate__{category}'] = category_row['json_schema_valid_rate']
        if 'json_recoverable_parse_rate' in category_row:
            overall[f'json_recoverable_parse_rate__{category}'] = category_row['json_recoverable_parse_rate']
        if enable_nli:
            overall[f'mean_first_similarity_nli_over_asked__{category}'] = category_row.get('mean_first_similarity_nli_over_asked')
            overall[f'mean_best_similarity_nli_over_asked__{category}'] = category_row.get('mean_best_similarity_nli_over_asked')
            overall[f'resolved_nli_first_rate__{category}'] = category_row.get('resolved_nli_first_rate')
            overall[f'resolved_nli_any_rate__{category}'] = category_row.get('resolved_nli_any_rate')
            overall[f'resolved_nli_rate__{category}'] = category_row.get('resolved_nli_rate')

    return pd.DataFrame([overall]), pd.DataFrame(category_rows)
