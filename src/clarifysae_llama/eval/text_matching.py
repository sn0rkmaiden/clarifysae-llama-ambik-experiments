from __future__ import annotations

import re
from functools import lru_cache


def normalize_text(text: str | None) -> str:
    if text is None:
        return ''
    return re.sub(r'\s+', ' ', str(text).strip().lower())


def exact_contains_match(a: str | None, b: str | None) -> tuple[bool, float]:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return False, 0.0
    if a_norm == b_norm:
        return True, 1.0
    if a_norm in b_norm or b_norm in a_norm:
        return True, 0.95
    return False, 0.0


@lru_cache(maxsize=1)
def _get_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception:
        return None


def embedding_similarity(a: str | None, b: str | None) -> float:
    model = _get_sentence_transformer()
    if model is None:
        return 0.0

    a_text = (a or '').strip()
    b_text = (b or '').strip()
    if not a_text or not b_text:
        return 0.0

    try:
        embeddings = model.encode([a_text, b_text], normalize_embeddings=True)
        return float((embeddings[0] * embeddings[1]).sum())
    except Exception:
        return 0.0


def best_match_score(model_out: str | None, gold: str | None, threshold: float = 0.75, return_pass: bool = False):
    matched, lexical_score = exact_contains_match(model_out, gold)
    if matched:
        return (lexical_score, lexical_score >= threshold) if return_pass else lexical_score

    sim = embedding_similarity(model_out, gold)
    return (sim, sim >= threshold) if return_pass else sim


@lru_cache(maxsize=1)
def _get_nli_components():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception:
        return None

    model_name = 'roberta-large-mnli'
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        label2id = model.config.label2id
        entail_id = label2id.get('ENTAILMENT') or label2id.get('entailment') or 2
        return tokenizer, model, device, entail_id
    except Exception:
        return None


def nli_question_similarity(a: str | None, b: str | None) -> float:
    matched, lexical_score = exact_contains_match(a, b)
    if matched:
        return lexical_score

    components = _get_nli_components()
    if components is None:
        return 0.0

    tokenizer, model, device, entail_id = components
    premise = (a or '').strip()
    hypothesis = (b or '').strip()
    if not premise or not hypothesis:
        return 0.0

    try:
        import torch
        encoded = tokenizer(
            premise,
            hypothesis,
            return_tensors='pt',
            truncation=True,
            max_length=256,
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = logits.softmax(dim=-1).squeeze(0)
        p1 = float(probs[entail_id].item())

        encoded_rev = tokenizer(
            hypothesis,
            premise,
            return_tensors='pt',
            truncation=True,
            max_length=256,
        ).to(device)
        with torch.no_grad():
            logits_rev = model(**encoded_rev).logits
            probs_rev = logits_rev.softmax(dim=-1).squeeze(0)
        p2 = float(probs_rev[entail_id].item())
        return 0.5 * (p1 + p2)
    except Exception:
        return 0.0
