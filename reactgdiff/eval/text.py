"""Text metrics for decoded procedure strings."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from reactgdiff.eval.lev import levenshtein_similarity, normalize_text

TOKEN_RE = re.compile(r"\$-?\d+\$|@\d+@|#\d+#|\d+(?:\.\d+)?|[A-Za-z]+|%")
NUMERIC_FRAGMENT_RE = re.compile(
    r"@\d+@|#\d+#|(?<![$@#])[-+]?\d+(?:[.,]\d+)?\s*(?:°\s*)?"
    r"(?:ug|µg|mg|g|kg|ul|µl|ml|l|liter|liters|litre|litres|umol|µmol|"
    r"mmol|mol|molar|normal|equiv|equivalent|eq|percent|%|h|hr|hrs|"
    r"hour|hours|min|minute|minutes|s|sec|second|seconds|day|days|"
    r"c|°c|℃|degc|degree|degrees)?(?=\b|[^A-Za-z0-9_])",
    re.IGNORECASE,
)


def corpus_text_metrics(pairs: Iterable[tuple[str, str]]) -> dict[str, float]:
    pairs = list(pairs)
    if not pairs:
        return {
            "count": 0.0,
            "bleu_2": 0.0,
            "rouge_1": 0.0,
            "rouge_1_precision": 0.0,
            "rouge_1_recall": 0.0,
            "levenshtein_90_rate": 0.0,
            "levenshtein_75_rate": 0.0,
            "levenshtein_50_rate": 0.0,
            "number_normalized_levenshtein_90_rate": 0.0,
            "number_normalized_levenshtein_75_rate": 0.0,
            "number_normalized_levenshtein_50_rate": 0.0,
            "mean_number_normalized_levenshtein_similarity": 0.0,
            "exact_match_rate": 0.0,
        }

    pred_tokens = [_tokens(prediction) for prediction, _ in pairs]
    ref_tokens = [_tokens(reference) for _, reference in pairs]
    similarities = [levenshtein_similarity(prediction, reference) for prediction, reference in pairs]
    number_normalized_pairs = [
        (number_normalize_text(prediction), number_normalize_text(reference))
        for prediction, reference in pairs
    ]
    number_normalized_similarities = [
        levenshtein_similarity(prediction, reference)
        for prediction, reference in number_normalized_pairs
    ]
    rouge_scores = [_rouge_1(prediction, reference) for prediction, reference in zip(pred_tokens, ref_tokens)]
    exact_matches = sum(
        normalize_text(prediction) == normalize_text(reference)
        for prediction, reference in pairs
    )

    return {
        "count": float(len(pairs)),
        "bleu_2": _corpus_bleu_2(pred_tokens, ref_tokens),
        "rouge_1": sum(score[2] for score in rouge_scores) / len(rouge_scores),
        "rouge_1_precision": sum(score[0] for score in rouge_scores) / len(rouge_scores),
        "rouge_1_recall": sum(score[1] for score in rouge_scores) / len(rouge_scores),
        "levenshtein_90_rate": sum(sim >= 0.90 for sim in similarities) / len(similarities),
        "levenshtein_75_rate": sum(sim >= 0.75 for sim in similarities) / len(similarities),
        "levenshtein_50_rate": sum(sim >= 0.50 for sim in similarities) / len(similarities),
        "mean_number_normalized_levenshtein_similarity": (
            sum(number_normalized_similarities) / len(number_normalized_similarities)
        ),
        "number_normalized_levenshtein_90_rate": (
            sum(sim >= 0.90 for sim in number_normalized_similarities)
            / len(number_normalized_similarities)
        ),
        "number_normalized_levenshtein_75_rate": (
            sum(sim >= 0.75 for sim in number_normalized_similarities)
            / len(number_normalized_similarities)
        ),
        "number_normalized_levenshtein_50_rate": (
            sum(sim >= 0.50 for sim in number_normalized_similarities)
            / len(number_normalized_similarities)
        ),
        "exact_match_rate": exact_matches / len(pairs),
    }


def number_normalize_text(text: str) -> str:
    protected: dict[str, str] = {}

    def protect(match: re.Match[str]) -> str:
        token = f"MATREFPLACEHOLDER{_alpha_index(len(protected))}END"
        protected[token] = match.group(0)
        return token

    value = re.sub(r"\$-?\d+\$", protect, str(text))
    value = NUMERIC_FRAGMENT_RE.sub("<NUM>", value)
    for token, original in protected.items():
        value = value.replace(token, original)
    return normalize_text(value)


def _alpha_index(index: int) -> str:
    index = max(int(index), 0)
    letters = []
    while True:
        letters.append(chr(ord("A") + index % 26))
        index = index // 26 - 1
        if index < 0:
            break
    return "".join(reversed(letters))


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(normalize_text(text))]


def _corpus_bleu_2(predictions: list[list[str]], references: list[list[str]]) -> float:
    pred_len = sum(len(tokens) for tokens in predictions)
    ref_len = sum(len(tokens) for tokens in references)
    if pred_len == 0:
        return 0.0

    precisions: list[float] = []
    for order in (1, 2):
        clipped = 0
        total = 0
        for prediction, reference in zip(predictions, references, strict=True):
            pred_counts = _ngram_counts(prediction, order)
            ref_counts = _ngram_counts(reference, order)
            clipped += sum(min(count, ref_counts.get(ngram, 0)) for ngram, count in pred_counts.items())
            total += sum(pred_counts.values())
        precisions.append((clipped + 1.0) / (total + 1.0))

    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / max(pred_len, 1))
    return brevity_penalty * math.exp(sum(math.log(precision) for precision in precisions) / 2.0)


def _ngram_counts(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < order:
        return Counter()
    return Counter(tuple(tokens[idx : idx + order]) for idx in range(len(tokens) - order + 1))


def _rouge_1(prediction: list[str], reference: list[str]) -> tuple[float, float, float]:
    if not prediction and not reference:
        return 1.0, 1.0, 1.0
    if not prediction or not reference:
        return 0.0, 0.0, 0.0
    overlap = sum((Counter(prediction) & Counter(reference)).values())
    precision = overlap / len(prediction)
    recall = overlap / len(reference)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1
