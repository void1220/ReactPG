"""Levenshtein text-gap metrics for decoded procedure strings."""

from __future__ import annotations

from typing import Iterable, Sequence


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split()).rstrip(".")


def edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    """Compute Levenshtein edit distance with O(min(n, m)) memory."""

    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_idx, left_item in enumerate(left, start=1):
        current = [left_idx]
        for right_idx, right_item in enumerate(right, start=1):
            insertion = current[right_idx - 1] + 1
            deletion = previous[right_idx] + 1
            substitution = previous[right_idx - 1] + (left_item != right_item)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def levenshtein_similarity(prediction: str, reference: str) -> float:
    """Return normalized character-level Levenshtein similarity."""

    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    distance = edit_distance(pred, ref)
    return 1.0 - distance / max(len(pred), len(ref), 1)


def text_gap(prediction: str, reference: str) -> float:
    """Return normalized character-level text gap; lower is better."""

    return 1.0 - levenshtein_similarity(prediction, reference)


def corpus_levenshtein_metrics(
    pairs: Iterable[tuple[str, str]],
    *,
    threshold: float = 0.75,
) -> dict[str, float]:
    similarities: list[float] = []
    gaps: list[float] = []
    exact_matches = 0
    for prediction, reference in pairs:
        similarity = levenshtein_similarity(prediction, reference)
        gap = 1.0 - similarity
        similarities.append(similarity)
        gaps.append(gap)
        if normalize_text(prediction) == normalize_text(reference):
            exact_matches += 1

    total = len(similarities)
    if total == 0:
        return {
            "count": 0.0,
            "mean_text_gap": 0.0,
            "mean_levenshtein_similarity": 0.0,
            "levenshtein_75_rate": 0.0,
            "exact_match_rate": 0.0,
        }

    return {
        "count": float(total),
        "mean_text_gap": sum(gaps) / total,
        "mean_levenshtein_similarity": sum(similarities) / total,
        "levenshtein_75_rate": sum(sim >= threshold for sim in similarities) / total,
        "exact_match_rate": exact_matches / total,
    }
