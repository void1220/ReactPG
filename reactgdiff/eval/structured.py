"""Thoth-inspired OpenExp metrics, version 1 (not SciRecipe scores).

Step-M/Order-S/Order-LCS follow Appendix F of arXiv:2510.15600v2.
Semantic-A uses deterministic placeholder/quantity tokens instead of SciRecipe
objects and subword compensation. LCS anchors are monotone, so their Tau is
necessarily 1 when two anchors exist; occurrence_tau is a separate diagnostic.
"""
from collections import Counter, defaultdict, deque
import math
from reactgdiff.data.action_parser import parse_action_sequence
from reactgdiff.data.numeric_evidence import parse_numeric_value_unit

VERSION = "thoth_openexp_v1"


def anchors(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) - 1, -1, -1):
        for j in range(len(b) - 1, -1, -1):
            dp[i][j] = 1 + dp[i+1][j+1] if a[i] == b[j] else max(dp[i+1][j], dp[i][j+1])
    pairs, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            pairs.append((i, j)); i += 1; j += 1
        elif dp[i+1][j] >= dp[i][j+1]:
            i += 1
        else:
            j += 1
    return pairs


def edits(a, b):
    """Minimal edits from predicted operations to reference; deterministic ties."""
    dp = [[0] * (len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): dp[i][0] = i
    for j in range(len(b)+1): dp[0][j] = j
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(a[i-1] != b[j-1]))
    counts = Counter(missing=0, extra=0, substitution=0)
    confusion = Counter()
    i, j = len(a), len(b)
    while i or j:
        if i and j and dp[i][j] == dp[i-1][j-1]+(a[i-1] != b[j-1]):
            if a[i-1] != b[j-1]:
                counts['substitution'] += 1; confusion[a[i-1]+' -> '+b[j-1]] += 1
            i -= 1; j -= 1
        elif i and dp[i][j] == dp[i-1][j]+1:
            counts['extra'] += 1; i -= 1
        else:
            counts['missing'] += 1; j -= 1
    return dict(counts), dict(confusion)


def iou(a, b):
    return len(a & b) / len(a | b) if a or b else 1.0


def parameters(step):
    result = set(step.duration_refs + step.temperature_refs)
    for q in step.quantities:
        parsed = parse_numeric_value_unit(q)
        result.add(f'{parsed[0]:.8g}:{parsed[1]}' if parsed else q.strip().lower())
    return result


def pair_metrics(prediction, reference):
    p, r = parse_action_sequence(prediction), parse_action_sequence(reference)
    a, b = [s.operation_type for s in p], [s.operation_type for s in r]
    pairs = anchors(a, b)
    n, m = len(a), len(b)
    valid = bool(n and m)
    alignment = 0.0
    objects_sum = params_sum = 0.0
    for i, j in pairs:
        objects = iou(set(p[i].material_refs), set(r[j].material_refs))
        param = iou(parameters(p[i]), parameters(r[j])) if objects >= .5 else 0.0
        decay = max(0.0, 1 - (abs(i-j)/max(m, 1))**1.5)
        alignment += decay * (objects + .5 * param)
        objects_sum += objects; params_sum += param
    strict_subsequence = float(valid and len(pairs) == min(n, m))
    positions = defaultdict(deque)
    for j, op in enumerate(b): positions[op].append(j)
    matched = [positions[op].popleft() for op in a if positions[op]]
    concordant = sum(x < y for i, x in enumerate(matched) for y in matched[i+1:])
    total_pairs = len(matched) * (len(matched)-1) // 2
    errors, _ = edits(a, b)
    return {
        'thoth_step_m': float(valid and n == m),
        'thoth_order_s': float(valid and a == b),
        'thoth_order_lcs': 2*len(pairs)/max(n+m, 1),
        'thoth_order_tau_monotone': float(len(pairs) >= 2),
        'thoth_semantic_a_adapted': strict_subsequence + alignment/max(len(pairs), 1),
        'aligned_object_iou': objects_sum/max(n, m, 1),
        'aligned_parameter_iou': params_sum/max(n, m, 1),
        'occurrence_order_tau': (2*concordant-total_pairs)/total_pairs if total_pairs else 0.0,
        'occurrence_order_tau_defined': float(total_pairs > 0),
        'invalid_empty_prediction': float(not n),
        'same_operation_multiset_wrong_order': float(valid and a != b and Counter(a) == Counter(b)),
        **{'skeleton_'+k: float(v) for k, v in errors.items()},
    }


def corpus_structured_metrics(pairs):
    rows = [pair_metrics(p, r) for p, r in pairs]
    keys = pair_metrics('', '').keys()
    return {'structured_count': len(rows), **{k: sum(x[k] for x in rows)/max(len(rows), 1) for k in keys}}
