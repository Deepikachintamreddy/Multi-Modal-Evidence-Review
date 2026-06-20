"""
evaluator.py — Metrics computation for comparing predicted vs expected outputs.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


def exact_match(predicted: str, expected: str) -> bool:
    """Case-insensitive exact match."""
    return predicted.strip().lower() == expected.strip().lower()


def set_f1(predicted_str: str, expected_str: str) -> float:
    """F1 score for semicolon-separated sets (e.g., risk_flags, supporting_image_ids)."""
    pred_set = _to_set(predicted_str)
    exp_set = _to_set(expected_str)

    if not pred_set and not exp_set:
        return 1.0
    if not pred_set or not exp_set:
        return 0.0

    tp = len(pred_set & exp_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(exp_set) if exp_set else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _to_set(s: str) -> set:
    """Convert semicolon-separated string to a set, normalizing."""
    if not s or s.strip().lower() == "none":
        return {"none"}
    return {item.strip().lower() for item in s.split(";") if item.strip()}


def evaluate_predictions(
    predictions: List[Dict[str, str]],
    expectations: List[Dict[str, str]],
) -> Dict[str, object]:
    """
    Compare predictions against expectations, return per-field metrics.

    Both inputs are lists of dicts with the same keys.
    Returns a dict with:
      - per-field accuracy/F1
      - overall score
      - per-row details
    """
    n = min(len(predictions), len(expectations))
    if n == 0:
        return {"error": "No rows to evaluate", "n": 0}

    # Fields to evaluate
    exact_fields = [
        "claim_status",
        "issue_type",
        "object_part",
        "severity",
        "evidence_standard_met",
        "valid_image",
    ]
    set_fields = [
        "risk_flags",
        "supporting_image_ids",
    ]

    results = {f: {"correct": 0, "total": n} for f in exact_fields}
    results.update({f: {"f1_sum": 0.0, "total": n} for f in set_fields})
    row_details = []

    for i in range(n):
        pred = predictions[i]
        exp = expectations[i]
        row_detail = {"row": i + 1, "user_id": exp.get("user_id", "")}

        for field in exact_fields:
            p = pred.get(field, "")
            e = exp.get(field, "")
            match = exact_match(p, e)
            results[field]["correct"] += int(match)
            row_detail[field] = {"predicted": p, "expected": e, "match": match}

        for field in set_fields:
            p = pred.get(field, "none")
            e = exp.get(field, "none")
            f1 = set_f1(p, e)
            results[field]["f1_sum"] += f1
            row_detail[field] = {"predicted": p, "expected": e, "f1": f1}

        row_details.append(row_detail)

    # Compute summary metrics
    metrics = {}
    total_score = 0
    num_metrics = 0

    for field in exact_fields:
        acc = results[field]["correct"] / results[field]["total"]
        metrics[field] = {"accuracy": acc, "correct": results[field]["correct"], "total": n}
        total_score += acc
        num_metrics += 1

    for field in set_fields:
        avg_f1 = results[field]["f1_sum"] / results[field]["total"]
        metrics[field] = {"avg_f1": avg_f1, "total": n}
        total_score += avg_f1
        num_metrics += 1

    overall = total_score / num_metrics if num_metrics > 0 else 0

    return {
        "n": n,
        "overall_score": overall,
        "per_field": metrics,
        "row_details": row_details,
    }


def format_evaluation_report(
    results: Dict,
    strategy_name: str = "unknown",
) -> str:
    """Format evaluation results as a readable text report."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  EVALUATION: {strategy_name}")
    lines.append(f"{'='*60}")
    lines.append(f"  Rows evaluated: {results['n']}")
    lines.append(f"  Overall score:  {results['overall_score']:.3f}")
    lines.append("")

    lines.append("  Per-Field Breakdown:")
    lines.append(f"  {'Field':<28} {'Metric':<12} {'Score':<8}")
    lines.append(f"  {'-'*48}")

    for field, data in results["per_field"].items():
        if "accuracy" in data:
            lines.append(f"  {field:<28} {'accuracy':<12} {data['accuracy']:.3f}  "
                         f"({data['correct']}/{data['total']})")
        elif "avg_f1" in data:
            lines.append(f"  {field:<28} {'avg_f1':<12} {data['avg_f1']:.3f}")

    lines.append(f"{'='*60}")

    # Per-row mismatches
    mismatches = []
    for row in results.get("row_details", []):
        row_misses = []
        for field in ["claim_status", "issue_type", "object_part", "severity",
                       "evidence_standard_met", "valid_image"]:
            if field in row and not row[field].get("match", True):
                row_misses.append(
                    f"    {field}: predicted='{row[field]['predicted']}' "
                    f"expected='{row[field]['expected']}'"
                )
        for field in ["risk_flags", "supporting_image_ids"]:
            if field in row and row[field].get("f1", 1.0) < 1.0:
                row_misses.append(
                    f"    {field}: predicted='{row[field]['predicted']}' "
                    f"expected='{row[field]['expected']}' (F1={row[field]['f1']:.2f})"
                )
        if row_misses:
            mismatches.append(f"  Row {row['row']} ({row['user_id']}):")
            mismatches.extend(row_misses)

    if mismatches:
        lines.append("\n  MISMATCHES:")
        lines.extend(mismatches)
        lines.append("")

    return "\n".join(lines)
