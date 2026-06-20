"""
Evaluation module for Multi-Modal Evidence Review System.
Compares predictions against ground truth from sample_claims.csv.
Supports comparing two strategies and generating evaluation reports.

Usage:
    python evaluation/main.py                           # Evaluate strategy A
    python evaluation/main.py --compare                 # Compare A vs B
    python evaluation/main.py --pred path/to/output.csv # Custom predictions file
"""

import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import SAMPLE_CLAIMS_CSV, OUTPUT_COLUMNS, GEMINI_MODEL, REQUEST_DELAY, MAX_RETRIES


def load_ground_truth() -> pd.DataFrame:
    """Load ground truth from sample_claims.csv."""
    df = pd.read_csv(SAMPLE_CLAIMS_CSV, dtype=str)
    return df


def load_predictions(pred_path: Path) -> pd.DataFrame:
    """Load predictions CSV."""
    df = pd.read_csv(pred_path, dtype=str)
    return df


def compute_exact_match(gt_series: pd.Series, pred_series: pd.Series) -> float:
    """Compute exact match accuracy between two series."""
    matches = (gt_series.str.strip().str.lower() == pred_series.str.strip().str.lower())
    return matches.mean()


def compute_set_f1(gt_series: pd.Series, pred_series: pd.Series) -> float:
    """Compute average F1 for semicolon-separated set fields (risk_flags, supporting_image_ids)."""
    f1_scores = []
    for gt_val, pred_val in zip(gt_series, pred_series):
        gt_set = set(s.strip() for s in str(gt_val).split(";") if s.strip())
        pred_set = set(s.strip() for s in str(pred_val).split(";") if s.strip())

        if not gt_set and not pred_set:
            f1_scores.append(1.0)
            continue
        if not gt_set or not pred_set:
            f1_scores.append(0.0)
            continue

        tp = len(gt_set & pred_set)
        precision = tp / len(pred_set) if pred_set else 0
        recall = tp / len(gt_set) if gt_set else 0

        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2 * precision * recall / (precision + recall))

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def evaluate(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    strategy_name: str = "Strategy A",
) -> dict:
    """Evaluate predictions against ground truth.

    Returns dict with per-field accuracy/F1 and overall score.
    """
    print(f"\n{'='*60}")
    print(f"📊 Evaluation: {strategy_name}")
    print(f"{'='*60}")

    if len(gt_df) != len(pred_df):
        print(f"⚠️  Row count mismatch: ground truth={len(gt_df)}, predictions={len(pred_df)}")
        min_len = min(len(gt_df), len(pred_df))
        gt_df = gt_df.head(min_len)
        pred_df = pred_df.head(min_len)

    results = {}

    # Exact match fields
    exact_match_fields = [
        "claim_status",
        "issue_type",
        "object_part",
        "evidence_standard_met",
        "valid_image",
        "severity",
    ]

    for field in exact_match_fields:
        if field in gt_df.columns and field in pred_df.columns:
            acc = compute_exact_match(gt_df[field], pred_df[field])
            results[field] = {"metric": "accuracy", "value": acc}
            emoji = "✅" if acc >= 0.8 else "⚠️" if acc >= 0.6 else "❌"
            print(f"  {emoji} {field:35s} accuracy: {acc:.1%}")

            # Show mismatches for debugging
            if acc < 1.0:
                mismatches = gt_df[field].str.strip().str.lower() != pred_df[field].str.strip().str.lower()
                mismatch_indices = mismatches[mismatches].index.tolist()
                for mi in mismatch_indices[:3]:  # Show max 3
                    print(f"      Row {mi}: expected='{gt_df[field].iloc[mi]}' got='{pred_df[field].iloc[mi]}'")
                if len(mismatch_indices) > 3:
                    print(f"      ... and {len(mismatch_indices) - 3} more mismatches")

    # Set-based F1 fields
    set_fields = ["risk_flags", "supporting_image_ids"]

    for field in set_fields:
        if field in gt_df.columns and field in pred_df.columns:
            f1 = compute_set_f1(gt_df[field], pred_df[field])
            results[field] = {"metric": "f1", "value": f1}
            emoji = "✅" if f1 >= 0.8 else "⚠️" if f1 >= 0.6 else "❌"
            print(f"  {emoji} {field:35s} F1:       {f1:.1%}")

    # Overall score (weighted average)
    weights = {
        "claim_status": 3.0,       # Most important
        "issue_type": 2.0,
        "object_part": 2.0,
        "evidence_standard_met": 1.5,
        "valid_image": 1.0,
        "severity": 1.5,
        "risk_flags": 2.0,
        "supporting_image_ids": 1.5,
    }

    total_weight = 0
    weighted_sum = 0
    for field, data in results.items():
        w = weights.get(field, 1.0)
        weighted_sum += data["value"] * w
        total_weight += w

    overall = weighted_sum / total_weight if total_weight > 0 else 0
    results["overall_weighted_score"] = overall

    print(f"\n  {'🏆' if overall >= 0.8 else '📈'} Overall weighted score: {overall:.1%}")
    print(f"{'='*60}\n")

    return results


def generate_report(
    results_a: dict,
    results_b: dict | None = None,
    stats_a: dict | None = None,
    stats_b: dict | None = None,
    output_path: Path | None = None,
) -> str:
    """Generate evaluation report markdown."""
    report = "# Evaluation Report\n\n"
    report += "## Strategy Comparison\n\n"

    # Strategy A results
    report += "### Strategy A: Single-Pass VLM\n\n"
    report += "| Field | Metric | Score |\n|---|---|---|\n"
    for field, data in results_a.items():
        if field != "overall_weighted_score":
            report += f"| {field} | {data['metric']} | {data['value']:.1%} |\n"
    report += f"\n**Overall Weighted Score: {results_a.get('overall_weighted_score', 0):.1%}**\n\n"

    if results_b:
        report += "### Strategy B: Two-Pass VLM\n\n"
        report += "| Field | Metric | Score |\n|---|---|---|\n"
        for field, data in results_b.items():
            if field != "overall_weighted_score":
                report += f"| {field} | {data['metric']} | {data['value']:.1%} |\n"
        report += f"\n**Overall Weighted Score: {results_b.get('overall_weighted_score', 0):.1%}**\n\n"

        # Winner
        score_a = results_a.get("overall_weighted_score", 0)
        score_b = results_b.get("overall_weighted_score", 0)
        winner = "Strategy A (Single-Pass)" if score_a >= score_b else "Strategy B (Two-Pass)"
        report += f"### Winner: {winner}\n\n"
    else:
        report += "*Strategy B not evaluated. Run with --compare to compare both strategies.*\n\n"

    # Operational Analysis
    report += "## Operational Analysis\n\n"

    if stats_a:
        report += "### Resource Usage (Strategy A)\n\n"
        report += f"- **Model calls**: {stats_a.get('total_calls', 'N/A')}\n"
        report += f"- **Images processed**: {stats_a.get('total_images', 'N/A')}\n"
        report += f"- **Input tokens**: {stats_a.get('total_input_tokens', 'N/A'):,}\n"
        report += f"- **Output tokens**: {stats_a.get('total_output_tokens', 'N/A'):,}\n"
        report += f"- **Errors**: {stats_a.get('errors', 0)}\n"
        report += f"- **Runtime**: {stats_a.get('elapsed_seconds', 'N/A')}s\n"
        report += f"- **Avg time per call**: {stats_a.get('avg_seconds_per_call', 'N/A')}s\n\n"

        # Cost estimation
        input_tokens = stats_a.get("total_input_tokens", 0)
        output_tokens = stats_a.get("total_output_tokens", 0)
        # Gemini 2.5 Flash pricing (approximate)
        input_cost = (input_tokens / 1_000_000) * 0.15   # $0.15 per 1M input tokens
        output_cost = (output_tokens / 1_000_000) * 0.60  # $0.60 per 1M output tokens
        total_cost = input_cost + output_cost

        report += "### Cost Estimation\n\n"
        report += f"- **Model**: {GEMINI_MODEL}\n"
        report += f"- **Input token cost**: ${input_cost:.4f} (at $0.15/1M tokens)\n"
        report += f"- **Output token cost**: ${output_cost:.4f} (at $0.60/1M tokens)\n"
        report += f"- **Total estimated cost**: ${total_cost:.4f}\n\n"

        # Extrapolate to 44 test claims if this is a sample run
        num_calls = stats_a.get("total_calls", 1)
        if num_calls < 30: # Sample run has 20 claims
            extrapolated_calls = 42 # 44 claims total, 2 skipped due to no images
            extrapolated_input_tokens = int(input_tokens * (extrapolated_calls / num_calls))
            extrapolated_output_tokens = int(output_tokens * (extrapolated_calls / num_calls))
            extrapolated_input_cost = (extrapolated_input_tokens / 1_000_000) * 0.15
            extrapolated_output_cost = (extrapolated_output_tokens / 1_000_000) * 0.60
            extrapolated_total_cost = extrapolated_input_cost + extrapolated_output_cost
            
            report += "### Extrapolation to Full Test Set (44 claims)\n\n"
            report += f"- **Estimated Model Calls**: {extrapolated_calls}\n"
            report += f"- **Estimated Input Tokens**: {extrapolated_input_tokens:,}\n"
            report += f"- **Estimated Output Tokens**: {extrapolated_output_tokens:,}\n"
            report += f"- **Estimated Test Set Cost**: ${extrapolated_total_cost:.4f} (at $0.15/1M input, $0.60/1M output tokens)\n\n"

    report += "### TPM/RPM Considerations\n\n"
    report += f"- **Rate limiting**: {REQUEST_DELAY} second delay between API calls to stay within free tier limits\n"
    report += f"- **Retry strategy**: Exponential backoff with max {MAX_RETRIES} retries\n"
    report += "- **Batching**: Single-pass strategy minimizes API calls (1 per claim vs 2+ for two-pass)\n"
    report += "- **Caching**: Not implemented but would be valuable for re-runs on same dataset\n"
    report += "- **Image optimization**: Images loaded as-is (PIL handles format conversion)\n\n"

    report += "### Scalability Notes\n\n"
    report += "- For production, batch API calls using async/concurrent execution\n"
    report += "- Implement response caching to avoid redundant calls on re-runs\n"
    report += "- Consider image resizing for large photos to reduce token usage\n"
    report += "- Monitor token usage per request to optimize prompt length\n"

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"📝 Report saved to {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate claim predictions")
    parser.add_argument(
        "--pred",
        type=str,
        default=None,
        help="Path to predictions CSV (default: evaluation/sample_output_strategy_A.csv)",
    )
    parser.add_argument(
        "--pred-b",
        type=str,
        default=None,
        help="Path to Strategy B predictions CSV for comparison",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare Strategy A and B predictions",
    )
    args = parser.parse_args()

    eval_dir = Path(__file__).resolve().parent

    # Load ground truth
    gt_df = load_ground_truth()
    print(f"📋 Ground truth loaded: {len(gt_df)} rows from sample_claims.csv")

    # Strategy A evaluation
    pred_a_path = Path(args.pred) if args.pred else eval_dir / "sample_output_strategy_A.csv"
    if not pred_a_path.exists():
        print(f"❌ Predictions file not found: {pred_a_path}")
        print("   Run: python main.py --sample --strategy A")
        sys.exit(1)

    pred_a_df = load_predictions(pred_a_path)
    results_a = evaluate(gt_df, pred_a_df, "Strategy A (Single-Pass)")

    # Load stats if available
    stats_a = None
    stats_a_path = eval_dir / "stats.json"
    if stats_a_path.exists():
        with open(stats_a_path) as f:
            stats_a = json.load(f)

    # Strategy B evaluation (if requested)
    results_b = None
    stats_b = None
    if args.compare or args.pred_b:
        pred_b_path = Path(args.pred_b) if args.pred_b else eval_dir / "sample_output_strategy_B.csv"
        if pred_b_path.exists():
            pred_b_df = load_predictions(pred_b_path)
            results_b = evaluate(gt_df, pred_b_df, "Strategy B (Two-Pass)")
        else:
            print(f"⚠️  Strategy B predictions not found: {pred_b_path}")
            print("   Run: python main.py --sample --strategy B")

    # Generate report
    report_path = eval_dir / "evaluation_report.md"
    generate_report(
        results_a=results_a,
        results_b=results_b,
        stats_a=stats_a,
        stats_b=stats_b,
        output_path=report_path,
    )

    # Save results JSON
    eval_results = {"strategy_a": results_a}
    if results_b:
        eval_results["strategy_b"] = results_b
    results_json_path = eval_dir / "evaluation_results.json"
    with open(results_json_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"📊 Results saved to {results_json_path}")


if __name__ == "__main__":
    main()
