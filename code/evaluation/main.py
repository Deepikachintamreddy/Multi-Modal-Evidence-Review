#!/usr/bin/env python3
"""
evaluation/main.py — Runs both strategies on sample_claims.csv and compares.

Usage:
    cd code/evaluation
    python main.py
    python main.py --strategy single_call   # run only one strategy
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Add parent code/ to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from models import ClaimInput
from processor import ClaimProcessor
from utils import (
    read_evidence_requirements,
    read_sample_claims,
    read_user_history,
    write_output_csv,
)
from evaluator import evaluate_predictions, format_evaluation_report


def run_strategy(
    strategy: str,
    claims,
    user_history,
    evidence_reqs,
) -> tuple:
    """Run a strategy on all claims, return (outputs, processor)."""
    processor = ClaimProcessor(strategy=strategy)
    outputs = []
    start = time.time()

    for i, claim in enumerate(claims, 1):
        hist = user_history.get(claim.user_id)
        print(f"  [{i}/{len(claims)}] {claim.user_id} | {claim.claim_object}")
        output = processor.process_claim(claim, hist, evidence_reqs)
        outputs.append(output)
        if i < len(claims):
            time.sleep(3)  # respect free-tier per-minute limits

    elapsed = time.time() - start
    return outputs, processor, elapsed


def main():
    parser = argparse.ArgumentParser(description="Evaluation runner")
    parser.add_argument(
        "--strategy",
        choices=["single_call", "two_stage", "both"],
        default="both",
        help="Which strategy to evaluate (default: both)",
    )
    args = parser.parse_args()

    if not config.GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY not set.")
        print("Get your FREE key at: https://aistudio.google.com/apikey")
        sys.exit(1)

    print("=" * 60)
    print("  EVALUATION: Multi-Modal Evidence Review")
    print("=" * 60)

    # Load labeled data
    print("\nLoading sample_claims.csv (labeled data)...")
    sample_rows = read_sample_claims()
    user_history = read_user_history()
    evidence_reqs = read_evidence_requirements()

    claims = [
        ClaimInput(
            user_id=r.user_id,
            image_paths=r.image_paths,
            user_claim=r.user_claim,
            claim_object=r.claim_object,
        )
        for r in sample_rows
    ]

    # Expected outputs as dicts
    expectations = [r.model_dump() for r in sample_rows]

    print(f"  {len(claims)} labeled claims loaded\n")

    strategies_to_run = (
        ["single_call", "two_stage"] if args.strategy == "both"
        else [args.strategy]
    )

    all_results = {}

    for strategy in strategies_to_run:
        print(f"\n{'='*60}")
        print(f"  Running strategy: {strategy}")
        print(f"{'='*60}")

        outputs, processor, elapsed = run_strategy(
            strategy, claims, user_history, evidence_reqs
        )

        # Convert outputs to dicts for evaluation
        predictions = [o.model_dump() for o in outputs]

        # Evaluate
        results = evaluate_predictions(predictions, expectations)
        results["elapsed"] = elapsed
        results["usage"] = processor.usage.model_dump()
        all_results[strategy] = results

        # Print report
        report = format_evaluation_report(results, strategy)
        print(report)

        # Print operational stats
        print(f"  OPERATIONAL STATS ({strategy}):")
        print(f"    Time: {elapsed:.1f}s ({elapsed/len(claims):.1f}s/claim)")
        print(f"    API calls: {processor.usage.api_calls}")
        print(f"    Images: {processor.usage.images_processed}")
        print(f"    Input tokens: {processor.usage.input_tokens:,}")
        print(f"    Output tokens: {processor.usage.output_tokens:,}")
        input_cost = 0.0  # Gemini free tier
        output_cost = 0.0  # Gemini free tier
        print(f"    Est. cost: ${input_cost + output_cost:.2f}")

        # Save predictions
        out_path = Path(__file__).parent / f"sample_output_{strategy}.csv"
        write_output_csv(outputs, out_path)

    # Comparison
    if len(all_results) == 2:
        print("\n" + "=" * 60)
        print("  STRATEGY COMPARISON")
        print("=" * 60)
        s1, s2 = list(all_results.keys())
        r1, r2 = all_results[s1], all_results[s2]
        print(f"  {'Metric':<30} {s1:<15} {s2:<15} {'Winner':<15}")
        print(f"  {'-'*75}")
        print(f"  {'Overall Score':<30} {r1['overall_score']:.3f}{'':>9} "
              f"{r2['overall_score']:.3f}{'':>9} "
              f"{'← ' + s1 if r1['overall_score'] >= r2['overall_score'] else '← ' + s2}")

        for field in r1["per_field"]:
            v1 = r1["per_field"][field].get("accuracy", r1["per_field"][field].get("avg_f1", 0))
            v2 = r2["per_field"][field].get("accuracy", r2["per_field"][field].get("avg_f1", 0))
            winner = s1 if v1 >= v2 else s2
            print(f"  {field:<30} {v1:.3f}{'':>9} {v2:.3f}{'':>9} {'← ' + winner}")

        winner = s1 if r1["overall_score"] >= r2["overall_score"] else s2
        print(f"\n  RECOMMENDED STRATEGY: {winner}")
        print(f"  Run: python main.py --strategy {winner} --output ../output.csv")
        print("=" * 60)


if __name__ == "__main__":
    main()
