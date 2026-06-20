import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
Multi-Modal Evidence Review System
Main entry point for processing damage claims.

Usage:
    python main.py                    # Process claims.csv -> output.csv
    python main.py --sample           # Process sample_claims.csv (for evaluation)
    python main.py --strategy B       # Use two-pass strategy
    python main.py --sample --strategy B  # Evaluate with two-pass

Environment:
    GEMINI_API_KEY: Google Gemini API key (required)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add code dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CLAIMS_CSV,
    OUTPUT_CSV,
    SAMPLE_CLAIMS_CSV,
    DATASET_DIR,
    OUTPUT_COLUMNS,
)
from image_analyzer import ImageAnalyzer
from utils import (
    get_relevant_requirements,
    load_claims,
    load_evidence_requirements,
    load_user_history,
    save_output_csv,
    validate_and_format_row,
)


def process_claims(
    input_csv: Path,
    output_csv: Path,
    strategy: str = "A",
    api_key: str | None = None,
) -> list[dict]:
    """Process all claims from input CSV and produce output CSV.

    Args:
        input_csv: Path to claims CSV (claims.csv or sample_claims.csv)
        output_csv: Path for output CSV
        strategy: "A" for single-pass, "B" for two-pass
        api_key: Optional API key override

    Returns:
        List of output row dicts
    """
    print(f"\n{'='*70}")
    print(f"Multi-Modal Evidence Review System")
    print(f"{'='*70}")
    print(f"Input:    {input_csv}")
    print(f"Output:   {output_csv}")
    print(f"Strategy: {'Single-pass (A)' if strategy == 'A' else 'Two-pass (B)'}")
    print(f"{'='*70}\n")

    # Load data
    print("[LOAD] Loading data...")
    claims_df = load_claims(input_csv)
    user_history = load_user_history()
    evidence_requirements = load_evidence_requirements()

    print(f"   Claims: {len(claims_df)} rows")
    print(f"   Users with history: {len(user_history)}")
    print(f"   Evidence requirements: {len(evidence_requirements)}")

    # Initialize analyzer
    print("\n[INIT] Initializing Gemini VLM analyzer...")
    analyzer = ImageAnalyzer(api_key=api_key)
    print(f"   Model: {analyzer.model}")

    # Process each claim
    output_rows = []
    total = len(claims_df)

    for idx, (_, row) in enumerate(claims_df.iterrows()):
        claim_row = row.to_dict()
        user_id = claim_row["user_id"]
        claim_object = claim_row["claim_object"]

        print(f"\n{'─'*50}")
        print(f"[CLAIM] {idx + 1}/{total}: {user_id} ({claim_object})")
        print(f"   Images: {claim_row['image_paths'][:80]}...")

        # Get user history
        history = user_history.get(user_id, {})
        if history:
            flags = history.get("history_flags", "none")
            print(f"   User history: {history.get('past_claim_count', '?')} past claims, flags={flags}")
        else:
            print(f"   User history: not found")

        # Get relevant evidence requirements
        relevant_reqs = get_relevant_requirements(evidence_requirements, claim_object)

        # Analyze
        if strategy == "B":
            result = analyzer.analyze_claim_two_pass(claim_row, history, relevant_reqs)
        else:
            result = analyzer.analyze_claim(claim_row, history, relevant_reqs)

        if result is None:
            print(f"   [FAIL] Analysis failed, using default")
            result = analyzer._default_insufficient_result([])

        # Validate and format
        output_row = validate_and_format_row(result, claim_row)
        output_rows.append(output_row)

        # Print summary
        status = output_row["claim_status"]
        issue = output_row["issue_type"]
        part = output_row["object_part"]
        severity = output_row["severity"]
        flags = output_row["risk_flags"]
        status_marker = {"supported": "[OK]", "contradicted": "[X]", "not_enough_information": "[?]"}
        print(f"   {status_marker.get(status, '?')} Status: {status}")
        print(f"   Issue: {issue} | Part: {part} | Severity: {severity}")
        if flags != "none":
            print(f"   [WARN] Flags: {flags}")

    # Save output
    save_output_csv(output_rows, output_csv)

    # Print stats
    stats = analyzer.get_stats()
    print(f"\n{'='*70}")
    print(f"[STATS] Operational Statistics")
    print(f"{'='*70}")
    print(f"   API calls:        {stats['total_calls']}")
    print(f"   Images processed: {stats['total_images']}")
    print(f"   Input tokens:     {stats['total_input_tokens']:,}")
    print(f"   Output tokens:    {stats['total_output_tokens']:,}")
    print(f"   Errors:           {stats['errors']}")
    print(f"   Elapsed time:     {stats['elapsed_seconds']:.1f}s")
    print(f"   Avg time/call:    {stats['avg_seconds_per_call']:.1f}s")
    print(f"{'='*70}\n")

    # Save stats for evaluation report
    stats_path = Path(__file__).parent / "evaluation" / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return output_rows


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Modal Evidence Review System"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Process sample_claims.csv instead of claims.csv",
    )
    parser.add_argument(
        "--strategy",
        choices=["A", "B"],
        default="A",
        help="A=single-pass (default), B=two-pass",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (overrides GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output CSV path",
    )
    args = parser.parse_args()

    # Determine input/output paths
    if args.sample:
        input_csv = SAMPLE_CLAIMS_CSV
        if args.output:
            output_csv = Path(args.output)
        else:
            output_csv = Path(__file__).parent / "evaluation" / f"sample_output_strategy_{args.strategy}.csv"
    else:
        input_csv = CLAIMS_CSV
        output_csv = Path(args.output) if args.output else OUTPUT_CSV

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Process
    process_claims(
        input_csv=input_csv,
        output_csv=output_csv,
        strategy=args.strategy,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
