#!/usr/bin/env python3
"""
main.py — Entry point for the Multi-Modal Evidence Review system.

Usage:
    python main.py                                    # default: single_call, output to ../output.csv
    python main.py --strategy single_call             # explicit strategy
    python main.py --strategy two_stage               # two-stage strategy
    python main.py --input ../dataset/claims.csv      # custom input
    python main.py --output ../output.csv             # custom output path
    python main.py --sample                           # run on sample_claims.csv instead
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure code/ is on the path
sys.path.insert(0, str(Path(__file__).parent))

import config
from models import ClaimInput
from processor import ClaimProcessor
from utils import (
    read_claims,
    read_evidence_requirements,
    read_sample_claims,
    read_user_history,
    write_output_csv,
)


def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Evidence Review System")
    parser.add_argument(
        "--strategy",
        choices=["single_call", "two_stage"],
        default="single_call",
        help="Processing strategy (default: single_call)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV path (default: dataset/claims.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: output.csv in project root)",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run on sample_claims.csv instead of claims.csv",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the VLM model name",
    )
    args = parser.parse_args()

    # Resolve paths
    if args.input:
        input_path = Path(args.input)
    elif args.sample:
        input_path = config.SAMPLE_CLAIMS_CSV
    else:
        input_path = config.CLAIMS_CSV

    output_path = Path(args.output) if args.output else config.DEFAULT_OUTPUT_CSV

    # Override model if specified
    if args.model:
        config.MODEL_VISION = args.model

    # Validate API key
    if not config.GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY environment variable is not set.")
        print("Get your FREE key at: https://aistudio.google.com/apikey")
        print("Then set it with:")
        print("  PowerShell: $env:GOOGLE_API_KEY='your-key-here'")
        print("  Linux/Mac:  export GOOGLE_API_KEY=your-key-here")
        sys.exit(1)

    print("=" * 70)
    print("  Multi-Modal Evidence Review System")
    print("=" * 70)
    print(f"  Strategy:  {args.strategy}")
    print(f"  Model:     {config.MODEL_VISION}")
    print(f"  Input:     {input_path}")
    print(f"  Output:    {output_path}")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading data...")
    if args.sample:
        sample_rows = read_sample_claims(input_path)
        claims = [
            ClaimInput(
                user_id=r.user_id,
                image_paths=r.image_paths,
                user_claim=r.user_claim,
                claim_object=r.claim_object,
            )
            for r in sample_rows
        ]
    else:
        claims = read_claims(input_path)

    user_history = read_user_history()
    evidence_reqs = read_evidence_requirements()

    print(f"  Loaded {len(claims)} claims")
    print(f"  Loaded {len(user_history)} user history records")
    print(f"  Loaded {len(evidence_reqs)} evidence requirements")

    # Initialize processor
    print(f"\n[2/4] Initializing {args.strategy} processor...")
    processor = ClaimProcessor(strategy=args.strategy)

    # Process all claims
    print(f"\n[3/4] Processing {len(claims)} claims...")
    outputs = []
    start_time = time.time()

    for i, claim in enumerate(claims, 1):
        hist = user_history.get(claim.user_id)
        print(f"\n  [{i}/{len(claims)}] {claim.user_id} | {claim.claim_object} | "
              f"images: {claim.image_paths[:60]}...")

        output = processor.process_claim(claim, hist, evidence_reqs)
        outputs.append(output)

        print(f"    -> {output.claim_status} | {output.issue_type} | "
              f"{output.object_part} | severity={output.severity}")

        # Small delay between calls to respect rate limits
        if i < len(claims):
            time.sleep(3)  # respect free-tier per-minute limits

    elapsed = time.time() - start_time

    # Write output
    print(f"\n[4/4] Writing output...")
    write_output_csv(outputs, output_path)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Claims processed:  {len(outputs)}")
    print(f"  Total time:        {elapsed:.1f}s ({elapsed/len(outputs):.1f}s per claim)")
    print(f"  API calls:         {processor.usage.api_calls}")
    print(f"  Images processed:  {processor.usage.images_processed}")
    print(f"  Input tokens:      {processor.usage.input_tokens:,}")
    print(f"  Output tokens:     {processor.usage.output_tokens:,}")

    # Cost estimate (Gemini Flash = FREE tier)
    print(f"  Estimated cost:    $0.00 (Gemini free tier)")
    print(f"  Output saved to:   {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
