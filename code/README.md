# Multi-Modal Evidence Review System

## Overview
A system that verifies damage claims (car, laptop, package) using submitted images, claim conversations, user history, and evidence requirements. Built for the HackerRank Orchestrate June 2026 hackathon.

## Quick Start

### Prerequisites
- Python 3.10+
- Google Gemini API key

### Setup
```bash
cd code
pip install -r requirements.txt
```

Get your **FREE** API key at https://aistudio.google.com/apikey, then:
```bash
export GOOGLE_API_KEY=your-key-here          # Linux/Mac
set GOOGLE_API_KEY=your-key-here              # Windows CMD
$env:GOOGLE_API_KEY="your-key-here"           # Windows PowerShell
```

### Run Evaluation (on labeled sample data)
```bash
cd code/evaluation
python main.py                   # runs both strategies, compares
python main.py --strategy single_call   # run one strategy only
```

### Generate Predictions (on test data)
```bash
cd code
python main.py --strategy single_call --output ../output.csv
```

### Options
```
--strategy {single_call,two_stage}   Processing strategy (default: single_call)
--input PATH                         Custom input CSV path
--output PATH                        Custom output CSV path
--sample                             Run on sample_claims.csv instead
--model MODEL_NAME                   Override VLM model name
```

## Architecture

```
claims.csv + user_history.csv + evidence_requirements.csv + images/
    │
    ▼
┌────────────────────────────────────────┐
│  Strategy: single_call                 │
│  One VLM call per claim with:          │
│  - All images (base64, resized)        │
│  - Conversation text                   │
│  - User history context                │
│  - Evidence requirements               │
│  → Structured JSON verdict             │
├────────────────────────────────────────┤
│  Strategy: two_stage                   │
│  Stage 1: Text-only extraction (fast)  │
│  Stage 2: Vision analysis with context │
└────────────────────────────────────────┘
    │
    ▼
Post-validation: enum coercion + history risk merge
    │
    ▼
output.csv (14 columns, exact schema)
```

## File Structure
```
code/
├── main.py              # CLI entry point
├── config.py            # Constants, paths, allowed values
├── models.py            # Pydantic data models
├── utils.py             # CSV I/O, image loading, validation
├── processor.py         # Core VLM processing (both strategies)
├── prompts/
│   ├── system_prompt.txt      # Main system prompt (single_call)
│   ├── extraction_prompt.txt  # Stage 1 prompt (two_stage)
│   └── analysis_prompt.txt    # Stage 2 prompt (two_stage)
├── evaluation/
│   ├── main.py                # Evaluation runner
│   ├── evaluator.py           # Metrics computation
│   └── evaluation_report.md   # Results + operational analysis
├── requirements.txt
└── README.md
```

## Key Design Decisions

1. **Images are primary source of truth** — VLM directly inspects every image; justifications reference image IDs.
2. **Prompt injection defense** — System prompt explicitly instructs the VLM to flag (never follow) embedded instructions. Detects `text_instruction_present` risk.
3. **Color/side/identity matching** — Explicit instructions to verify object color, side, and identity against the claim.
4. **Deterministic post-validation** — All VLM outputs are coerced to valid enum values. User history flags are merged additively.
5. **Two strategies compared** — `single_call` (1 VLM call/claim) vs `two_stage` (2 calls/claim). Evaluation on labeled data determines the winner.

## Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | (required) | Google AI Studio API key (FREE) |
| `MODEL_VISION` | `gemini-2.0-flash-lite` | Vision model for analysis |
| `MODEL_FAST` | `gemini-2.0-flash-lite` | Fast model for extraction |
