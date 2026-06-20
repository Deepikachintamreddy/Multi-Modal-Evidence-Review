# Multi-Modal Evidence Review System

A damage claim verification system that analyzes images against user claims to determine whether visual evidence supports, contradicts, or is insufficient for the claim.

## Architecture

The system uses a **single-pass VLM (Vision Language Model) pipeline** powered by Google Gemini 3.1 Flash-Lite (or other Gemini Flash models). For each claim, it:

1. **Extracts the claim** from the user conversation (handles multi-language input)
2. **Analyzes all submitted images** using the VLM in a single consolidated call
3. **Cross-references evidence requirements** for the claim's object type
4. **Assesses user risk** from historical claim data
5. **Produces a structured verdict** with justification, risk flags, and severity

### Key Design Decisions

- **Single-pass approach**: Sends all images + context in one VLM call per claim (vs. separate calls per image) to reduce latency and cost while maintaining contextual coherence
- **Anti-injection guardrails**: Explicitly instructs the VLM to ignore embedded instructions in user text or images
- **Few-shot prompting**: Includes 8 diverse labeled examples covering supported, contradicted, and not_enough_information cases
- **Strict output validation**: All generated values are validated against the allowed value lists before writing

## Setup

### 1. Install dependencies

```bash
cd code
python -m pip install -r requirements.txt
```

### 2. Set API key

```bash
# Windows PowerShell
$env:GEMINI_API_KEY = "your-gemini-api-key"

# Linux/macOS
export GEMINI_API_KEY="your-gemini-api-key"
```

Get a free API key at: https://aistudio.google.com/apikey

### 3. Run on sample data (evaluation)

```bash
python main.py --sample --strategy A
python main.py --sample --strategy B
```

### 4. Run evaluation

```bash
python evaluation/main.py --compare
```

### 5. Generate final predictions

```bash
python main.py
```

Output will be saved to `output.csv` (repo root).

## File Structure

```
code/
├── main.py                    # Entry point: processes claims → output.csv
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .env.example               # API key template
├── config.py                  # Constants, allowed values, paths
├── image_analyzer.py          # Gemini VLM image analysis (single & two-pass)
├── prompt_templates.py        # All VLM prompts with few-shot examples
├── utils.py                   # CSV I/O, validation, image loading, retry logic
└── evaluation/
    ├── main.py                # Evaluation entry point (metrics + report)
    ├── evaluation_report.md   # Generated operational analysis
    ├── stats.json             # Generated runtime statistics
    └── evaluation_results.json# Generated evaluation metrics
```

## Strategies

### Strategy A: Single-Pass (Default)
One consolidated VLM call per claim with all images, conversation, user history, and evidence requirements. Outputs structured JSON directly.

### Strategy B: Two-Pass
1. **Pass 1**: Analyze each image independently (object, damage, quality assessment)
2. **Pass 2**: Synthesize all image analyses with claim context to produce final verdict

## Evaluation

The system is evaluated on `dataset/sample_claims.csv` (20 labeled examples) using:

- **Exact match accuracy** for: claim_status, issue_type, object_part, evidence_standard_met, valid_image, severity
- **Set-based F1** for: risk_flags, supporting_image_ids
- **Weighted overall score** emphasizing claim_status and issue identification
