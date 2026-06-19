# Evaluation Report

## Strategy Comparison

### Strategy A: Single-Pass VLM

Evaluated on `dataset/sample_claims.csv` (20 labeled examples):

| Field | Metric | Score |
|---|---|---|
| claim_status | accuracy | 90.0% |
| issue_type | accuracy | 80.0% |
| object_part | accuracy | 90.0% |
| evidence_standard_met | accuracy | 100.0% |
| valid_image | accuracy | 100.0% |
| severity | accuracy | 75.0% |
| risk_flags | f1 | 92.3% |
| supporting_image_ids | f1 | 83.3% |

**Overall Weighted Score: 88.4%**

## Operational Analysis

### Resource Usage (Full Test Set — 44 claims)

- **Model calls**: 44
- **Images processed**: 82
- **Input tokens**: 293,231
- **Output tokens**: 7,673
- **Errors**: 0
- **Retries**: 0
- **Runtime**: 702.2s (~11.7 minutes)
- **Avg time per call**: 16.0s

### Cost Estimation

- **Model**: gemini-3.1-flash-lite
- **Input token cost**: $0.0440 (at $0.15/1M tokens)
- **Output token cost**: $0.0046 (at $0.60/1M tokens)
- **Total estimated cost**: $0.0486

### TPM/RPM Considerations

- **Rate limiting**: 4.5 second delay between API calls to stay within free tier limits
- **Retry strategy**: Exponential backoff with max 5 retries; parses server-suggested retry delay from 429 errors
- **Batching**: Single-pass strategy minimizes API calls (1 per claim vs 2+ for two-pass)
- **Caching**: Not implemented but would be valuable for re-runs on same dataset
- **Image optimization**: Images loaded as-is via PIL; no resizing applied

### Scalability Notes

- For production, batch API calls using async/concurrent execution
- Implement response caching to avoid redundant calls on re-runs
- Consider image resizing for large photos to reduce token usage
- Monitor token usage per request to optimize prompt length
