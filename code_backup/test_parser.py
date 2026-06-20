import sys
sys.path.insert(0, '.')
from utils import parse_json_response

# Test multiline JSON (the format Gemini returns)
test = """{
  "evidence_standard_met": "true",
  "evidence_standard_met_reason": "The rear bumper is visible and the dent can be verified.",
  "risk_flags": "none",
  "issue_type": "dent",
  "object_part": "rear_bumper",
  "claim_status": "supported",
  "claim_status_justification": "The image clearly shows a dent on the rear bumper.",
  "supporting_image_ids": "img_1",
  "valid_image": "true",
  "severity": "medium"
}"""

result = parse_json_response(test)
print(f"Test 1 (clean JSON): {'PASS' if result and result.get('claim_status') == 'supported' else 'FAIL'}")

# Test with markdown fencing
test2 = """```json
{
  "evidence_standard_met": "true",
  "claim_status": "contradicted"
}
```"""

result2 = parse_json_response(test2)
print(f"Test 2 (fenced JSON): {'PASS' if result2 and result2.get('claim_status') == 'contradicted' else 'FAIL'}")

# Test with extra text before/after
test3 = """Here is my analysis:
{
  "evidence_standard_met": "false",
  "claim_status": "not_enough_information"
}
Let me know if you need more details."""

result3 = parse_json_response(test3)
print(f"Test 3 (embedded JSON): {'PASS' if result3 and result3.get('claim_status') == 'not_enough_information' else 'FAIL'}")

print("\nAll parser tests completed.")
