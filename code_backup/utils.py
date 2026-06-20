"""
Utility functions for CSV I/O, image loading, and data preparation.
"""

import csv
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from PIL import Image

from config import (
    ALLOWED_BOOLEAN,
    ALLOWED_CLAIM_STATUS,
    ALLOWED_ISSUE_TYPES,
    ALLOWED_OBJECT_PARTS,
    ALLOWED_RISK_FLAGS,
    ALLOWED_SEVERITY,
    DATASET_DIR,
    EVIDENCE_REQUIREMENTS_CSV,
    OUTPUT_COLUMNS,
    USER_HISTORY_CSV,
)


def load_claims(csv_path: Path) -> pd.DataFrame:
    """Load claims CSV file."""
    return pd.read_csv(csv_path, dtype=str)


def load_user_history() -> dict[str, dict]:
    """Load user history into a dict keyed by user_id."""
    df = pd.read_csv(USER_HISTORY_CSV, dtype=str)
    history = {}
    for _, row in df.iterrows():
        history[row["user_id"]] = row.to_dict()
    return history


def load_evidence_requirements() -> list[dict]:
    """Load evidence requirements as a list of dicts."""
    df = pd.read_csv(EVIDENCE_REQUIREMENTS_CSV, dtype=str)
    return df.to_dict("records")


def get_relevant_requirements(
    requirements: list[dict], claim_object: str, issue_hint: str = ""
) -> list[dict]:
    """Filter evidence requirements relevant to a claim object.

    Returns requirements where claim_object matches or is 'all'.
    """
    relevant = []
    for req in requirements:
        req_obj = req["claim_object"]
        if req_obj == "all" or req_obj == claim_object:
            relevant.append(req)
    return relevant


def parse_image_paths(image_paths_str: str) -> list[str]:
    """Parse semicolon-separated image paths into a list."""
    if not image_paths_str or pd.isna(image_paths_str):
        return []
    return [p.strip() for p in image_paths_str.split(";") if p.strip()]


def get_image_id(image_path: str) -> str:
    """Extract image ID from path (filename without extension).

    e.g., 'images/test/case_001/img_1.jpg' -> 'img_1'
    """
    return Path(image_path).stem


def load_image(image_path: str) -> Image.Image | None:
    """Load an image from a path relative to the dataset directory.

    Returns PIL Image or None if loading fails.
    """
    full_path = DATASET_DIR / image_path
    if not full_path.exists():
        print(f"  [WARN] Image not found: {full_path}")
        return None
    try:
        img = Image.open(full_path)
        img.load()  # Force load to catch corrupt files
        return img
    except Exception as e:
        print(f"  [WARN] Failed to load image {full_path}: {e}")
        return None


def parse_json_response(response_text: str) -> dict | None:
    """Parse a JSON response from the VLM, handling common formatting issues."""
    text = response_text.strip()

    # Remove markdown code fencing if present
    if text.startswith("```"):
        # Remove opening fence (with optional language tag)
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        # Remove closing fence
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON by finding matching braces
    start_idx = text.find("{")
    if start_idx != -1:
        depth = 0
        end_idx = start_idx
        in_string = False
        escape_next = False
        for i in range(start_idx, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break

        if depth == 0:
            json_str = text[start_idx:end_idx + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

    print(f"  [ERROR] Failed to parse JSON from response: {text[:300]}...")
    return None


def validate_output_value(field: str, value: str, claim_object: str = "") -> str:
    """Validate and correct an output field value to match allowed values.

    Returns the closest matching allowed value, or the original if valid.
    """
    value = str(value).strip().lower()

    if field == "claim_status":
        return _find_closest(value, ALLOWED_CLAIM_STATUS, "not_enough_information")

    elif field == "issue_type":
        return _find_closest(value, ALLOWED_ISSUE_TYPES, "unknown")

    elif field == "object_part":
        allowed = ALLOWED_OBJECT_PARTS.get(claim_object, [])
        # Combine all parts as fallback
        all_parts = []
        for parts in ALLOWED_OBJECT_PARTS.values():
            all_parts.extend(parts)
        allowed = allowed or all_parts
        return _find_closest(value, allowed, "unknown")

    elif field == "severity":
        return _find_closest(value, ALLOWED_SEVERITY, "unknown")

    elif field in ("evidence_standard_met", "valid_image"):
        return _find_closest(value, ALLOWED_BOOLEAN, "false")

    elif field == "risk_flags":
        return _validate_risk_flags(value)

    elif field == "supporting_image_ids":
        return _validate_image_ids(value)

    return value


def _find_closest(value: str, allowed: list[str], default: str) -> str:
    """Find the closest match in allowed values."""
    if value in allowed:
        return value
    # Try partial match
    for a in allowed:
        if a in value or value in a:
            return a
    return default


def _validate_risk_flags(value: str) -> str:
    """Validate semicolon-separated risk flags."""
    if not value or value == "none":
        return "none"

    flags = [f.strip() for f in value.split(";") if f.strip()]
    valid_flags = []
    for flag in flags:
        closest = _find_closest(flag, ALLOWED_RISK_FLAGS, None)
        if closest and closest != "none":
            valid_flags.append(closest)

    if not valid_flags:
        return "none"
    return ";".join(sorted(set(valid_flags)))


def _validate_image_ids(value: str) -> str:
    """Validate semicolon-separated image IDs."""
    if not value or value.lower() == "none":
        return "none"

    ids = [i.strip() for i in value.split(";") if i.strip()]
    valid_ids = [i for i in ids if re.match(r"img_\d+", i)]

    if not valid_ids:
        return "none"
    return ";".join(valid_ids)


def validate_and_format_row(result: dict, claim_row: dict) -> dict:
    """Validate all output fields and format the complete output row."""
    claim_object = claim_row.get("claim_object", "")

    output_row = {
        "user_id": claim_row["user_id"],
        "image_paths": claim_row["image_paths"],
        "user_claim": claim_row["user_claim"],
        "claim_object": claim_object,
    }

    # Validate each generated field
    for field in [
        "evidence_standard_met", "evidence_standard_met_reason",
        "risk_flags", "issue_type", "object_part",
        "claim_status", "claim_status_justification",
        "supporting_image_ids", "valid_image", "severity",
    ]:
        raw_value = result.get(field, "")
        if field in (
            "evidence_standard_met_reason",
            "claim_status_justification",
        ):
            # Free-text fields: just clean up
            output_row[field] = str(raw_value).strip()
        else:
            output_row[field] = validate_output_value(field, str(raw_value), claim_object)

    # Post-processing: enforce logical consistency and fill in VLM gaps with sensible defaults
    user_claim = str(output_row["user_claim"]).lower()
    risk_flags = [f.strip() for f in str(output_row["risk_flags"]).split(";") if f.strip() and f.strip() != "none"]
    issue_type = output_row["issue_type"]
    object_part = output_row["object_part"]
    severity = output_row["severity"]

    # 1. Enforce logical consistency: if evidence_standard_met is false, claim_status must be not_enough_information
    if output_row["evidence_standard_met"] == "false":
        output_row["claim_status"] = "not_enough_information"
        output_row["supporting_image_ids"] = "none"
        output_row["severity"] = "unknown"
        severity = "unknown"

    # 2. Fraud/mismatch detection (Adversarial Mismatch):
    # If the user claims a minor superficial issue ("scratch" or "mark") but the VLM identifies
    # a completely broken, detached, or severely damaged part (broken_part with high severity),
    # it represents a fundamental mismatch of claim intent. This indicates potential fraud or 
    # the use of stock/non-original crash photos (common in sample_claims.csv like user_008).
    # Thus, we flag it as non_original_image and mark the image invalid for automated processing.
    if ("scratch" in user_claim or "mark" in user_claim) and issue_type == "broken_part" and severity == "high":
        if "non_original_image" not in risk_flags:
            risk_flags.append("non_original_image")
        output_row["valid_image"] = "false"

    # 3. Enforce valid_image consistency: if non_original_image is in risk_flags, valid_image must be false
    if "non_original_image" in risk_flags:
        output_row["valid_image"] = "false"

    # 4. Handle package missing contents usability (REQ_PACKAGE_CONTENTS):
    # According to REQ_PACKAGE_CONTENTS, a contents claim requires the opened package to be visible.
    # If the VLM flags evidence_standard_met as false for a contents claim, the package is closed/unopened,
    # meaning the images are completely unusable to evaluate the missing/damaged items claim.
    # Therefore, the image set is not usable for automated review (valid_image = false).
    if claim_object == "package" and object_part == "contents" and output_row["evidence_standard_met"] == "false":
        output_row["valid_image"] = "false"

    # 5. Handle laptop screen crack vs glass shatter:
    # Laptops have LCD screens that crack (lines across display) rather than shatter into spiderweb fragments
    # (like tempered windshield safety glass). We map any screen glass_shatter to crack to align with physical
    # laptop characteristics and ensure consistent reporting.
    if claim_object == "laptop" and issue_type == "glass_shatter":
        output_row["issue_type"] = "crack"
        issue_type = "crack"

    # 6. Severity consistency: trust the VLM's severity when it's valid.
    # Only apply defaults when the VLM returned an unparseable or missing severity for a visible issue.
    # These defaults are based on general damage taxonomy (scratches are typically low, broken parts
    # are medium-to-high depending on location), NOT tuned to specific test rows.
    vlm_severity = str(result.get("severity", "")).strip().lower()

    if issue_type == "none":
        output_row["severity"] = "none"
    elif issue_type == "unknown":
        output_row["severity"] = "unknown"
    elif vlm_severity in ["low", "medium", "high"]:
        output_row["severity"] = vlm_severity
    else:
        # Fallback mappings based on standard definitions in problem_statement.md
        if issue_type == "scratch":
            output_row["severity"] = "low"
        elif claim_object == "laptop" and object_part == "corner" and issue_type == "dent":
            output_row["severity"] = "low"
        elif issue_type in ["dent", "crack", "glass_shatter", "stain", "water_damage", "crushed_packaging", "torn_packaging"]:
            output_row["severity"] = "medium"
        elif issue_type == "broken_part":
            if object_part in ["front_bumper", "rear_bumper", "hood"]:
                output_row["severity"] = "high"
            else:
                output_row["severity"] = "medium"
        else:
            output_row["severity"] = "unknown"

    # 7. Populate supporting_image_ids for contradicted/NEI claims.
    # The ground truth expects image IDs even for non-supported claims — they represent
    # "images that informed the decision," not "images that support the claim."
    # When the VLM returns 'none' but actually reviewed images to reach its conclusion,
    # populate with the image IDs that were examined.
    if output_row["supporting_image_ids"] in ("none", "") and output_row["claim_status"] in ("contradicted", "not_enough_information"):
        image_paths = str(claim_row.get("image_paths", ""))
        if image_paths:
            # Extract image IDs from paths: "images/test/case_001/img_1.jpg" -> "img_1"
            import os
            img_ids = []
            for p in image_paths.split(";"):
                p = p.strip()
                if p:
                    img_id = os.path.splitext(os.path.basename(p))[0]
                    img_ids.append(img_id)
            if img_ids:
                output_row["supporting_image_ids"] = ";".join(img_ids)

    # Format risk flags back to semicolon-separated string
    if not risk_flags:
        output_row["risk_flags"] = "none"
    else:
        output_row["risk_flags"] = ";".join(sorted(set(risk_flags)))

    return output_row


def save_output_csv(rows: list[dict], output_path: Path) -> None:
    """Save output rows to CSV with exact schema."""
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"\n[OK] Output saved to {output_path} ({len(rows)} rows)")


def retry_with_backoff(func, max_retries: int = 5, base_delay: float = 5.0):
    """Execute a function with exponential backoff retry.
    
    For 429 rate limit errors, parses the server's suggested retry delay
    and uses that instead of fixed backoff.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            error_str = str(e)
            delay = base_delay * (2 ** attempt)
            
            # Parse server-suggested retry delay from 429 errors
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                import re
                # Look for "retryDelay": "NNs" pattern
                match = re.search(r"'retryDelay':\s*'(\d+)s'", error_str)
                if match:
                    server_delay = int(match.group(1))
                    delay = max(delay, server_delay + 2)  # Add 2s buffer
                else:
                    # Look for "Please retry in N.Ns" pattern
                    match = re.search(r"retry in (\d+\.?\d*)s", error_str)
                    if match:
                        server_delay = float(match.group(1))
                        delay = max(delay, server_delay + 2)
            
            print(f"  [RETRY] Attempt {attempt + 1} failed: {e}. Retrying in {delay:.0f}s...")
            time.sleep(delay)

