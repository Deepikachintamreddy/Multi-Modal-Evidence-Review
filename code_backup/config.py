"""
Configuration module for Multi-Modal Evidence Review system.
Contains all constants, allowed values, and path configurations.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from code directory
load_dotenv(Path(__file__).resolve().parent / ".env")

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Base paths - resolve relative to the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
CODE_DIR = REPO_ROOT / "code"

# Input files
CLAIMS_CSV = DATASET_DIR / "claims.csv"
SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV = DATASET_DIR / "evidence_requirements.csv"

# Output - write to repo root for submission
OUTPUT_CSV = REPO_ROOT / "output.csv"

# Images base directory
IMAGES_DIR = DATASET_DIR / "images"

# ============================================================================
# API CONFIGURATION
# ============================================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.1-flash-lite"

# Rate limiting
MAX_RETRIES = 5
RETRY_DELAY_BASE = 10  # seconds, exponential backoff
REQUEST_DELAY = 4.5    # seconds between requests to avoid rate limits

# ============================================================================
# OUTPUT SCHEMA
# ============================================================================

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

# ============================================================================
# ALLOWED VALUES
# ============================================================================

ALLOWED_CLAIM_STATUS = ["supported", "contradicted", "not_enough_information"]

ALLOWED_ISSUE_TYPES = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
]

ALLOWED_CAR_PARTS = [
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender",
    "quarter_panel", "body", "unknown",
]

ALLOWED_LAPTOP_PARTS = [
    "screen", "keyboard", "trackpad", "hinge", "lid",
    "corner", "port", "base", "body", "unknown",
]

ALLOWED_PACKAGE_PARTS = [
    "box", "package_corner", "package_side", "seal",
    "label", "contents", "item", "unknown",
]

ALLOWED_OBJECT_PARTS = {
    "car": ALLOWED_CAR_PARTS,
    "laptop": ALLOWED_LAPTOP_PARTS,
    "package": ALLOWED_PACKAGE_PARTS,
}

ALLOWED_RISK_FLAGS = [
    "none", "blurry_image", "cropped_or_obstructed",
    "low_light_or_glare", "wrong_angle", "wrong_object",
    "wrong_object_part", "damage_not_visible", "claim_mismatch",
    "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk",
    "manual_review_required",
]

ALLOWED_SEVERITY = ["none", "low", "medium", "high", "unknown"]

ALLOWED_BOOLEAN = ["true", "false"]
