"""
Configuration constants, allowed values, and paths for the claim verification system.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CODE_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
CLAIMS_CSV = DATASET_DIR / "claims.csv"
SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV = DATASET_DIR / "evidence_requirements.csv"
IMAGES_DIR = DATASET_DIR / "images"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "output.csv"
PROMPTS_DIR = CODE_DIR / "prompts"

# ---------------------------------------------------------------------------
# API Configuration — Google Gemini (FREE tier)
# ---------------------------------------------------------------------------
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
# Support multiple comma-separated keys or GOOGLE_API_KEY_N env variables
GOOGLE_API_KEYS = [k.strip() for k in GOOGLE_API_KEY.split(",") if k.strip()]
if not GOOGLE_API_KEYS:
    for i in range(1, 10):
        val = os.getenv(f"GOOGLE_API_KEY_{i}")
        if val:
            GOOGLE_API_KEYS.append(val.strip())

if not GOOGLE_API_KEYS and GOOGLE_API_KEY:
    GOOGLE_API_KEYS = [GOOGLE_API_KEY.strip()]

GOOGLE_API_KEY = GOOGLE_API_KEYS[0] if GOOGLE_API_KEYS else ""

MODEL_VISION = os.getenv("MODEL_VISION", "gemini-2.0-flash-lite")
MODEL_FAST = os.getenv("MODEL_FAST", "gemini-2.0-flash-lite")

# Fallback model rotation (tried in order when current model hits quota)
MODEL_ROTATION = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]

MAX_RETRIES = 5
RETRY_DELAY_BASE = 10  # seconds, exponential backoff

# ---------------------------------------------------------------------------
# Allowed enum values (from problem_statement.md)
# ---------------------------------------------------------------------------
CLAIM_STATUS_VALUES = frozenset([
    "supported", "contradicted", "not_enough_information",
])

ISSUE_TYPE_VALUES = frozenset([
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
])

SEVERITY_VALUES = frozenset(["none", "low", "medium", "high", "unknown"])

RISK_FLAG_VALUES = frozenset([
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
])

CAR_PARTS = [
    "front_bumper", "rear_bumper", "door", "hood", "windshield",
    "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
    "body", "unknown",
]
LAPTOP_PARTS = [
    "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
    "port", "base", "body", "unknown",
]
PACKAGE_PARTS = [
    "box", "package_corner", "package_side", "seal", "label",
    "contents", "item", "unknown",
]

OBJECT_PARTS = {
    "car": CAR_PARTS,
    "laptop": LAPTOP_PARTS,
    "package": PACKAGE_PARTS,
}

# Output column order (MUST match problem_statement.md exactly)
OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids",
    "valid_image", "severity",
]
