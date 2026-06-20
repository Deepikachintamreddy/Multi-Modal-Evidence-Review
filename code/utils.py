"""
Utility functions: CSV I/O, image loading, validation helpers.
"""
from __future__ import annotations

import base64
import csv
import io
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

import config
from models import (
    ClaimInput,
    ClaimOutput,
    EvidenceRequirement,
    SampleClaimRow,
    UserHistory,
)


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------

def read_claims(csv_path: Path = config.CLAIMS_CSV) -> List[ClaimInput]:
    """Read claims.csv (input-only columns)."""
    rows: List[ClaimInput] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(ClaimInput(**{k: v for k, v in row.items() if k in ClaimInput.model_fields}))
    return rows


def read_sample_claims(csv_path: Path = config.SAMPLE_CLAIMS_CSV) -> List[SampleClaimRow]:
    """Read sample_claims.csv (input + expected output columns)."""
    rows: List[SampleClaimRow] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(SampleClaimRow(**row))
    return rows


def read_user_history(csv_path: Path = config.USER_HISTORY_CSV) -> Dict[str, UserHistory]:
    """Read user_history.csv into a dict keyed by user_id."""
    history: Dict[str, UserHistory] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uh = UserHistory(**row)
            history[uh.user_id] = uh
    return history


def read_evidence_requirements(
    csv_path: Path = config.EVIDENCE_REQUIREMENTS_CSV,
) -> List[EvidenceRequirement]:
    """Read evidence_requirements.csv."""
    reqs: List[EvidenceRequirement] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reqs.append(EvidenceRequirement(**row))
    return reqs


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_output_csv(outputs: List[ClaimOutput], csv_path: Path) -> None:
    """Write output.csv with exact column order from problem statement."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=config.OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for out in outputs:
            writer.writerow(out.model_dump())
    print(f"  Written {len(outputs)} rows to {csv_path}")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def parse_image_paths(image_paths_str: str) -> List[str]:
    """Split semicolon-separated image paths."""
    return [p.strip() for p in image_paths_str.split(";") if p.strip()]


def image_id_from_path(image_path: str) -> str:
    """Extract image ID (filename without extension). e.g. 'images/test/case_001/img_1.jpg' -> 'img_1'."""
    return Path(image_path).stem


def load_image_as_base64(image_path: str) -> Tuple[str, str]:
    """
    Load an image file and return (base64_data, media_type).
    Resolves path relative to dataset/ directory.
    Resizes large images to save tokens.
    """
    # Resolve path relative to dataset directory
    full_path = config.DATASET_DIR / image_path
    if not full_path.exists():
        raise FileNotFoundError(f"Image not found: {full_path}")

    # Determine media type
    suffix = full_path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(suffix, "image/jpeg")

    # Load and optionally resize
    img = Image.open(full_path)
    max_dim = 1568  # Claude's recommended max dimension for vision
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    # Convert to base64
    buffer = io.BytesIO()
    fmt = "PNG" if suffix == ".png" else "JPEG"
    if img.mode == "RGBA" and fmt == "JPEG":
        img = img.convert("RGB")
    img.save(buffer, format=fmt, quality=85)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return b64, media_type


def load_all_images_for_claim(image_paths_str: str) -> List[Dict]:
    """
    Load all images for a claim, return list of dicts with:
    {image_id, image_path, base64_data, media_type, error}
    """
    paths = parse_image_paths(image_paths_str)
    results = []
    for p in paths:
        img_id = image_id_from_path(p)
        try:
            b64, mt = load_image_as_base64(p)
            results.append({
                "image_id": img_id,
                "image_path": p,
                "base64_data": b64,
                "media_type": mt,
                "error": None,
            })
        except Exception as e:
            results.append({
                "image_id": img_id,
                "image_path": p,
                "base64_data": None,
                "media_type": None,
                "error": str(e),
            })
    return results


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def closest_enum_value(value: str, allowed: frozenset) -> str:
    """Return the value if it's allowed, otherwise find closest match or 'unknown'."""
    if not value:
        return "unknown"
    v = value.strip().lower().replace(" ", "_").replace("-", "_")
    if v in allowed:
        return v
    # Try partial match
    for a in allowed:
        if v in a or a in v:
            return a
    return "unknown"


def validate_risk_flags(flags_str: str) -> str:
    """Validate semicolon-separated risk flags, remove invalid ones."""
    if not flags_str or flags_str.strip().lower() == "none":
        return "none"
    flags = [f.strip().lower().replace(" ", "_").replace("-", "_") for f in flags_str.split(";")]
    valid = [f for f in flags if f in config.RISK_FLAG_VALUES and f != "none"]
    return ";".join(valid) if valid else "none"


def validate_supporting_ids(ids_str: str, available_ids: List[str]) -> str:
    """Validate supporting image IDs against available images."""
    if not ids_str or ids_str.strip().lower() == "none":
        return "none"
    ids = [i.strip() for i in ids_str.split(";") if i.strip()]
    valid = [i for i in ids if i in available_ids]
    return ";".join(valid) if valid else "none"


def validate_object_part(part: str, claim_object: str) -> str:
    """Ensure object_part is valid for the given claim_object type."""
    allowed = config.OBJECT_PARTS.get(claim_object, [])
    if not allowed:
        return "unknown"
    p = part.strip().lower().replace(" ", "_").replace("-", "_")
    if p in allowed:
        return p
    # Try partial match
    for a in allowed:
        if p in a or a in p:
            return a
    return "unknown"
