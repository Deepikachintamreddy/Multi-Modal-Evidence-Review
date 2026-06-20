"""
Pydantic data models for claim inputs, outputs, and intermediate structures.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, Optional, List


class ClaimInput(BaseModel):
    """One row from claims.csv (input columns only)."""
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str


class SampleClaimRow(BaseModel):
    """One row from sample_claims.csv (input + expected output columns)."""
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: str
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: str
    severity: str


class UserHistory(BaseModel):
    """One row from user_history.csv."""
    user_id: str
    past_claim_count: int
    accept_claim: int
    manual_review_claim: int
    rejected_claim: int
    last_90_days_claim_count: int
    history_flags: str
    history_summary: str


class EvidenceRequirement(BaseModel):
    """One row from evidence_requirements.csv."""
    requirement_id: str
    claim_object: str
    applies_to: str
    minimum_image_evidence: str


class ImageNote(BaseModel):
    """Note about a specific image."""
    image_id: str
    note: str


class VLMAnalysis(BaseModel):
    """Structured output expected from the VLM call."""
    evidence_standard_met: bool = True
    evidence_standard_met_reason: str = ""
    risk_flags: str = "none"
    issue_type: str = "unknown"
    object_part: str = "unknown"
    claim_status: str = "not_enough_information"
    claim_status_justification: str = ""
    supporting_image_ids: str = "none"
    valid_image: bool = True
    severity: str = "unknown"
    per_image_notes: Optional[List[ImageNote]] = None


class ClaimOutput(BaseModel):
    """One row for output.csv — final prediction."""
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: str  # "true" / "false"
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: str  # "true" / "false"
    severity: str


class ExtractionResult(BaseModel):
    """Output of Stage-1 claim extraction (two_stage strategy only)."""
    claimed_damage_type: str = ""
    claimed_object_part: str = ""
    claimed_object: str = ""
    is_multi_part_claim: bool = False
    secondary_parts: List[str] = Field(default_factory=list)
    conversation_red_flags: List[str] = Field(default_factory=list)
    specified_color_or_side: Optional[str] = None
    claim_summary: str = ""


class TokenUsage(BaseModel):
    """Tracks API token usage for operational reporting."""
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    images_processed: int = 0
