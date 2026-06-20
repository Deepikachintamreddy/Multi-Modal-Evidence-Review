"""
Image analyzer module using Google Gemini VLM.
Handles single-pass and two-pass claim analysis strategies.
"""

import os
import time
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

from config import DATASET_DIR, GEMINI_MODEL, REQUEST_DELAY, MAX_RETRIES, RETRY_DELAY_BASE
from prompt_templates import (
    build_claim_prompt,
    build_few_shot_examples,
    build_system_prompt,
    build_two_pass_image_prompt,
)
from utils import (
    get_image_id,
    load_image,
    parse_image_paths,
    parse_json_response,
    retry_with_backoff,
)


class ImageAnalyzer:
    """Analyzes damage claim images using Google Gemini VLM."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        """Initialize the analyzer with API credentials.

        Args:
            api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
            model: Model name. Defaults to config.GEMINI_MODEL.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "No Gemini API key provided. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.model = model or GEMINI_MODEL
        self.client = genai.Client(api_key=self.api_key)

        # Build system prompt once
        self._system_prompt = build_system_prompt() + "\n\n" + build_few_shot_examples()

        # Tracking for operational analysis
        self.stats = {
            "total_calls": 0,
            "total_images": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "errors": 0,
            "retries": 0,
            "start_time": time.time(),
        }

    def analyze_claim(
        self,
        claim_row: dict,
        user_history: dict,
        evidence_requirements: list[dict],
    ) -> dict | None:
        """Analyze a single claim using single-pass VLM strategy.

        Args:
            claim_row: Dict with user_id, image_paths, user_claim, claim_object
            user_history: User history dict from user_history.csv
            evidence_requirements: List of relevant evidence requirement dicts

        Returns:
            Dict with all output fields, or None on failure
        """
        claim_object = claim_row["claim_object"]
        user_claim = claim_row["user_claim"]
        image_paths_str = claim_row["image_paths"]

        # Parse image paths and IDs
        image_paths = parse_image_paths(image_paths_str)
        image_ids = [get_image_id(p) for p in image_paths]

        # Load images
        images = []
        loaded_ids = []
        for path, img_id in zip(image_paths, image_ids):
            img = load_image(path)
            if img is not None:
                images.append(img)
                loaded_ids.append(img_id)

        if not images:
            print(f"  [WARN] No images loaded for claim. Returning default not_enough_info.")
            return self._default_insufficient_result(image_ids)

        # Build the analysis prompt
        prompt = build_claim_prompt(
            claim_object=claim_object,
            user_claim=user_claim,
            image_ids=image_ids,
            user_history=user_history,
            evidence_requirements=evidence_requirements,
        )

        # Build content parts: images first, then the prompt text
        content_parts = []
        for img in images:
            content_parts.append(img)
        content_parts.append(prompt)

        # Make the API call with retry
        def _call_api():
            response = self.client.models.generate_content(
                model=self.model,
                contents=content_parts,
                config=types.GenerateContentConfig(
                    system_instruction=self._system_prompt,
                    temperature=0.1,  # Low temperature for consistency
                    max_output_tokens=1024,
                ),
            )
            return response

        try:
            response = retry_with_backoff(_call_api, max_retries=MAX_RETRIES, base_delay=RETRY_DELAY_BASE)
            self.stats["total_calls"] += 1
            self.stats["total_images"] += len(images)

            # Track token usage
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                self.stats["total_input_tokens"] += getattr(
                    response.usage_metadata, "prompt_token_count", 0
                )
                self.stats["total_output_tokens"] += getattr(
                    response.usage_metadata, "candidates_token_count", 0
                )

            # Parse response
            response_text = response.text
            result = parse_json_response(response_text)

            if result is None:
                print(f"  [ERROR] Could not parse VLM response")
                self.stats["errors"] += 1
                return self._default_insufficient_result(image_ids)

            # Rate limiting
            time.sleep(REQUEST_DELAY)

            return result

        except Exception as e:
            print(f"  [ERROR] API call failed: {e}")
            self.stats["errors"] += 1
            return self._default_insufficient_result(image_ids)

    def analyze_claim_two_pass(
        self,
        claim_row: dict,
        user_history: dict,
        evidence_requirements: list[dict],
    ) -> dict | None:
        """Strategy B: Two-pass analysis.

        Pass 1: Analyze each image independently.
        Pass 2: Synthesize image analyses with claim context.
        """
        claim_object = claim_row["claim_object"]
        user_claim = claim_row["user_claim"]
        image_paths_str = claim_row["image_paths"]

        image_paths = parse_image_paths(image_paths_str)
        image_ids = [get_image_id(p) for p in image_paths]

        # Pass 1: Individual image analysis
        image_analyses = []
        for path, img_id in zip(image_paths, image_ids):
            img = load_image(path)
            if img is None:
                continue

            prompt = build_two_pass_image_prompt(claim_object, img_id)

            def _call_pass1():
                return self.client.models.generate_content(
                    model=self.model,
                    contents=[img, prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=512,
                    ),
                )

            try:
                response = retry_with_backoff(_call_pass1, max_retries=2)
                self.stats["total_calls"] += 1
                self.stats["total_images"] += 1

                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    self.stats["total_input_tokens"] += getattr(
                        response.usage_metadata, "prompt_token_count", 0
                    )
                    self.stats["total_output_tokens"] += getattr(
                        response.usage_metadata, "candidates_token_count", 0
                    )

                analysis = parse_json_response(response.text)
                if analysis:
                    analysis["image_id"] = img_id
                    image_analyses.append(analysis)

                time.sleep(REQUEST_DELAY)
            except Exception as e:
                print(f"  [WARN] Pass 1 failed for {img_id}: {e}")

        if not image_analyses:
            return self._default_insufficient_result(image_ids)

        # Pass 2: Synthesize with full context
        analyses_text = "\n".join(
            [f"Image {a.get('image_id', '?')}: {a}" for a in image_analyses]
        )

        history_text = ""
        if user_history:
            history_text = f"User: {user_history.get('user_id', '?')}, "
            history_text += f"past claims: {user_history.get('past_claim_count', 0)}, "
            history_text += f"rejected: {user_history.get('rejected_claim', 0)}, "
            history_text += f"flags: {user_history.get('history_flags', 'none')}, "
            history_text += f"summary: {user_history.get('history_summary', '')}"

        synthesis_prompt = f"""Based on the image analyses below, synthesize a final claim verdict.

## IMAGE ANALYSES
{analyses_text}

## CLAIM CONTEXT
Object: {claim_object}
User Conversation: {user_claim}
User History: {history_text}

Respond with ONLY valid JSON matching the output schema from the system prompt.
"""

        def _call_pass2():
            return self.client.models.generate_content(
                model=self.model,
                contents=[synthesis_prompt],
                config=types.GenerateContentConfig(
                    system_instruction=self._system_prompt,
                    temperature=0.1,
                    max_output_tokens=1024,
                ),
            )

        try:
            response = retry_with_backoff(_call_pass2, max_retries=2)
            self.stats["total_calls"] += 1

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                self.stats["total_input_tokens"] += getattr(
                    response.usage_metadata, "prompt_token_count", 0
                )
                self.stats["total_output_tokens"] += getattr(
                    response.usage_metadata, "candidates_token_count", 0
                )

            result = parse_json_response(response.text)
            time.sleep(REQUEST_DELAY)
            return result

        except Exception as e:
            print(f"  [ERROR] Pass 2 failed: {e}")
            self.stats["errors"] += 1
            return self._default_insufficient_result(image_ids)

    def _default_insufficient_result(self, image_ids: list[str]) -> dict:
        """Return a default result when analysis fails."""
        return {
            "evidence_standard_met": "false",
            "evidence_standard_met_reason": "Unable to analyze the submitted images.",
            "risk_flags": "manual_review_required",
            "issue_type": "unknown",
            "object_part": "unknown",
            "claim_status": "not_enough_information",
            "claim_status_justification": "The images could not be analyzed automatically. Manual review is required.",
            "supporting_image_ids": "none",
            "valid_image": "false",
            "severity": "unknown",
        }

    def get_stats(self) -> dict:
        """Get operational statistics."""
        elapsed = time.time() - self.stats["start_time"]
        stats = dict(self.stats)
        stats["elapsed_seconds"] = round(elapsed, 1)
        stats["avg_seconds_per_call"] = (
            round(elapsed / stats["total_calls"], 1) if stats["total_calls"] > 0 else 0
        )
        return stats
