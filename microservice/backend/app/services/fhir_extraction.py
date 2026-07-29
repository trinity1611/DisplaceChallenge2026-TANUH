"""
DISPLACE MedAI – FHIR R4 Extraction Service (Track 5)
=========================================================
Calls MedGemma-27B-IT via vLLM to convert clinical summaries
into HL7 FHIR R4 Bundles.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from backend.app.config import settings

logger = logging.getLogger("displace.fhir_extraction")

# Resolve prompt path
_BASELINES_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_PROMPT_PATH = _BASELINES_ROOT / "prompts" / "fhir_r4_extraction.txt"


class FHIRExtractionService:
    """FHIR R4 Bundle extraction using MedGemma via vLLM (Track 5)."""

    def __init__(self):
        self._client = None
        self._system_prompt: Optional[str] = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Initialize the OpenAI client and load the system prompt."""
        if self._loaded:
            logger.info("FHIR extraction service already loaded, skipping.")
            return

        # Load system prompt
        prompt_path = _PROMPT_PATH
        if not prompt_path.exists():
            # Try alternative path
            prompt_path = settings.baselines_root / "prompts" / "fhir_r4_extraction.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"FHIR extraction prompt not found at {prompt_path}. "
                f"Expected at: prompts/fhir_r4_extraction.txt"
            )

        self._system_prompt = prompt_path.read_text(encoding="utf-8")
        logger.info(f"Loaded FHIR system prompt ({len(self._system_prompt)} chars)")

        # Initialize OpenAI client pointing at vLLM
        from openai import OpenAI

        self._client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require a real API key
            base_url=settings.vllm_api_base,
        )

        self._loaded = True
        logger.info(
            f"FHIR extraction service ready "
            f"(model: {settings.medgemma_model_id}, "
            f"endpoint: {settings.vllm_api_base})"
        )

    def unload(self) -> None:
        """Release resources."""
        if self._client is not None:
            self._client = None
            self._system_prompt = None
            self._loaded = False
            logger.info("FHIR extraction service unloaded")

    def run(self, summary: str) -> Dict[str, Any]:
        """
        Convert a clinical summary into a FHIR R4 Bundle.

        Args:
            summary: Clinical dialogue summary text from Track 4.

        Returns:
            {
                "fhir_bundle": dict or None,
                "fhir_json": str,
                "valid": bool,
                "validation_errors": list,
                "elapsed_s": float,
            }
        """
        start = time.time()

        if not self._loaded:
            self.load()

        logger.info(f"Extracting FHIR bundle from summary ({len(summary)} chars)")

        try:
            response = self._client.chat.completions.create(
                model=settings.medgemma_model_id,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": summary},
                ],
                temperature=0.0,
                max_tokens=2048,
                top_p=1.0,
                seed=42,
            )
        except Exception as exc:
            elapsed = time.time() - start
            logger.error(f"vLLM request failed after {elapsed:.1f}s: {exc}")
            return {
                "fhir_bundle": None,
                "fhir_json": "",
                "valid": False,
                "validation_errors": [f"vLLM request failed: {exc}"],
                "elapsed_s": elapsed,
            }

        raw_text = response.choices[0].message.content.strip()
        elapsed = time.time() - start

        logger.info(f"Received FHIR response ({len(raw_text)} chars, {elapsed:.1f}s)")

        # Parse and validate
        bundle, valid, errors = self._parse_and_validate(raw_text)

        fhir_json = (
            json.dumps(bundle, indent=2) if bundle else raw_text
        )

        return {
            "fhir_bundle": bundle,
            "fhir_json": fhir_json,
            "valid": valid,
            "validation_errors": errors,
            "elapsed_s": elapsed,
        }

    def _parse_and_validate(self, raw: str):
        """Parse raw model output and validate FHIR structure."""
        errors = []

        # Strip code fences if model accidentally included them
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            bundle = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}")
            return None, False, [f"Invalid JSON: {e}"]

        if not isinstance(bundle, dict):
            return None, False, ["Response is not a JSON object"]

        # Top-level checks
        if bundle.get("resourceType") != "Bundle":
            errors.append(
                f"resourceType is '{bundle.get('resourceType')}', expected 'Bundle'"
            )
        if bundle.get("type") != "collection":
            errors.append(
                f"Bundle.type is '{bundle.get('type')}', expected 'collection'"
            )
        if "timestamp" not in bundle:
            errors.append("Bundle.timestamp is missing")

        entries = bundle.get("entry", [])
        if not isinstance(entries, list) or len(entries) == 0:
            errors.append("Bundle.entry is empty or missing")
            return bundle, False, errors

        # Check required resource types
        found_types = set()
        known_urls = set()
        all_refs = []
        seen_ids = set()
        uuid_re = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )

        for i, entry in enumerate(entries):
            resource = entry.get("resource", {})
            full_url = entry.get("fullUrl", "")
            res_type = resource.get("resourceType", "")
            res_id = resource.get("id", "")

            if res_type:
                found_types.add(res_type)
            if full_url:
                known_urls.add(full_url)
            if res_id:
                if not uuid_re.match(res_id):
                    errors.append(
                        f"entry[{i}] ({res_type}): id '{res_id}' is not a valid UUID"
                    )
                if res_id in seen_ids:
                    errors.append(
                        f"entry[{i}] ({res_type}): duplicate id '{res_id}'"
                    )
                seen_ids.add(res_id)

            # Collect references
            self._collect_refs(resource, f"entry[{i}]", all_refs)

        for req in ["Patient", "Encounter", "Condition", "Observation"]:
            if req not in found_types:
                errors.append(f"Required resource '{req}' missing from bundle")

        # Verify references resolve
        for loc, ref_val in all_refs:
            if ref_val.startswith("urn:uuid:") and ref_val not in known_urls:
                errors.append(f"{loc}: unresolved reference '{ref_val}'")

        valid = len(errors) == 0
        if valid:
            logger.info(f"FHIR bundle valid: {len(entries)} resources")
        else:
            logger.warning(f"FHIR bundle has {len(errors)} validation issues")

        return bundle, valid, errors

    def _collect_refs(self, obj, path, refs):
        """Recursively collect all references."""
        if isinstance(obj, dict):
            if "reference" in obj and isinstance(obj["reference"], str):
                refs.append((path, obj["reference"]))
            for k, v in obj.items():
                self._collect_refs(v, f"{path}.{k}", refs)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                self._collect_refs(item, f"{path}[{i}]", refs)


# Singleton instance
fhir_extraction_service = FHIRExtractionService()
