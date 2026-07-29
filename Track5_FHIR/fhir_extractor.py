"""
DISPLACE MedAI – FHIR R4 Bundle Extraction (Track 5)
========================================================
Converts clinical dialogue summaries into HL7 FHIR R4 Bundles
using MedGemma-27B-IT served via vLLM.

Standalone usage:
    python Track5_FHIR/fhir_extractor.py \
        --summary "The patient reports chest pain..." \
        --endpoint http://n2:8000

Programmatic usage:
    from Track5_FHIR.fhir_extractor import FHIRExtractor
    extractor = FHIRExtractor(api_base="http://n2:8000/v1")
    bundle = extractor.extract(summary_text)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger("displace.fhir_extractor")

# Resolve paths relative to this file
_TRACK5_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TRACK5_DIR.parent
_DEFAULT_CONFIG = _TRACK5_DIR / "config.yaml"


def _load_system_prompt(config: dict) -> str:
    """Load the FHIR R4 extraction system prompt from the configured file."""
    prompt_file = config.get("prompt", {}).get(
        "system_prompt_file", "prompts/fhir_r4_extraction.txt"
    )
    prompt_path = _REPO_ROOT / prompt_file
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"System prompt not found: {prompt_path}\n"
            f"Expected at: {prompt_file} relative to repo root {_REPO_ROOT}"
        )
    return prompt_path.read_text(encoding="utf-8")


def _load_config(config_path: Optional[Path] = None) -> dict:
    """Load Track 5 configuration from YAML."""
    path = config_path or _DEFAULT_CONFIG
    if not path.exists():
        logger.warning(f"Config not found at {path}, using defaults")
        return {
            "model": {
                "model_id": "google/medgemma-27b-text-it",
                "api_base": "http://n2:8000/v1",
            },
            "generation": {
                "temperature": 0.0,
                "max_tokens": 8192,
                "top_p": 1.0,
                "seed": 42,
            },
            "prompt": {
                "system_prompt_file": "prompts/fhir_r4_extraction.txt",
            },
        }
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class FHIRExtractor:
    """Extract FHIR R4 Bundles from clinical summaries via MedGemma + vLLM."""

    def __init__(
        self,
        api_base: Optional[str] = None,
        model_id: Optional[str] = None,
        config_path: Optional[Path] = None,
    ):
        self._config = _load_config(config_path)
        self._api_base = (
            api_base
            or self._config.get("model", {}).get("api_base", "http://n2:8000/v1")
        )
        self._model_id = (
            model_id
            or self._config.get("model", {}).get("model_id", "google/medgemma-27b-text-it")
        )
        self._gen_params = self._config.get("generation", {})
        self._system_prompt = _load_system_prompt(self._config)
        self._client = None

    def _get_client(self):
        """Lazy-init the OpenAI client pointing at the vLLM server."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key="EMPTY",  # vLLM doesn't need a real key
                base_url=self._api_base,
            )
        return self._client

    def extract(self, summary: str, timeout: float = 120.0) -> Dict[str, Any]:
        """
        Convert a clinical summary into a FHIR R4 Bundle.

        Args:
            summary: Clinical dialogue summary text from Stage 4.
            timeout: Request timeout in seconds.

        Returns:
            {
                "fhir_bundle": dict,      # The parsed FHIR R4 Bundle
                "raw_response": str,       # Raw model output
                "elapsed_s": float,        # Wall-clock time
                "model_id": str,           # Model used
                "valid": bool,             # Whether validation passed
                "validation_errors": list, # Any validation issues
            }
        """
        start = time.time()
        client = self._get_client()

        # Build the user message: system prompt already ends with "INPUT" section
        user_content = summary

        logger.info(
            f"Sending summary ({len(summary)} chars) to {self._model_id} "
            f"at {self._api_base}"
        )

        try:
            response = client.chat.completions.create(
                model=self._model_id,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=self._gen_params.get("temperature", 0.0),
                max_tokens=self._gen_params.get("max_tokens", 8192),
                top_p=self._gen_params.get("top_p", 1.0),
                seed=self._gen_params.get("seed", 42),
                timeout=timeout,
            )
        except Exception as exc:
            elapsed = time.time() - start
            logger.error(f"vLLM request failed after {elapsed:.1f}s: {exc}")
            raise RuntimeError(
                f"Failed to reach vLLM at {self._api_base}: {exc}"
            ) from exc

        raw_text = response.choices[0].message.content.strip()
        elapsed = time.time() - start

        logger.info(f"Received response ({len(raw_text)} chars, {elapsed:.1f}s)")

        # Parse and validate
        bundle, valid, errors = self._parse_and_validate(raw_text)

        return {
            "fhir_bundle": bundle,
            "raw_response": raw_text,
            "elapsed_s": elapsed,
            "model_id": self._model_id,
            "valid": valid,
            "validation_errors": errors,
        }

    def _parse_and_validate(self, raw: str) -> tuple:
        """
        Parse raw model output as JSON and validate FHIR structure.

        Returns: (bundle_dict_or_None, is_valid, error_list)
        """
        errors = []

        # Strip markdown code fences if present (model shouldn't, but be safe)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        # Attempt JSON parse
        try:
            bundle = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}")
            errors.append(f"Invalid JSON: {e}")
            return None, False, errors

        # Validate top-level structure
        if not isinstance(bundle, dict):
            errors.append("Response is not a JSON object")
            return None, False, errors

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

        # Check required resource types
        from Track5_FHIR.validate_bundle import validate_bundle
        validation_errors = validate_bundle(bundle, self._config)
        errors.extend(validation_errors)

        valid = len(errors) == 0
        if valid:
            logger.info(f"FHIR bundle valid: {len(entries)} entries")
        else:
            logger.warning(f"FHIR bundle has {len(errors)} validation issues")

        return bundle, valid, errors


def main() -> int:
    """CLI entry point for standalone FHIR extraction."""
    parser = argparse.ArgumentParser(
        description="Extract FHIR R4 Bundles from clinical summaries using MedGemma."
    )
    parser.add_argument(
        "--summary", "-s",
        help="Clinical summary text (or path to a .txt file)",
        required=True,
    )
    parser.add_argument(
        "--endpoint", "-e",
        help="vLLM server base URL (default: from config.yaml)",
        default=None,
    )
    parser.add_argument(
        "--model", "-m",
        help="Model ID on the vLLM server",
        default=None,
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to config.yaml",
        default=None,
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for the FHIR bundle JSON (default: stdout)",
        default=None,
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load summary
    summary_path = Path(args.summary)
    if summary_path.exists() and summary_path.is_file():
        summary = summary_path.read_text(encoding="utf-8")
    else:
        summary = args.summary

    if not summary.strip():
        print("ERROR: Summary text is empty.", file=sys.stderr)
        return 1

    # Build endpoint URL
    api_base = None
    if args.endpoint:
        api_base = args.endpoint.rstrip("/")
        if not api_base.endswith("/v1"):
            api_base += "/v1"

    config_path = Path(args.config) if args.config else None

    extractor = FHIRExtractor(
        api_base=api_base,
        model_id=args.model,
        config_path=config_path,
    )

    result = extractor.extract(summary)

    # Output
    output_json = json.dumps(result["fhir_bundle"], indent=2) if result["fhir_bundle"] else result["raw_response"]

    if args.output:
        Path(args.output).write_text(output_json, encoding="utf-8")
        print(f"Written to {args.output}")
    else:
        print(output_json)

    # Print validation status to stderr
    if result["valid"]:
        print(f"\n✓ Valid FHIR R4 Bundle ({result['elapsed_s']:.1f}s)", file=sys.stderr)
    else:
        print(f"\n✗ Validation issues ({result['elapsed_s']:.1f}s):", file=sys.stderr)
        for err in result["validation_errors"]:
            print(f"  - {err}", file=sys.stderr)

    return 0 if result["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
