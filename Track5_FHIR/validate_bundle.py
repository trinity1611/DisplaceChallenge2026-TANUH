"""
DISPLACE MedAI – FHIR R4 Bundle Validator (Track 5)
========================================================
Structural validation of FHIR R4 Bundles without external dependencies.
Checks resource presence, UUID format, reference resolution, and
data-type correctness per the extraction prompt's rules.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# RFC-4122 UUID pattern: 8-4-4-4-12 lowercase hex
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

_DEFAULT_REQUIRED_RESOURCES = ["Patient", "Encounter", "Condition", "Observation"]


def validate_bundle(
    bundle: Optional[Dict[str, Any]],
    config: Optional[dict] = None,
) -> List[str]:
    """
    Validate a FHIR R4 Bundle for structural correctness.

    Returns a list of error strings. Empty list = valid.
    """
    if bundle is None:
        return ["Bundle is None"]

    errors = []
    val_config = (config or {}).get("validation", {})
    required_resources = val_config.get(
        "required_resources", _DEFAULT_REQUIRED_RESOURCES
    )
    check_refs = val_config.get("check_references", True)
    check_uuids = val_config.get("check_uuids", True)

    # --- Top-level checks ---
    if bundle.get("resourceType") != "Bundle":
        errors.append(
            f"resourceType is '{bundle.get('resourceType')}', expected 'Bundle'"
        )
    if bundle.get("type") != "collection":
        errors.append(
            f"Bundle.type is '{bundle.get('type')}', expected 'collection'"
        )
    if not bundle.get("timestamp"):
        errors.append("Bundle.timestamp is missing")

    entries = bundle.get("entry", [])
    if not isinstance(entries, list):
        errors.append("Bundle.entry is not a list")
        return errors
    if len(entries) == 0:
        errors.append("Bundle.entry is empty")
        return errors

    # Collect resource types and fullUrls
    found_types: set = set()
    known_urls: set = set()
    all_references: list = []
    seen_ids: set = set()

    for i, entry in enumerate(entries):
        resource = entry.get("resource", {})
        full_url = entry.get("fullUrl", "")
        res_type = resource.get("resourceType", "")
        res_id = resource.get("id", "")

        if not res_type:
            errors.append(f"entry[{i}]: missing resourceType")
        else:
            found_types.add(res_type)

        if not full_url:
            errors.append(f"entry[{i}] ({res_type}): missing fullUrl")
        elif not full_url.startswith("urn:uuid:"):
            errors.append(
                f"entry[{i}] ({res_type}): fullUrl '{full_url}' "
                f"does not start with 'urn:uuid:'"
            )
        else:
            known_urls.add(full_url)

        if not res_id:
            errors.append(f"entry[{i}] ({res_type}): missing id")
        else:
            if check_uuids and not _UUID_RE.match(res_id):
                errors.append(
                    f"entry[{i}] ({res_type}): id '{res_id}' is not a valid RFC-4122 UUID"
                )
            expected_url = f"urn:uuid:{res_id}"
            if full_url and full_url != expected_url:
                errors.append(
                    f"entry[{i}] ({res_type}): fullUrl '{full_url}' "
                    f"does not match id '{res_id}'"
                )
            if res_id in seen_ids:
                errors.append(f"entry[{i}] ({res_type}): duplicate id '{res_id}'")
            seen_ids.add(res_id)

        # Collect all references from this resource
        _collect_references(resource, f"entry[{i}] ({res_type})", all_references)

    # --- Required resource types ---
    for req in required_resources:
        if req not in found_types:
            errors.append(f"Required resource type '{req}' is missing from the bundle")

    # --- Reference resolution ---
    if check_refs:
        for ref_location, ref_value in all_references:
            if ref_value.startswith("urn:uuid:") and ref_value not in known_urls:
                errors.append(
                    f"{ref_location}: reference '{ref_value}' does not resolve "
                    f"to any entry in the bundle"
                )

    return errors


def _collect_references(
    obj: Any, path: str, refs: list
) -> None:
    """Recursively find all { "reference": "urn:uuid:..." } in a resource."""
    if isinstance(obj, dict):
        if "reference" in obj and isinstance(obj["reference"], str):
            refs.append((path, obj["reference"]))
        for key, value in obj.items():
            _collect_references(value, f"{path}.{key}", refs)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _collect_references(item, f"{path}[{i}]", refs)
