"""Pre-download model weights to the shared cluster cache.

Run this on the login node (which has internet) before submitting a GPU job,
so the compute node never has to reach the network.

    export HF_TOKEN=hf_xxxxx          # or: huggingface-cli login
    python cluster/download_model.py --model medgemma_27b_vllm

Credentials come from the environment only. The previous version of this file
contained a hardcoded Hugging Face token; if you have not already, revoke it at
https://huggingface.co/settings/tokens - it must be treated as compromised.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fhirbench.config import load_config  # noqa: E402


def _cache_dir_for(model_id: str) -> str:
    """The hub cache folder name for a repo id, e.g. models--google--medgemma."""
    return "models--" + model_id.replace("/", "--")


def _explain(exc: Exception, model_id: str) -> None:
    """Turn the two failures that actually happen into actionable advice."""
    name = type(exc).__name__
    message = str(exc)
    hub = os.environ.get("HF_HOME", "~/.cache/huggingface").rstrip("/") + "/hub"
    folder = f"{hub}/{_cache_dir_for(model_id)}"

    if "GatedRepoError" in name or "403" in message or "restricted" in message:
        print("  -> This repo is GATED. Accept the licence while signed in as the")
        print(f"     account that owns HF_TOKEN: https://huggingface.co/{model_id}")
        print("     Then re-run. Access is granted per-account, not per-token.")
        print(f"  -> Delete the half-written cache first: rm -rf {folder}")
        return

    if "hex hash" in message or "Reconstruct" in message or "xet" in message.lower():
        print("  -> The Xet transfer backend cannot resume an inconsistent cache")
        print("     (this usually follows an earlier failed or interrupted download).")
        print(f"     rm -rf {folder}")
        print("     If it recurs, fall back to the plain HTTP backend:")
        print("       export HF_HUB_DISABLE_XET=1")
        return

    if "401" in message or "Unauthorized" in message or "token" in message.lower():
        print("  -> HF_TOKEN is missing, expired or revoked. Create a new one at")
        print("     https://huggingface.co/settings/tokens and export it.")
        return

    if "Connection" in name or "Timeout" in name or "Resolve" in message:
        print("  -> Network failure. This script must run on the MASTER node;")
        print("     compute nodes are on a private network with no internet.")
        return

    print(f"  -> If a partial download is suspected: rm -rf {folder}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache model weights for offline use.")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="model key from config/pipeline.yaml (repeatable; default: all local models)",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    keys = args.model or [
        key
        for key, spec in config.models.items()
        if spec.get("backend") in ("vllm", "transformers")
    ]
    if not keys:
        print("Nothing to download: no local (vllm/transformers) models configured.")
        return 0

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "WARNING: HF_TOKEN is not set. Gated repositories such as\n"
            "         google/medgemma-27b-it will fail with 401/403.\n"
            "         Export HF_TOKEN or run 'huggingface-cli login' first.\n"
        )

    from huggingface_hub import snapshot_download

    cache_dir = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    print(f"Cache location: {cache_dir or '~/.cache/huggingface (default)'}\n")

    failures = []
    for key in keys:
        spec = config.model(key)
        model_id = spec["model_id"]
        revision = spec.get("revision") or "main"
        print("=" * 70)
        print(f"Downloading {key}: {model_id}@{revision}")
        print("=" * 70)
        try:
            path = snapshot_download(
                repo_id=model_id,
                revision=revision,
                token=token,
                # Weights only: skip duplicate formats to save NFS quota.
                ignore_patterns=["*.msgpack", "*.h5", "*.onnx", "*.tflite"],
            )
            print(f"  cached at {path}\n")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}\n")
            _explain(exc, model_id)
            failures.append(key)

    if failures:
        print(f"Failed: {', '.join(failures)}")
        return 1
    print("All requested models are cached. Compute nodes can now run offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
