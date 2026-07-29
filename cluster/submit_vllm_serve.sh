#!/bin/bash
# =============================================================================
# Launch vLLM as an OpenAI-compatible API server on a GPU compute node.
#
# This script serves MedGemma-27B-IT for FHIR R4 extraction.
# The server listens on port 8000 and is accessible from the master node
# and other compute nodes via the private network.
#
# Usage:
#   sbatch --nodelist=n2 cluster/submit_vllm_serve.sh
#   sbatch --nodelist=n1 cluster/submit_vllm_serve.sh  # also works, but slower
#
# The server stays running until the time limit or scancel.
# Test it from the master node:
#   curl http://n2:8000/v1/models
#   curl http://n2:8000/health
#
# Weights must already be cached by running on the MASTER node:
#   export HF_TOKEN=hf_xxxxx
#   huggingface-cli download google/medgemma-27b-text-it
# =============================================================================
#SBATCH --job-name=vllm_serve
#SBATCH --output=logs/vllm-%j.out
#SBATCH --error=logs/vllm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=normal
#SBATCH --time=12:00:00

set -uo pipefail

# Model to serve - override with environment variable if needed
MODEL_ID="${VLLM_MODEL_ID:-google/medgemma-27b-text-it}"
PORT="${VLLM_PORT:-8000}"
# Max model length for MedGemma-27B context window
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

# --- Locate the repository --------------------------------------------------
find_repo_root() {
  local candidate
  for candidate in \
      "${FHIR_BENCH_ROOT:-}" \
      "${SLURM_SUBMIT_DIR:-}" \
      "$PWD" \
      "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" \
      "$HOME/DISPLACE-2026-Baselines" ; do
    if [[ -n "$candidate" && -d "$candidate/Track5_FHIR" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  # Fallback: look for cluster/ directory
  for candidate in \
      "${SLURM_SUBMIT_DIR:-}" \
      "$PWD" \
      "$HOME/DISPLACE-2026-Baselines" ; do
    if [[ -n "$candidate" && -d "$candidate/cluster" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

REPO_ROOT="$(find_repo_root || true)"
if [[ -z "$REPO_ROOT" ]]; then
  echo "ERROR: could not locate the repository." >&2
  echo "Submit from the repo root, or set FHIR_BENCH_ROOT:" >&2
  echo "  cd ~/DISPLACE-2026-Baselines && sbatch cluster/submit_vllm_serve.sh" >&2
  exit 1
fi
cd "$REPO_ROOT"
mkdir -p logs

# --- Software stack (LMod) --------------------------------------------------
module purge

load_conda() {
  local candidate
  for candidate in Miniforge3 miniforge3 miniconda miniforge anaconda conda; do
    if module load "$candidate" >/dev/null 2>&1 && command -v conda >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

CONDA_MODULE="$(load_conda || true)"
if [[ -n "$CONDA_MODULE" ]]; then
  echo "loaded conda module: $CONDA_MODULE"
else
  echo "WARNING: no conda module could be loaded" >&2
fi

CONDA_ENV="${CONDA_ENV:-$HOME/.conda/envs/fhir_env}"
PYTHON="$CONDA_ENV/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  if command -v conda >/dev/null 2>&1; then
    source activate "$CONDA_ENV" 2>/dev/null && PYTHON="$(command -v python)"
  fi
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found at $CONDA_ENV/bin/python" >&2
  echo "Create the environment on the MASTER node first:" >&2
  echo "  module load Miniforge3" >&2
  echo "  conda create -y -n fhir_env python=3.11" >&2
  echo "  source activate fhir_env" >&2
  echo "  pip install vllm openai huggingface_hub pyyaml" >&2
  exit 1
fi

# --- Offline mode (compute nodes have no internet) --------------------------
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# --- Native library fixes for Rocky Linux 8.10 + conda ---------------------
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export LIBRARY_PATH="/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}"

# --- GPU selection ----------------------------------------------------------
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  BEST_GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
              2>/dev/null | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')"
  export CUDA_VISIBLE_DEVICES="${BEST_GPU:-0}"
  echo "CUDA_VISIBLE_DEVICES was unset; selected GPU: $CUDA_VISIBLE_DEVICES"
fi

echo "=============================================================="
echo " job         : ${SLURM_JOB_ID:-local}"
echo " node        : $(hostname)"
echo " partition   : ${SLURM_JOB_PARTITION:-n/a}"
echo " model       : $MODEL_ID"
echo " port        : $PORT"
echo " max_model_len: $MAX_MODEL_LEN"
echo " repo        : $REPO_ROOT"
echo " python      : $PYTHON"
echo " HF_HOME     : $HF_HOME"
echo " CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "=============================================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
echo

# --- Launch vLLM server -----------------------------------------------------
# The server runs until the Slurm time limit or scancel.
# It exposes an OpenAI-compatible API at http://<hostname>:<PORT>/v1/
exec srun "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --dtype auto \
  --trust-remote-code
