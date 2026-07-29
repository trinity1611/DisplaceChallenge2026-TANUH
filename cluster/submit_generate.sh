#!/bin/bash
    # =============================================================================
    # Generate FHIR bundles on a Tanuh GPU compute node.
    #
    #   sbatch cluster/submit_generate.sh qwen25_7b_vllm
    #   sbatch --nodelist=n2 cluster/submit_generate.sh medgemma_27b_text_vllm
    #
    # Node tiers (Tanuh AI cluster guide, s1.1):
    #   n1  4x RTX Pro 6000, 96 GB VRAM   - fine for 7B, workable for 27B
    #   n2  8x H200,        141 GB VRAM   - preferred for the 27B model
    # Pick one with --nodelist on the sbatch command line; the script does not
    # hardcode a node so it stays usable on both tiers.
    #
    # Compute nodes have NO internet (private 192.168.1.0/24), so weights must
    # already be cached by cluster/download_model.py run on the MASTER node.
    # =============================================================================
    #SBATCH --job-name=fhir_gen
    #SBATCH --output=logs/gen-%j.out
    #SBATCH --error=logs/gen-%j.err
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=16
    #SBATCH --gres=gpu:1
    #SBATCH --partition=normal
    #SBATCH --time=04:00:00

    set -uo pipefail

    MODEL_KEY="${1:-}"
    if [[ -z "$MODEL_KEY" ]]; then
      echo "usage: sbatch cluster/submit_generate.sh <model_key>" >&2
      echo "see the models: section of config/pipeline.yaml" >&2
      exit 2
    fi

    # --- Locate the repository --------------------------------------------------
    # Slurm COPIES the batch script into a spool directory before running it, so
    # ${BASH_SOURCE[0]} points at /var/spool/slurmd/... and cannot be used to find
    # the repo. $SLURM_SUBMIT_DIR is the directory sbatch was invoked from, which is
    # the reliable answer. Fall back to the script path for plain shell runs.
    find_repo_root() {
      local candidate
      for candidate in \
          "${FHIR_BENCH_ROOT:-}" \
          "${SLURM_SUBMIT_DIR:-}" \
          "$PWD" \
          "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" \
          "$HOME/FHIR-Validator" ; do
        if [[ -n "$candidate" && -d "$candidate/src/fhirbench" ]]; then
          echo "$candidate"
          return 0
        fi
      done
      return 1
    }

    REPO_ROOT="$(find_repo_root || true)"
    if [[ -z "$REPO_ROOT" ]]; then
      echo "ERROR: could not locate the FHIR-Validator repository." >&2
      echo "Submit from the repo root, or set FHIR_BENCH_ROOT:" >&2
      echo "  cd ~/FHIR-Validator && sbatch cluster/submit_generate.sh <model_key>" >&2
      echo "  FHIR_BENCH_ROOT=/path/to/repo sbatch cluster/submit_generate.sh <model_key>" >&2
      exit 1
    fi
    cd "$REPO_ROOT"
    mkdir -p logs

    # --- Software stack (LMod). module purge first, per the guide s3. -----------
    module purge

    load_conda() {
      # Tanuh exposes Miniforge3/26.1.1-3 (confirmed via 'module avail'); the
      # guide's "module load miniconda" is generic boilerplate. Try the real name
      # first and confirm conda actually appeared on PATH.
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
      echo "WARNING: no conda module could be loaded; relying on the env path directly" >&2
      echo "         run 'module avail' on the master node to find the right name" >&2
    fi

    CONDA_ENV="${CONDA_ENV:-$HOME/.conda/envs/fhir_env}"
    PYTHON="$CONDA_ENV/bin/python"

    if [[ ! -x "$PYTHON" ]]; then
      # Fall back to the activation route the guide recommends (never conda init).
      if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source activate "$CONDA_ENV" 2>/dev/null && PYTHON="$(command -v python)"
      fi
    fi
    if [[ ! -x "$PYTHON" ]]; then
      echo "Python not found at $CONDA_ENV/bin/python" >&2
      echo "Create the environment on the MASTER node first:" >&2
      echo "  module load miniconda" >&2
      echo "  conda create -y -n fhir_env python=3.11" >&2
      echo "  source activate fhir_env" >&2
      echo "  pip install -r requirements-cluster.txt" >&2
      exit 1
    fi

    export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
    export TOKENIZERS_PARALLELISM=false

    # Fail fast and legibly if the repo/env pairing is wrong, rather than dying on an
    # import traceback after the allocation has already been granted.
    if ! "$PYTHON" -c "import fhirbench" >/dev/null 2>&1; then
      echo "ERROR: cannot import fhirbench with PYTHONPATH=$PYTHONPATH" >&2
      echo "       repo root resolved to: $REPO_ROOT" >&2
      echo "       expected package at  : $REPO_ROOT/src/fhirbench/__init__.py" >&2
      exit 1
    fi
    if [[ ! -f "$REPO_ROOT/config/pipeline.yaml" ]]; then
      echo "ERROR: $REPO_ROOT/config/pipeline.yaml not found" >&2
      exit 1
    fi

    # Weights live on NFS /home so every compute node sees the same cache.
    export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
    # Compute nodes cannot reach the internet; fail fast instead of hanging on a
    # network timeout if something is not cached.
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

    # --- Native library resolution (Rocky Linux 8.10 + conda) ----------------
    # Rocky 8.10 ships GCC 8, whose /lib64/libstdc++.so.6 provides only up to
    # GLIBCXX_3.4.25. FlashInfer JIT-compiles a sampling kernel against the conda
    # toolchain and then needs GLIBCXX_3.4.32 (GCC 13+) at load time, failing with
    #   "version `GLIBCXX_3.4.32' not found (required by .../sampling.so)".
    # Putting the conda env's newer libstdc++ first on the search path fixes it.
    export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
    # Belt and braces: decoding is greedy (temperature 0.0), so the FlashInfer
    # top-k/top-p sampler is never actually needed. Disabling it avoids building
    # and loading that kernel at all - the exact point that crashed on n2.
    export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
    # NOTE: VLLM_USE_V1 is intentionally NOT set. vLLM 0.11+ has only the V1
    # engine and logs "Unknown vLLM environment variable" for it.

    echo "=============================================================="
    echo " job         : ${SLURM_JOB_ID:-local}"
    echo " node        : $(hostname)"
    echo " partition   : ${SLURM_JOB_PARTITION:-n/a}"
    echo " model key   : $MODEL_KEY"
    echo " repo        : $REPO_ROOT"
    echo " python      : $PYTHON"
    echo " HF_HOME     : $HF_HOME"
    echo " CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "=============================================================="
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
    echo

    # nvidia-smi always lists every physical GPU, so the table above says nothing
    # about what Slurm allocated. If the scheduler did not isolate devices, pick the
    # least-used card explicitly - the nodes are shared, and defaulting to GPU 0
    # risks an OOM on a card another job already occupies.
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      BEST_GPU="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                  2>/dev/null | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')"
      export CUDA_VISIBLE_DEVICES="${BEST_GPU:-0}"
      echo "CUDA_VISIBLE_DEVICES was unset; selected the least-used GPU: $CUDA_VISIBLE_DEVICES"
      echo
    fi

    export VLLM_USE_FLASHINFER_SAMPLER=0
    export LIBRARY_PATH="/usr/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"

    srun "$PYTHON" -m fhirbench.generate --model "$MODEL_KEY"
    STATUS=$?

    echo
    if [[ $STATUS -eq 0 ]]; then
      echo "generation succeeded for all transcripts"
    else
      echo "generation FAILED (exit $STATUS) - at least one transcript did not"
      echo "produce valid JSON. Inspect the raw completions:"
      echo "  $REPO_ROOT/outputs/$MODEL_KEY/_raw/"
    fi
    echo "outputs  : $REPO_ROOT/outputs/$MODEL_KEY"
    echo "manifest : $REPO_ROOT/outputs/$MODEL_KEY/manifest.json"
    exit $STATUS