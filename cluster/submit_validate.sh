#!/bin/bash
# =============================================================================
# Validate generated bundles against FHIR R4, then score the run. CPU only.
#
#   sbatch cluster/submit_validate.sh test_fixtures     # smoke test first
#   sbatch cluster/submit_validate.sh outputs           # the real thing
#
# Uses the official HL7 validator_cli.jar through a local JVM. Tanuh has no
# apptainer or singularity module, and charliecloud 0.15 cannot pull a registry
# image without Docker, so the containerised Inferno service is not an option
# here.
#
# ONE-TIME SETUP ON THE MASTER NODE (compute nodes have no internet):
#
#   module purge && module load Miniforge3
#   source activate fhir_env
#   conda install -y -c conda-forge openjdk=17
#   curl -L -o $HOME/validator_cli.jar \
#     https://github.com/hapifhir/org.hl7.fhir.core/releases/latest/download/validator_cli.jar
#
#   # Prime the FHIR definitions cache in ~/.fhir - the jar downloads
#   # hl7.fhir.r4.core on first use, which a compute node cannot do.
#   cd ~/FHIR-Validator
#   java -jar $HOME/validator_cli.jar test_fixtures/valid/patient-valid.json \
#        -version 4.0.1 -tx n/a -output /tmp/probe.json
#   ls ~/.fhir/packages      # expect hl7.fhir.r4.core#4.0.1
# =============================================================================
#SBATCH --job-name=fhir_val
#SBATCH --output=logs/val-%j.out
#SBATCH --error=logs/val-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --partition=normal
#SBATCH --time=02:00:00

set -uo pipefail

TARGET="${1:-outputs}"

# --- Locate the repository --------------------------------------------------
# Slurm copies the batch script to a spool dir, so ${BASH_SOURCE[0]} is useless
# for finding the repo. $SLURM_SUBMIT_DIR is the sbatch invocation directory.
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
  echo "  cd ~/FHIR-Validator && sbatch cluster/submit_validate.sh [target]" >&2
  exit 1
fi
cd "$REPO_ROOT"
mkdir -p logs

# --- Software stack (LMod), per the cluster guide s3 ------------------------
module purge

load_conda() {
  # Tanuh exposes Miniforge3; the guide's "miniconda" is generic boilerplate.
  local candidate
  for candidate in Miniforge3 miniforge3 miniconda miniforge anaconda conda; do
    if module load "$candidate" >/dev/null 2>&1 && command -v conda >/dev/null 2>&1; then
      echo "$candidate"; return 0
    fi
  done
  return 1
}
CONDA_MODULE="$(load_conda || true)"
[[ -n "$CONDA_MODULE" ]] && echo "loaded conda module: $CONDA_MODULE"

CONDA_ENV="${CONDA_ENV:-$HOME/.conda/envs/fhir_env}"
PYTHON="$CONDA_ENV/bin/python"
if [[ ! -x "$PYTHON" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source activate "$CONDA_ENV" 2>/dev/null && PYTHON="$(command -v python)"
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found at $CONDA_ENV/bin/python" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

if ! "$PYTHON" -c "import fhirbench" >/dev/null 2>&1; then
  echo "ERROR: cannot import fhirbench with PYTHONPATH=$PYTHONPATH" >&2
  echo "       repo root resolved to: $REPO_ROOT" >&2
  exit 1
fi

# Java comes from the conda env (there is no java module on this cluster).
export PATH="$CONDA_ENV/bin:$PATH"
export JAVA_BIN="${JAVA_BIN:-$CONDA_ENV/bin/java}"
export FHIR_VALIDATOR_JAR="${FHIR_VALIDATOR_JAR:-$HOME/validator_cli.jar}"
# The definitions cache lives on NFS /home, so it is visible from every node.
export FHIR_PACKAGE_CACHE="${FHIR_PACKAGE_CACHE:-$HOME/.fhir}"

if [[ ! -f "$FHIR_VALIDATOR_JAR" ]]; then
  echo "ERROR: $FHIR_VALIDATOR_JAR not found." >&2
  echo "Download it on the MASTER node - see the header of this script." >&2
  exit 1
fi
if [[ ! -x "$JAVA_BIN" ]] && ! command -v java >/dev/null 2>&1; then
  echo "ERROR: no java. On the MASTER node:" >&2
  echo "  source activate fhir_env && conda install -y -c conda-forge openjdk=17" >&2
  exit 1
fi
if [[ ! -d "$FHIR_PACKAGE_CACHE/packages" ]]; then
  echo "WARNING: $FHIR_PACKAGE_CACHE/packages is missing. The validator will try" >&2
  echo "         to download hl7.fhir.r4.core, which fails on an offline node." >&2
  echo "         Prime it on the master node first - see this script's header." >&2
fi

echo "=============================================================="
echo " job       : ${SLURM_JOB_ID:-local}"
echo " node      : $(hostname)"
echo " partition : ${SLURM_JOB_PARTITION:-n/a}"
echo " target    : $TARGET"
echo " jar       : $FHIR_VALIDATOR_JAR"
echo " java      : $JAVA_BIN"
echo " pkg cache : $FHIR_PACKAGE_CACHE"
echo "=============================================================="

"$PYTHON" -m fhirbench.validate --root "$TARGET" --engine jar
VALIDATE_STATUS=$?

# Score even when validation reported failures - a leaderboard of failures is
# still the thing you need to read.
"$PYTHON" -m fhirbench.score || true

echo
echo "validation exit status $VALIDATE_STATUS (non-zero == at least one file failed)"
echo "results : $REPO_ROOT/runs/"
exit 0
