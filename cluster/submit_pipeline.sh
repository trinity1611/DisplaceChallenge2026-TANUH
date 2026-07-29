#!/bin/bash
#SBATCH --job-name=medai_pipeline
#SBATCH --output=logs/pipeline-%j.out
#SBATCH --error=logs/pipeline-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=normal

# Load environment
module purge
module load Miniforge3 2>/dev/null || true
source activate fhir_env

cd ~/DISPLACE-2026-Baselines/microservice

# Usage: sbatch cluster/submit_pipeline.sh <audio_file.wav> [rttm_file.rttm]

AUDIO_FILE=$1
RTTM_FILE=$2

if [ -z "$AUDIO_FILE" ]; then
    echo "Error: Must provide an audio file as the first argument."
    echo "Usage: sbatch cluster/submit_pipeline.sh ../2006763.wav ../2006763_SPEAKER.rttm"
    exit 1
fi

if [ -n "$RTTM_FILE" ]; then
    echo "Running pipeline with provided RTTM..."
    $HOME/.conda/envs/fhir_env/bin/python cli.py --audio "$AUDIO_FILE" --rttm "$RTTM_FILE" --output "output_$(basename "$AUDIO_FILE" .wav).json"
else
    echo "Running full pipeline from scratch..."
    $HOME/.conda/envs/fhir_env/bin/python cli.py --audio "$AUDIO_FILE" --output "output_$(basename "$AUDIO_FILE" .wav).json"
fi
