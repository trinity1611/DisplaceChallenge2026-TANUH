echo "Running ASR..."
python -u ../Track2_ASR/scripts/asr.py

echo "Running Summarization..."
python -u scripts/summarization.py