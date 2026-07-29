import argparse
import sys
import os
import json
import time

# Add the current directory to path so we can import backend modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.app.services.diarization import diarization_service
from backend.app.services.transcription import transcription_service
from backend.app.services.topic_extraction import topic_extraction_service
from backend.app.services.summarization import summarization_service
from backend.app.services.fhir_extraction import fhir_extraction_service

def main():
    parser = argparse.ArgumentParser(description="DISPLACE-2026 CLI Pipeline (Track 1 -> 5)")
    parser.add_argument("--audio", type=str, required=True, help="Path to the input .wav file")
    parser.add_argument("--rttm", type=str, help="Path to existing .rttm file (skips Track 1)")
    parser.add_argument("--lang", type=str, default="hi", help="Language code (e.g. hi, te, bn)")
    parser.add_argument("--output", type=str, default="output_fhir.json", help="Output JSON path")
    args = parser.parse_args()

    args.audio = args.audio.strip()
    if args.rttm:
        args.rttm = args.rttm.strip()

    print(f"\n=======================================================")
    print(f" DISPLACE MedAI - Full Cascading Pipeline")
    print(f"=======================================================\n")
    
    total_start = time.time()

    # Track 1: Diarization
    diarization_segments = []
    if args.rttm:
        if not os.path.exists(args.rttm):
            raise FileNotFoundError(f"RTTM file was provided but not found at path: {args.rttm}")
        print(f"[Track 1] Using existing RTTM file: {args.rttm}")
        rttm_path = args.rttm
        
        # Parse the RTTM file into segments
        with open(rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    start_time = float(parts[3])
                    duration = float(parts[4])
                    speaker_id = parts[7]
                    diarization_segments.append({
                        "start_time": start_time,
                        "end_time": start_time + duration,
                        "speaker_id": speaker_id
                    })
    else:
        print(f"[Track 1] Running Pyannote Diarization on {args.audio}...")
        diar_result = diarization_service.run(args.audio)
        rttm_path = diar_result["rttm_path"]
        diarization_segments = diar_result["segments"]
        print(f"          -> Completed in {diar_result['elapsed_s']:.1f}s")
        diarization_service.unload_model()
        
    # Track 2: Transcription
    print(f"\n[Track 2] Running Whisper Transcription ({args.lang})...")
    trans_result = transcription_service.run(args.audio, diarization_segments, args.lang)
    print(f"          -> Completed in {trans_result['elapsed_s']:.1f}s")
    transcription_service.unload_model()
    
    transcript = trans_result["full_transcript"]
    
    # Track 4: Summarization
    print(f"\n[Track 4] Running Llama 3 8B Summarization...")
    summary_result = summarization_service.run(transcript)
    print(f"          -> Completed in {summary_result['elapsed_s']:.1f}s")
    summarization_service.unload_model()
    
    summary = summary_result["summary"]
    print(f"\n[Summary Generated]:\n{summary}\n")

    # Track 5: FHIR Extraction (via MedGemma Server)
    print(f"\n[Track 5] Extracting FHIR R4 Bundle...")
    fhir_result = fhir_extraction_service.run(summary)
    print(f"          -> Completed in {fhir_result['elapsed_s']:.1f}s")
    
    if not fhir_result["valid"]:
        print("\nWARNING: FHIR Bundle had validation errors!")
        print(fhir_result["validation_errors"])
        
    # Save output
    with open(args.output, "w") as f:
        json.dump(fhir_result["fhir_json"], f, indent=2)
        
    total_time = time.time() - total_start
    print(f"\n=======================================================")
    print(f" PIPELINE COMPLETE in {total_time:.1f}s")
    print(f" Output saved to {args.output}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    main()
