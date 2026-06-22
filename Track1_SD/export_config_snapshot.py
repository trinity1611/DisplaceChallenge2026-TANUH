import json

config_snapshot = {
    "track": "Track_1_SD (Speaker Diarization Baseline)",
    "dataset": "DISPLACE-2026 DevData (Hindi)",
    "framework": "DiariZen / Pyannote-Audio Integration",
    "achieved_der": "10.15%",
    "pipeline_parameters": {
        "segmentation_model": "pyannote/segmentation-3.0",
        "clustering": "Segmentation Clustering / Agglomerative Hierarchy",
        "voice_activity_detection": {
            "onset": 0.5,
            "offset": 0.5,
            "min_duration_on": 0.1,
            "min_duration_off": 0.1
        }
    }
}

with open("pipeline_config.json", "w") as f:
    json.dump(config_snapshot, f, indent=4)
print("Saved pipeline parameters successfully to pipeline_config.json!")
