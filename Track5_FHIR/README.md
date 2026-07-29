# Track 5 – FHIR R4 Bundle extraction from clinical summaries

This track handles extracting HL7 FHIR R4 Bundles from clinical summaries.

## Model and Serving
- **Model:** MedGemma-27B-IT
- **Serving:** Served via vLLM on the Tanuh AI cluster

## How to run standalone
You can run the extraction script directly:
```bash
python Track5_FHIR/fhir_extractor.py --summary "The patient reports..." --endpoint http://n2:8000
```

## Architecture
Summary → MedGemma via vLLM → FHIR R4 Bundle

## Configuration
Configuration is managed via `config.yaml`.

## Integration
This component integrates with the larger microservice pipeline for the DISPLACE 2026 challenge.
