# Multimodal Document Intelligence API

This project page presents the source repository as a local-first applied AI service. The implementation lives in the public GitHub repository:

https://github.com/sodiqadewole/multimodal_documentIntelligence_api

## What the service demonstrates

- FastAPI routes with explicit request and response models.
- Local Qwen2.5 text and vision inference through Transformers.
- PDF and image document extraction for invoice-like inputs.
- Health, readiness, OpenAPI, request IDs, stable errors, and resource limits.
- Provider dependency injection so API tests do not load model weights.

## Primary endpoints

- `GET /health`
- `GET /ready`
- `POST /v1/chat/completions`
- `POST /v1/document/completions`
- `GET /docs`
- `GET /openapi.json`

## Run the source project

```bash
git clone https://github.com/sodiqadewole/multimodal_documentIntelligence_api.git
cd multimodal_documentIntelligence_api
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m llm_service.download_model
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
  uvicorn llm_service.app:app --host 127.0.0.1 --port 8000
```

The portfolio page is the presentation layer for the project. The GitHub repository remains the source of truth for implementation, model setup, examples, tests, and deployment decisions.
