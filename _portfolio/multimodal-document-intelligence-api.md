---
title: "Multimodal Document Intelligence API"
excerpt: "A local-first FastAPI service that turns chat prompts and PDF or image documents into structured model responses with explicit health, readiness, tracing, and resource controls."
collection: portfolio
permalink: /portfolio/multimodal-document-intelligence-api/
---

<div class="project-hero">
  <p class="project-hero__eyebrow">Applied AI Infrastructure / PyTorch and FastAPI</p>
  <h2>Private document understanding, exposed as a clean API.</h2>
  <p class="project-hero__lede">A local-first service for text generation and multimodal document extraction, backed by Qwen2.5 models and designed around predictable HTTP contracts, offline inference, and operational boundaries that are visible in the code.</p>
  <p class="project-hero__actions"><a class="btn" href="https://github.com/sodiqadewole/multimodal_documentIntelligence_api">View source on GitHub</a> <a class="btn btn--inverse" href="https://github.com/sodiqadewole/multimodal_documentIntelligence_api/blob/main/API_DESIGN.md">Read the API design</a></p>
</div>

## The project in one view

| Surface | What it provides |
| :--- | :--- |
| Text generation | OpenAI-shaped `POST /v1/chat/completions` requests with generation controls. |
| Document extraction | PDF, PNG, JPEG, and WebP uploads through `POST /v1/document/completions`. |
| Model boundary | Local Qwen2.5 text and vision models loaded from an explicit model store. |
| Service visibility | `/health`, `/ready`, `/docs`, and `/openapi.json` for operators and clients. |
| Request tracing | A stable `X-Request-ID` on every response, with caller-supplied IDs supported. |
| Failure behavior | Consistent error envelopes, `502` provider failures, and `503` unavailable-model responses. |

## Why this is useful

Document intelligence demos often stop at a notebook cell that calls a model. This project takes the next step: it puts the model behind a small service boundary that another application can call, monitor, test, and eventually place behind an authenticated gateway.

The local-first design is deliberate. API requests use `local_files_only=True`, so serving a request never silently downloads weights. A separate consent-gated downloader handles model acquisition, while the API focuses on inference. That separation makes the system easier to reason about in restricted environments and easier to test without loading model weights.

## Request flow

```text
Client
  |
  |  JSON chat request or PDF/image upload
  v
FastAPI routes and Pydantic contracts
  |
  +--> request ID, validation, size limits, stable errors
  v
Application orchestration
  |
  +--> text provider (Qwen2.5-0.5B-Instruct)
  +--> vision provider (Qwen2.5-VL-3B-Instruct)
  v
Local model store
  |
  +--> generated text or structured document extraction
```

The HTTP contract lives in the repository's models layer, orchestration is handled by the application layer, and local Transformers inference is isolated behind providers. Tests replace the provider through FastAPI dependency injection, which keeps contract tests fast and independent of model downloads.

## A practical invoice path

The repository includes a fictional image-only invoice in PNG and PDF formats. A client can send the document and an extraction prompt directly to the vision endpoint:

```bash
curl http://127.0.0.1:8000/v1/document/completions \\
  -H 'X-Request-ID: invoice-demo-001' \\
  -F 'file=@examples/sample_invoice.pdf;type=application/pdf' \\
  -F 'prompt=Extract valid JSON. Include invoice number, dates, line items, subtotal, tax, amount paid, and amount due.' \\
  -F 'max_tokens=512'
```

The same workflow is available through the repository's Python example and multimodal notebook client, making the project easy to inspect from both an HTTP and experimentation perspective.

## Engineering decisions

### Local inference is explicit

The service loads models on the first completion request, uses CUDA when available, and falls back to CPU when necessary. It serializes generation through a worker thread because one in-process model instance is shared by the service.

### Limits are part of the contract

The reference service accepts uploads up to 20 MB, PDFs up to 10 pages, and images up to 25 megapixels. These are safety defaults for a synchronous service, not claims about the underlying model limits. The repository documents a larger-document job architecture as a production extension.

### The service is honest about production gaps

Before public exposure, the project identifies the next required layers: authentication, per-tenant rate and token limits, structured metrics, timeouts, streaming responses, and a dedicated inference engine such as vLLM for higher concurrency.

## Local setup

The source repository requires Python 3.11 or newer. A minimal local setup is:

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

Open `http://127.0.0.1:8000/docs` for the generated OpenAPI interface or run `python -m pytest -q` for the contract and unit tests.

## Repository map

```text
src/llm_service/       FastAPI app, processors, providers, and model store
tests/                 Unit and API contract tests
examples/              Sample invoice assets and executable clients
models/                Local model store; weights are ignored by Git
API_DESIGN.md          Architecture, capacity, security, and deployment decisions
*.ipynb                Text and multimodal notebook clients
```

## Project links

- [Source repository](https://github.com/sodiqadewole/multimodal_documentIntelligence_api)
- [API design decisions](https://github.com/sodiqadewole/multimodal_documentIntelligence_api/blob/main/API_DESIGN.md)
- [Open issues and future production work](https://github.com/sodiqadewole/multimodal_documentIntelligence_api/issues)
