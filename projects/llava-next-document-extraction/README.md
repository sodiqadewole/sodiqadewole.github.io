# Finetuning Llava-Next for Document Extraction

This project documents the notebook that fine-tunes LLaVA-NeXT with QLoRA on CORD-V2 receipt images.

## Components

- LLaVA-NeXT model and processor.
- CORD-V2 dataset loading and ground-truth inspection.
- JSON-to-token conversion for extraction labels.
- Train/evaluation collators for multimodal batches.
- PyTorch Lightning training module.
- Inference and token-to-JSON parsing.

## Related Page

- Blog walkthrough: `/posts/2026/09/finetuning-llava-next-document-extraction/`
- Portfolio page: `/portfolio/llava-next-document-extraction/`
