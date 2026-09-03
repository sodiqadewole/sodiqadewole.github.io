---
title: "Finetuning Llava-Next for Document Extraction"
excerpt: "A notebook project that fine-tunes LLaVA-NeXT with QLoRA on CORD-V2 receipt images for document extraction."
collection: portfolio
permalink: /portfolio/llava-next-document-extraction/
---

This project converts the LLaVA-NeXT document extraction notebook into a portfolio entry. It covers a multimodal finetuning workflow for turning receipt images into structured JSON-style outputs.

[Read the full blog walkthrough](/posts/2026/09/finetuning-llava-next-document-extraction/)

## Project Components

- Model: `llava-hf/llava-v1.6-mistral-7b-hf`
- Dataset: CORD-V2 receipt images and ground-truth parses
- Training approach: QLoRA with PEFT adapters
- Trainer: PyTorch Lightning
- Evaluation: normalized edit distance between generated and target extraction strings
- Post-processing: `token2json` conversion from generated tags to JSON

## Notes

The training run demonstrated the full pipeline and also hit a CUDA out-of-memory error on a 16GB GPU. That memory result is part of the project documentation because it shows the practical constraints of local multimodal finetuning.