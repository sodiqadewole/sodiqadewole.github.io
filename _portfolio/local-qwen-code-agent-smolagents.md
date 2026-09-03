---
title: "Local Qwen Code Agent with SmolAgents"
excerpt: "A notebook project that builds a local code-execution agent with Qwen2.5-0.5B-Instruct and SmolAgents."
collection: portfolio
permalink: /portfolio/local-qwen-code-agent-smolagents/
---

This project converts the local Qwen code-agent notebook into a publishable project page. It demonstrates how to wrap a local Transformers model with SmolAgents `CodeAgent`, run simple code-generation tasks, and inspect where a small local model succeeds or fails.

[Read the full blog walkthrough](/posts/2026/08/local-qwen-code-agent-smolagents/)

## Project Components

- Local model: `Qwen/Qwen2.5-0.5B-Instruct`
- Agent framework: SmolAgents `CodeAgent`
- Execution style: generated Python code run inside the agent loop
- Evaluation tasks: arithmetic, student ranking, text analysis, prime generation, and sales-data reporting

## Notes

The notebook showed that simple arithmetic worked well, while structured-data and longer reasoning tasks needed stronger prompting, explicit input definitions, and output validation.