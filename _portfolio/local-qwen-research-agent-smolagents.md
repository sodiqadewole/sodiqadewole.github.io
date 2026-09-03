---
title: "Local Qwen Research Agent with SmolAgents"
excerpt: "A notebook project that builds a local web-search research agent with Qwen2.5-0.5B-Instruct, SmolAgents, and DuckDuckGo search."
collection: portfolio
permalink: /portfolio/local-qwen-research-agent-smolagents/
---

This project converts the research-agent notebook into a portfolio project. It demonstrates how to use a local Qwen model with SmolAgents `ToolCallingAgent` and a web-search tool to perform fact lookup, comparison, trend analysis, and investigative research tasks.

[Read the full blog walkthrough](/posts/2026/08/local-qwen-research-agent-smolagents/)

## Project Components

- Local model: `Qwen/Qwen2.5-0.5B-Instruct`
- Agent framework: SmolAgents `ToolCallingAgent`
- Tooling: `DuckDuckGoSearchTool`
- Research tasks: capital lookup, Python facts, ML vs. DL comparison, AI trends, remote work analysis, and AI-for-climate research

## Notes

The notebook surfaced practical research-agent failure modes, including repeated searches, malformed tool calls, stale facts, and long-running tasks that reached max steps.