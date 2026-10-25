---
title: "Local Knowledge Conversation Agent"
excerpt: "A local LangChain and Streamlit assistant powered by Qwen2.5-VL through vLLM, with private document search using SQLite FTS5 and BM25."
collection: portfolio
permalink: /portfolio/local-knowledge-conversation-agent/
---

A local conversational assistant that answers questions using a private collection of Markdown, plain-text, and text-based PDF files. The project combines a Streamlit chat interface, a LangChain agent and tools, an OpenAI-compatible vLLM server, and a local SQLite FTS5 index.

[Read the blog walkthrough](/posts/2026/10/local-langchain-knowledge-agent/)

## Project Components

- Chat interface: Streamlit
- Agent orchestration: LangChain `create_agent`
- Local model: Qwen2.5-VL-3B-Instruct served by vLLM
- Knowledge retrieval: SQLite FTS5 full-text search with BM25 ranking
- Supported documents: Markdown, plain text, and text-based PDFs
- Tools: local knowledge lookup and timezone-aware current date/time

The index splits files into heading- or page-aware passages, updates when source files change, and retrieves up to three ranked passages for a chat question. On the documented test corpus of two files and 127 passages, median end-to-end local retrieval was 9.326 ms; fresh indexing took a median of 1,820.422 ms. These figures cover indexing and retrieval only, not model inference or the complete chat response.

## Project Links

- [Source repository](https://github.com/sodiqadewole/conversation_agent)
- [Blog walkthrough](/posts/2026/10/local-langchain-knowledge-agent/)
- [Retrieval algorithm and latency report](https://github.com/sodiqadewole/conversation_agent/blob/main/docs/knowledge-retrieval-algorithm.md)
