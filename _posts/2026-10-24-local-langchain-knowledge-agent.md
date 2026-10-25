---
title: "Local Agentic RAG Assistant"
date: 2026-10-24
permalink: /posts/2026/10/local-langchain-knowledge-agent/
blog_category: ai-agents
blog_section: Local Knowledge Agents
blog_summary: "A local LangChain chat agent combines Qwen2.5-VL, vLLM, and SQLite FTS5 BM25 to answer from private Markdown, text, and PDF files—with measured indexing and retrieval latency."
read_time: true
tags:
  - Local RAG Assistant
  - Local Assistant
  - RAG
  - LangChain
  - vLLM
  - BM25
  - Local LLM
  - Qwen2.5-VL
  - Retrieval
  - SQLite FTS5
---

Retrieval-augmented generation is often introduced as a vector database plus an embedding model. But a useful knowledge assistant does not always need either. In [`conversation_agent`](https://github.com/sodiqadewole/conversation_agent), I built a local conversational agent that searches Markdown, plain-text, and text-based PDF files with SQLite full-text search, ranks passages using BM25, and supplies the best matches to a local language model.

The project brings together three pieces: a Streamlit chat interface, a LangChain agent with tools, and a local OpenAI-compatible vLLM server. The result is a small, inspectable knowledge-retrieval pipeline with measured latencies at each important boundary.

## Project architecture

```text
User
  |
  v
Streamlit chat UI
  |
  +--> Synchronize local knowledge index
  +--> Retrieve up to three relevant passages
  |
  v
LangChain agent and tools
  |
  v
OpenAI-compatible local API
  |
  v
Qwen2.5-VL-3B-Instruct served by vLLM
```

The interface is built with **Streamlit**. **LangChain** creates the conversational agent with `create_agent`, connects it to the OpenAI-compatible model client, and exposes two tools: a current date/time lookup and an explicit local-knowledge search. The model server uses **vLLM** for local inference.

The configured checkpoint is **Qwen2.5-VL-3B-Instruct**. Its vision-language capability is enabled in the serving setup, but the current chat interface sends text prompts. The local knowledge index also extracts text; it is not an image-understanding or OCR pipeline.

## Supported documents and boundaries

The knowledge directory accepts:

| Type | How it is processed |
|---|---|
| Markdown (`.md`) | Split at headings and paragraph boundaries; the nearest heading is retained for attribution. |
| Plain text (`.txt`) | Split by paragraphs into word-counted passages. |
| Text-based PDF (`.pdf`) | Extract text page by page with `pypdf`; each passage retains its page number. |

The application supports up to 100 files. Markdown and text files are limited to 1 MB each, and PDFs to 20 MB. A PDF must have an extractable text layer: scanned, image-only PDFs need OCR before indexing. File names and page or heading labels help identify a source, but the file path itself is not searched or ranked.

## How indexing works

Each supported file is parsed into passages targeting **1,000 words**. Paragraphs are accumulated until the target would be exceeded; an oversized paragraph is split at word boundaries. Chunks do not cross Markdown heading or PDF page boundaries, so returned evidence has useful local context and an attribution label.

Passages are stored in a local **SQLite FTS5** virtual table. FTS5 tokenizes the indexed heading and passage text with its Unicode-aware `unicode61` tokenizer and maintains an inverted index: a mapping from terms to the passages in which they occur. That structure lets SQLite find matching candidates without comparing the query against every passage one by one.

The ordinary metadata table tracks each file's relative path, modification timestamp, and byte size. During synchronization, the index checks for added, changed, or removed sources and only reparses files whose signature has changed. A configuration fingerprint also records the parser and chunking settings; if those settings change, the generated index is rebuilt. The Streamlit sidebar polls the directory every 20 seconds, while each search synchronizes immediately so a new or edited file can be picked up without waiting for the next poll.

## How retrieval and BM25 ranking work

For a question such as “How do I configure motion detection on the camera?”, the retriever:

1. Case-folds the question and extracts Unicode word terms.
2. Drops one-character terms and a small configured set of stopwords.
3. Removes duplicate terms, quotes them, and combines them with `OR`.
4. Uses FTS5's inverted index to find chunks containing at least one query term in a heading or passage.
5. Orders the matching chunks by SQLite FTS5's built-in BM25 score and returns the requested top results.
6. Extracts a short snippet from each result and attaches up to three passages to the latest user message, with source labels for citations.

BM25 is a probabilistic ranking function designed to estimate how well a document matches a query. Here, a “document” is one passage-sized chunk. Its score combines three intuitions:

- **Rare query terms are more informative.** Inverse document frequency gives a term more weight when it appears in fewer chunks than a common term.
- **More occurrences help, but with diminishing returns.** Term-frequency saturation prevents repeating a word many times from multiplying its influence without bound.
- **Passage length matters.** Length normalization avoids systematically favoring long chunks simply because they have more opportunities to contain query terms.

SQLite FTS5 returns BM25 values with the sign inverted from the traditional presentation, so a better match has a smaller (typically more negative) score. The query therefore sorts with `ORDER BY bm25(chunks)` in ascending order.

This is **lexical**, not semantic, search: there is no embedding model, vector database, stemming, or synonym expansion. A query can miss a passage when it uses different wording from the source. Joining terms with `OR` favors recall—a passage matching one query term can still be considered—while BM25 places passages with more useful term matches higher. That trade-off keeps the implementation local and lightweight, but it means retrieval quality still depends on the language used in the question and documents.

## Example conversations

These screenshots show the Streamlit interface retrieving local document passages and displaying source citations alongside the answers.

<figure>
  <img src="/images/projects/local-knowledge-assistant/chat-motion-detection.png" alt="Local Knowledge Assistant answers a question about configuring camera motion detection and cites pages 49 and 50 of the network camera manual." loading="lazy">
  <figcaption>Asking how to configure motion detection retrieves relevant passages from the camera manual, with page citations shown below the answer.</figcaption>
</figure>

<figure>
  <img src="/images/projects/local-knowledge-assistant/chat-supported-files.png" alt="Local Knowledge Assistant describes supported Markdown, text, and PDF files, including the need for OCR on scanned PDFs, with local source citations." loading="lazy">
  <figcaption>Asking which file types the local index can search demonstrates a second grounded response with source labels.</figcaption>
</figure>

### Patent correction guidance

<figure>
  <img src="/images/projects/local-knowledge-assistant/q5.png" alt="Chat prompt asking how patents are corrected." loading="lazy">
  <figcaption>The question asks how patent corrections work.</figcaption>
</figure>

<figure>
  <img src="/images/projects/local-knowledge-assistant/chat_5.png" alt="Local Knowledge Assistant summarizes patent correction options and cites passages from local patent examination references." loading="lazy">
  <figcaption>The assistant summarizes correction options and cites relevant sections of the local patent references.</figcaption>
</figure>

### Patent examination process

<figure>
  <img src="/images/projects/local-knowledge-assistant/q6.png" alt="Chat prompt asking about the patent examination process." loading="lazy">
  <figcaption>The question asks for an overview of the patent examination process.</figcaption>
</figure>

<figure>
  <img src="/images/projects/local-knowledge-assistant/chat_6.png" alt="Local Knowledge Assistant summarizes the patent examination process with citations to local PDF references and reports response latency and token counts." loading="lazy">
  <figcaption>The answer is grounded in several PDF passages and shows the recorded response latency and token counts.</figcaption>
</figure>

## Measured indexing and retrieval latency

The project measured the local retrieval pipeline on **October 24, 2026**. The test corpus contained two sources: the knowledge README and a 126-page network-camera manual. Together they were 1,859,629 bytes and produced 127 indexed chunks. Measurements ran on Linux x86-64 with Python 3.12.3, SQLite 3.45.1, and `pypdf` 6.21.2.

| Operation | Runs | Median (p50) | p95 | Maximum |
|---|---:|---:|---:|---:|
| Fresh full index build | 5 | 1,820.422 ms | 1,961.605 ms | 1,961.605 ms |
| Synchronize unchanged sources | 200 | 5.206 ms | 7.061 ms | 9.588 ms |
| End-to-end local retrieval | 300 | 9.326 ms | 12.594 ms | 16.948 ms |
| SQLite FTS/BM25 query only | 1,000 | 0.195 ms | 0.821 ms | 1.157 ms |

The **fresh index build** includes parsing both sources, extracting all PDF pages, splitting the text into chunks, and building the full-text index. PDF extraction is the main cost. On a typical unchanged request, synchronization takes roughly 5 ms at the median, and the complete local retrieval path takes roughly 9 ms. The database's FTS/BM25 matching and ranking itself is below a millisecond at this corpus size; opening the database, checking the files, and preparing snippets make up most of the remaining search time.

These are empirical results for the stated machine and small corpus, not universal performance guarantees. The benchmark did not clear operating-system file caches between fresh builds. More importantly, retrieval measurements **exclude model inference**, vLLM/network overhead, Streamlit rendering, and model tool-selection time. They describe indexing and local retrieval, not total time until a generated answer appears.

## Where this fits with my other projects

This project is part of a broader set of experiments in local agents, retrieval, and document AI:

- [Retrieval-Augmented Generation with Phi-3 and LlamaCpp](/posts/2026/03/retrieval-augmented-generation/) explores embedding-based retrieval with `thenlper/gte-small` and FAISS. It offers a useful contrast with this project's BM25 approach: semantic vectors can help bridge wording differences, while lexical search avoids the embedding and vector-store components.
- [Advanced Search Optimization using Dense Retrieval](/posts/2026/03/advanced-search-using-search-retrieval/) walks through sentence embeddings and cosine-similarity ranking.
- [Local Qwen Research Agent with SmolAgents](/portfolio/local-qwen-research-agent-smolagents/) and [Local Qwen Code Agent with SmolAgents](/portfolio/local-qwen-code-agent-smolagents/) apply a different agent framework to web research and code execution.
- [Multimodal Document Intelligence API](/portfolio/multimodal-document-intelligence-api/) focuses on a FastAPI service for local Qwen-based chat and PDF or image extraction.
- [Fine-tuning LLaVA-NeXT for Document Extraction](/portfolio/llava-next-document-extraction/) explores adapting a vision-language model to receipt parsing.

Taken together, these projects compare different ways to build grounded assistants: local keyword retrieval, dense semantic search, tool-using agents, and multimodal document processing. The conversation agent is a particularly compact example of how far a local full-text index can go before adding embeddings or a separate search service.

## Try the project

The [source repository](https://github.com/sodiqadewole/conversation_agent) includes setup instructions, tests, the detailed [retrieval-algorithm and latency report](https://github.com/sodiqadewole/conversation_agent/blob/main/docs/knowledge-retrieval-algorithm.md), and a sample network-camera manual. Add trusted `.md`, `.txt`, or text-based `.pdf` files to the knowledge directory, start the local vLLM server, and launch the Streamlit app to explore retrieval with your own questions.

## References

- [SQLite FTS5 documentation](https://www.sqlite.org/fts5.html)
- [SQLite FTS5 BM25 function](https://www.sqlite.org/fts5.html#the_bm25_function)
- [SQLite FTS5 snippet function](https://www.sqlite.org/fts5.html#the_snippet_function)
