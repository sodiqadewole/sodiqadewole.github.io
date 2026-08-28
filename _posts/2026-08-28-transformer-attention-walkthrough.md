---
title: 'Transformer Attention: A Step-by-Step Walkthrough'
date: 2026-08-28
permalink: /posts/2026/08/transformer-attention-walkthrough/
blog_category: llm-from-scratch
blog_section: Transformer Building Blocks
blog_summary: "Step through how a Transformer turns token embeddings into contextual representations using queries, keys, values, and softmax attention."
tags:
  - Transformers
  - Attention
  - LLM From Scratch
---

{% include toc %}

Self-attention gives each token a way to gather information from the rest of a sequence. For a token representation $x_i$, the layer creates a query, key, and value:

$$
q_i = x_iW_Q, \qquad k_i = x_iW_K, \qquad v_i = x_iW_V.
$$

The query for token $i$ scores every key. Softmax normalizes those scores, then the resulting weights combine the value vectors:

$$
\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

{% include step-through.html id="transformer-attention" data="attention-walkthrough" title="Follow one token through self-attention" %}

## What the Walkthrough Shows

The active token is **model**. Its query expresses the contextual features it needs, and each token's key is scored against that query. In a real Transformer these vectors are much larger, and every token runs this calculation in parallel.

The important consequence is that a token's final representation is contextual. The vector for `model` after attention is not simply the vector it started with: it contains information gathered from `The`, `model`, and `learns` according to the learned attention weights.

## From One Head to Multi-Head Attention

A single attention head can learn one relationship pattern. Multi-head attention repeats the same operation with separate projection matrices, allowing different heads to focus on different patterns such as syntax, position, or long-range dependencies:

$$
\operatorname{MultiHead}(Q,K,V) = \operatorname{Concat}(\operatorname{head}_1, \ldots, \operatorname{head}_h)W_O.
$$

This is the core operation that makes Transformer layers context-aware. The next useful step is to implement the calculation in PyTorch and inspect the tensor shapes at every stage.
