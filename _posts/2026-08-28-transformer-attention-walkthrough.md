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

## Attention Mechanism

Attention lets a model choose which information matters for its current decision. Rather than compressing an entire input into one fixed representation, it compares a **query** with a collection of **keys** and uses the resulting relevance scores to blend their associated **values**.

For a query $q$ and key-value pairs $(k_j, v_j)$, attention produces a context vector $c$:

$$
\alpha_j = \operatorname{softmax}(qk_j^\top), \qquad c = \sum_j \alpha_jv_j.
$$

Each weight $\alpha_j$ says how much information to draw from value $v_j$. A query that closely matches a key receives a larger weight, so the output can focus on the most relevant parts of the input. In encoder-decoder models, the decoder's current state commonly provides the query while encoder states supply keys and values.

<figure class="attention-example" aria-labelledby="attention-example-title">
  <figcaption id="attention-example-title">Example: resolving a pronoun with attention</figcaption>
  <p class="attention-example__sentence">The <mark>animal</mark> did not cross the street because <mark class="attention-example__query">it</mark> was too <mark>tired</mark>.</p>
  <div class="attention-example__query-label">Query: the token <strong>it</strong> asks which earlier token it refers to.</div>
  <div class="attention-example__weights" aria-label="Attention weights from it to earlier context: animal 62 percent, street 5 percent, because 9 percent, tired 24 percent.">
    <div class="attention-example__weight"><span>animal</span><b style="--weight: 62%">62%</b></div>
    <div class="attention-example__weight"><span>street</span><b style="--weight: 5%">5%</b></div>
    <div class="attention-example__weight"><span>because</span><b style="--weight: 9%">9%</b></div>
    <div class="attention-example__weight"><span>tired</span><b style="--weight: 24%">24%</b></div>
  </div>
  <p class="attention-example__takeaway">The largest weight falls on <strong>animal</strong>, so the contextual representation of <strong>it</strong> carries information that it refers to the animal.</p>
</figure>

## Self-Attention

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
