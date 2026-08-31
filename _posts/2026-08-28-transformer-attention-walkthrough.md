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

## Attention Mechanism

Attention lets a model choose which information matters for its current decision. Rather than compressing an entire input into one fixed representation, it compares a **query** with a collection of **keys** and uses the resulting relevance scores to blend their associated **values**.

For a query $q$ and key-value pairs $(k_j, v_j)$, attention produces a context vector $c$:

$$
\alpha_j = \operatorname{softmax}(qk_j^\top), \qquad c = \sum_j \alpha_jv_j.
$$

Each weight $\alpha_j$ says how much information to draw from value $v_j$. A query that closely matches a key receives a larger weight, so the output can focus on the most relevant parts of the input. In encoder-decoder models, the decoder's current state commonly provides the query while encoder states supply keys and values.

<figure class="attention-example" aria-labelledby="attention-example-title">
  <figcaption id="attention-example-title">Example: resolving a pronoun with attention</figcaption>
  <p class="attention-example__sentence">After Maya placed the <mark>trophy</mark> beside the <mark>suitcase</mark>, she realized that <mark class="attention-example__query">it</mark> was too <mark>large</mark> to take on the train.</p>
  <div class="attention-example__query-label">Query: the token <strong>it</strong> asks which earlier token it refers to.</div>
  <div class="attention-example__weights" aria-label="Illustrative attention weights from it to earlier context: trophy 46 percent, large 22 percent, suitcase 12 percent, take 8 percent, train 5 percent, and remaining context 7 percent.">
    <div class="attention-example__weight"><span>trophy</span><b style="--weight: 46%">46%</b></div>
    <div class="attention-example__weight"><span>large</span><b style="--weight: 22%">22%</b></div>
    <div class="attention-example__weight"><span>suitcase</span><b style="--weight: 12%">12%</b></div>
    <div class="attention-example__weight"><span>take</span><b style="--weight: 8%">8%</b></div>
    <div class="attention-example__weight"><span>train</span><b style="--weight: 5%">5%</b></div>
    <div class="attention-example__weight"><span>other context</span><b style="--weight: 7%">7%</b></div>
  </div>
  <p class="attention-example__takeaway">The largest weight falls on <strong>trophy</strong>, so the contextual representation of <strong>it</strong> carries information that it refers to the trophy. These are illustrative weights, not measurements from a specific trained model.</p>
</figure>

<figure class="attention-example" aria-labelledby="technical-attention-example-title">
  <figcaption id="technical-attention-example-title">Example: connecting a conclusion to evidence</figcaption>
  <p class="attention-example__sentence">When the model reads a long <mark>research paper</mark>, it uses attention to connect the final <mark class="attention-example__query">conclusion</mark> with <mark>evidence</mark> introduced many paragraphs <mark>earlier</mark>.</p>
  <div class="attention-example__query-label">Query: the token <strong>conclusion</strong> searches the preceding context for the information that supports it.</div>
  <div class="attention-example__weights" aria-label="Illustrative attention weights from conclusion to earlier context: evidence 41 percent, research paper 20 percent, introduced 16 percent, earlier 13 percent, and remaining context 10 percent.">
    <div class="attention-example__weight"><span>evidence</span><b style="--weight: 41%">41%</b></div>
    <div class="attention-example__weight"><span>research paper</span><b style="--weight: 20%">20%</b></div>
    <div class="attention-example__weight"><span>introduced</span><b style="--weight: 16%">16%</b></div>
    <div class="attention-example__weight"><span>earlier</span><b style="--weight: 13%">13%</b></div>
    <div class="attention-example__weight"><span>other context</span><b style="--weight: 10%">10%</b></div>
  </div>
  <p class="attention-example__takeaway">A later token can pull supporting signals from much earlier context, which is one reason attention is useful for long documents and multi-step reasoning.</p>
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
