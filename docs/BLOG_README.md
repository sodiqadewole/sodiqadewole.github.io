# Blog Authoring README

This site uses a Markdown-first blog system. Posts live in `_posts/`, categories
are defined in `_data/blog-categories.yml`, and the main blog page at `/blog/`
groups posts automatically by category and section.

## Create a New Blog Post

1. Copy the template from `docs/content-templates/blog-post.md` into `_posts/`.
2. Rename the file with the Jekyll post date prefix:

```text
_posts/YYYY-MM-DD-short-post-slug.md
```

Example:

```text
_posts/2026-09-01-attention-from-scratch.md
```

3. Fill in the front matter:

```yaml
---
title: "Attention From Scratch"
date: 2026-09-01
permalink: /posts/2026/09/attention-from-scratch/
blog_category: llm-from-scratch
blog_section: Transformer Building Blocks
blog_summary: "Build scaled dot-product attention step by step, from token embeddings to contextual representations."
read_time: true
tags:
  - Transformers
  - Attention
  - LLM From Scratch
---
```

4. Add the table of contents include below the front matter:

```liquid
{% include toc %}
```

5. Write the article in Markdown.

Use normal Markdown for prose, headings, links, images, tables, and fenced code
blocks. MathJax is enabled, so inline math can use `$x_i$` and display math can
use `$$...$$`.

## Front Matter Fields

Required fields:

- `title`: Display title for the post.
- `date`: Publication date. Keep this aligned with the filename date.
- `permalink`: Stable public URL.
- `blog_category`: Must match an `id` in `_data/blog-categories.yml`.

Recommended fields:

- `blog_section`: Subtopic shown under the category on the blog page.
- `blog_summary`: Short card summary shown on `/blog/`.
- `read_time: true`: Enables estimated reading time.
- `tags`: Search and taxonomy metadata.

## Choose a Category and Section

The category controls the top-level grouping on `/blog/` and the left topic
browser. Use one of the IDs from `_data/blog-categories.yml`, for example:

```yaml
blog_category: transformers
```

The section creates a nested subtopic inside that category:

```yaml
blog_section: Attention
```

Posts with the same `blog_category` and `blog_section` are grouped together.
If `blog_section` is omitted, the post appears directly under its category.

## Add a New Blog Category

1. Open `_data/blog-categories.yml`.
2. Add a new item with a stable lowercase ID, a display title, and a short
   description:

```yaml
- id: mlops
  title: MLOps
  description: Production machine learning systems, deployment, monitoring, and operations.
```

3. Use the new category ID in a post:

```yaml
blog_category: mlops
blog_section: Model Serving
```

The main blog page will automatically show the new category because `_pages/blog.html`
loops over `_data/blog-categories.yml`. The left topic browser will also pick it
up automatically.

## Add Visuals and Interactive Walkthroughs

Keep the article content in Markdown and reuse shared components for richer
visuals.

Available options:

- Mermaid diagrams with fenced `mermaid` blocks.
- Plotly charts with fenced `plotly` JSON blocks.
- MathJax equations with `$...$` and `$$...$$`.
- Reusable interactive walkthroughs with Liquid includes, for example:

```liquid
{% include step-through.html id="transformer-attention" data="attention-walkthrough" title="Follow one token through self-attention" %}
```

For interactive walkthroughs, keep the data in `_data/interactive/` and the
component code in `_includes/`, `_sass/`, and `assets/js/` instead of embedding
large scripts directly inside posts.

## Validate Before Publishing

Run the content validator and Jekyll build before committing:

```bash
npm run validate:content
bundle exec jekyll build
```

For local preview, use the Docker/local config combination so local assets are
served from `localhost`:

```bash
bundle exec jekyll serve --config _config.yml,_config_docker.yml --host localhost --port 4000
```

Open `http://localhost:4000/blog/` and check that the post appears in the
expected category and section.

## Publishing Checklist

- Filename starts with the same date as `date` in front matter.
- `permalink` is unique and stable.
- `blog_category` matches an existing category ID.
- `blog_summary` is short enough to fit cleanly in a card.
- Math, diagrams, and interactive components render locally.
- `npm run validate:content` passes.
- `bundle exec jekyll build` passes.