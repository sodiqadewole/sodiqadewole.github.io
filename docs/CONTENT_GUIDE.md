# Content Guide

Content lives in Jekyll collections. Use the templates in
`docs/content-templates/` instead of starting from an existing post, because
each collection has a different front matter contract.

## Workflow

1. Copy the relevant template into its collection.
2. Choose a lowercase, URL-friendly filename and permalink.
3. Replace all empty or placeholder values.
4. Write the content below the closing `---`.
5. Run `npm run validate:content`.
6. Build locally with `bundle exec jekyll build --trace`.

## Content types

| Type | Collection | Required fields |
| --- | --- | --- |
| Blog post | `_posts/` | `title`, `date`, `permalink` |
| Portfolio | `_portfolio/` | `title`, `collection`, `permalink` |
| Publication | `_publications/` | `title`, `collection`, `category`, `date`, `permalink` |
| Talk | `_talks/` | `title`, `collection`, `type`, `date`, `permalink` |
| Teaching | `_teaching/` | `title`, `collection`, `date`, `permalink` |

Existing URLs are part of the public site. Do not change an old permalink
without adding a redirect. Posts must use a filename beginning with the same
date as their front matter, for example `2026-08-27-new-post.md`.