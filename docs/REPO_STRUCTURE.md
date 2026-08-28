# Repository Structure

This document describes the intended organization of this Jekyll site repository.

## Top-level source folders

- `_data/` — YAML/JSON data used by templates and navigation.
- `_drafts/` — unpublished drafts.
- `_includes/` — reusable template partials.
- `_layouts/` — page and post layouts.
- `_pages/` — standalone pages.
- `_posts/` — blog posts.
- `_portfolio/` — portfolio collection content.
- `_publications/` — publications collection content.
- `_talks/` — talks collection content.
- `_teaching/` — teaching collection content.
- `_sass/` — Sass sources for styles.
- `assets/` — CSS/JS/images used by pages.
- `files/` and `images/` — static downloadable/public assets.

## Utility and tooling folders

- `scripts/` — repo maintenance scripts.
- `markdown_generator/` — optional markdown generation helpers.
- `docs/content-templates/` — starter front matter for each content type.
- `talkmap/` — talk-map related source and helper content.

## Build and dependency files

- `_config.yml` — primary Jekyll configuration.
- `Gemfile` / `Gemfile.lock` — Ruby dependencies.
- `package.json` — JS bundling dependencies and scripts.
- `Dockerfile` / `docker-compose.yaml` — containerized local run.

## Generated/local-only artifacts (not source)

These should stay untracked and are now covered by `.gitignore`:

- `_site/`
- `.sass-cache/`
- `.bundle/`
- `vendor/bundle/`
- `node_modules/`
- `.jekyll-cache/`

## Ongoing cleanup guidelines

- Keep source content in collection folders and `_pages/`.
- Keep generated outputs out of version control.
- Prefer adding new automation under `scripts/`.
- If a folder is temporary or experimental, prefix it in docs and add cleanup notes.

## Adding content

Start with the matching template in `docs/content-templates/`, copy it into the
corresponding collection, and replace every placeholder value:

- `post.md` -> `_posts/YYYY-MM-DD-title.md`
- `portfolio.md` -> `_portfolio/title.md`
- `publication.md` -> `_publications/title.md`
- `talk.md` -> `_talks/YYYY-MM-DD-title.md`
- `teaching.md` -> `_teaching/title.md`

Run `npm run validate:content` before building the site. The validator checks
front matter, dates, permalinks, post filename dates, and enabled navigation
links. Warnings identify placeholders that should be cleaned up but do not
prevent an existing site from being validated.
