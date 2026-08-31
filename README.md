# Sodiq Adewole — Personal Website

This repository contains the source for [https://sodiqadewole.github.io](https://sodiqadewole.github.io), built with Jekyll using the Academic Pages template as a base.

## Stack

- Jekyll and GitHub Pages for site generation and deployment
- Ruby dependencies managed by Bundler through `github-pages`
- npm used to build the JavaScript bundle
- Docker available for a repeatable local development environment

## Project layout

- `_pages/`: standalone pages (about, portfolio, talks, etc.)
- `_posts/`: blog posts
- `_publications/`, `_teaching/`, `_talks/`, `_portfolio/`: content collections
- `_includes/`, `_layouts/`, `_sass/`, `assets/`: theme and presentation layer
- `_data/`: YAML/JSON data for navigation and content metadata
- `files/`, `images/`: static assets
- `scripts/`: utility scripts used during maintenance
- `markdown_generator/` and `talkmap/`: optional content generation helpers

See [`docs/REPO_STRUCTURE.md`](docs/REPO_STRUCTURE.md) for a more detailed map and maintenance guidance.

## Quick start

Install Ruby, Bundler, Node.js, and npm, then run:

```powershell
bundle install
npm install
npm run build:js
npm run validate:content
bundle exec jekyll serve -l -H localhost
```

Open [http://localhost:4000](http://localhost:4000) after the server starts.

## Run locally

### Option 1: native Ruby/Jekyll

1. Install Ruby, Bundler, Node.js, and npm.
2. Install dependencies:
   - `bundle install`
   - `npm install`
3. Build JavaScript bundle:
   - `npm run build:js`
4. Serve locally:
   - `bundle exec jekyll serve -l -H localhost`

The site is then available at `http://localhost:4000`.

### Validate and build without serving

```powershell
npm run validate:content
bundle exec jekyll build --trace
```

The content validator checks required front matter, dates, duplicate
permalinks, post filename dates, placeholders, and navigation links. It is
intended to catch content mistakes before the GitHub Pages build.

### Option 2: Docker

Use `docker compose up --build` in the repository root and open `http://localhost:4000`.

## Add content

Use the templates in [`docs/content-templates/`](docs/content-templates/) and
follow [`docs/CONTENT_GUIDE.md`](docs/CONTENT_GUIDE.md). For blog-specific
instructions, including adding posts and categories, see
[`docs/BLOG_README.md`](docs/BLOG_README.md). The main content types
are:

- Blog posts in `_posts/`
- Portfolio projects in `_portfolio/`
- Publications in `_publications/`
- Talks in `_talks/`
- Teaching entries in `_teaching/`

After adding content, run `npm run validate:content` before building. Keep
existing permalinks stable because they are public URLs.

## Deployment

The repository is configured for GitHub Pages. Pushes to the `initial_commit`
branch trigger the Pages build and deployment workflow. A failed build should
first be reproduced with `bundle exec jekyll build --trace`, then checked with
`npm run validate:content`.

## Repository hygiene

This repository now ignores generated or machine-specific artifacts such as:

- `_site/`
- `.sass-cache/`
- `.bundle/`
- `vendor/bundle/`
- `node_modules/`

Generated artifacts are ignored and should not be committed. If an artifact
appears in a future change, check `.gitignore` before staging it.

## Notes

- Main site configuration lives in `_config.yml`.
- JavaScript build scripts are in `package.json`.
- Ruby dependencies are managed through `Gemfile` and `Gemfile.lock`.
- Content requirements and templates live in `docs/CONTENT_GUIDE.md`.
