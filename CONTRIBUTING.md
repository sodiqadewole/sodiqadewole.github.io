# Contributing

Thanks for your interest in improving this site.

## How to contribute

1. Open an issue describing the bug or improvement.
2. Create a branch from `main` (or your active default branch).
3. Make focused changes with clear commit messages.
4. Open a pull request with:
	- what changed
	- why it changed
	- screenshots (for visual updates), if applicable

## Local validation before PR

- Ensure JavaScript assets build successfully (`npm run build:js`).
- Ensure the Jekyll site builds/serves (`bundle exec jekyll serve`).
- Avoid committing generated artifacts (`_site/`, `.sass-cache/`, `.bundle/`, `vendor/bundle/`).

## Content conventions

- Add pages under `_pages/`.
- Add posts under `_posts/` using date-prefixed filenames.
- Add collection entries to their matching collection folder (`_publications/`, `_talks/`, `_teaching/`, `_portfolio/`).

Please keep PRs scoped and easy to review.

