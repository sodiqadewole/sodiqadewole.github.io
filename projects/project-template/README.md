# Multi-file Project Template

This folder is a starter scaffold for projects that need more than a portfolio markdown page.

## Files

- `index.html`: Static app or landing page for the project.
- `styles.css`: Project-specific styling.
- `app.js`: Project-specific JavaScript.
- `data/sample.json`: Example structured data loaded by the app.
- `docs/ARCHITECTURE.md`: Notes about the project design and file responsibilities.

## Publishing Flow

1. Duplicate this folder and rename it for the project.
2. Update `index.html`, `app.js`, and `styles.css`.
3. Add supporting data, assets, exported notebooks, or screenshots.
4. Create a `_portfolio/<project-name>.md` page that links to the folder.
5. Build with `bundle exec jekyll build`.
