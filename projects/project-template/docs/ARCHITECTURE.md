# Architecture Notes

The template keeps the portfolio page and runnable project files separate.

- `_portfolio/*.md` explains the project and links to the app.
- `projects/<project>/index.html` is the app entry point.
- `projects/<project>/app.js` contains browser behavior.
- `projects/<project>/styles.css` contains project-specific presentation.
- `projects/<project>/data/` stores small static data files.
- `projects/<project>/docs/` stores implementation notes, model cards, or setup instructions.

For larger apps, build the app separately and place the static build output in the project folder.