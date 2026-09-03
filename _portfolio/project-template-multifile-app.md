---
title: "Template: Multi-file Project or App"
excerpt: "A reusable scaffold for publishing a project page with supporting app files, data, assets, and documentation."
collection: portfolio
permalink: /portfolio/project-template-multifile-app/
---

Use this template when a project is more than a single write-up. It gives each project a portfolio page, a static app entry point, supporting JavaScript/CSS/data files, and documentation that explains how the pieces fit together.

[Open the template app](/projects/project-template/)

## Project Structure

```text
_portfolio/project-template-multifile-app.md
projects/project-template/
  README.md
  index.html
  app.js
  styles.css
  data/sample.json
  docs/ARCHITECTURE.md
```

## How to Use It

1. Copy `projects/project-template/` to a new folder such as `projects/my-project/`.
2. Create a matching portfolio page in `_portfolio/`.
3. Link from the portfolio page to `/projects/my-project/`.
4. Add screenshots, notebooks, model outputs, or datasets under the project folder.
5. Run `bundle exec jekyll build` and check both `/portfolio/` and the project URL.

## When to Use This Pattern

This pattern works well for interactive demos, notebook exports, small static apps, model cards, project documentation, and any project that needs multiple supporting files.