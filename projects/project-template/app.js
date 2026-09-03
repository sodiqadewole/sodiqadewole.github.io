const summary = document.querySelector('#project-summary');

fetch('data/sample.json')
  .then((response) => response.json())
  .then((project) => {
    summary.innerHTML = Object.entries(project)
      .map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`)
      .join('');
  })
  .catch(() => {
    summary.innerHTML = '<dt>Status</dt><dd>Sample data could not be loaded.</dd>';
  });