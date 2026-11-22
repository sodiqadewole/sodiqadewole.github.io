(() => {
  // Post pages: expand the sidebar tree down to the current article.
  const currentPage = document.querySelector('.blog-filter__link[aria-current="page"]');
  if (currentPage) {
    currentPage.closest('.blog-filter')?.querySelectorAll('details').forEach((details) => {
      if (details.contains(currentPage)) details.open = true;
    });
  }

  // Blog index: topic chips + search. Topics come from _data/blog-categories.yml.
  const home = document.querySelector('.blog-home');
  if (!home) return;

  const chips = [...home.querySelectorAll('.blog-chip')];
  const items = [...home.querySelectorAll('.blog-doc-item')];
  const search = home.querySelector('#blog-search');
  const status = home.querySelector('#blog-status');
  let topic = 'all';

  const apply = () => {
    const query = search.value.trim().toLowerCase();
    let shown = 0;
    items.forEach((item) => {
      const match = (topic === 'all' || item.dataset.topic === topic) &&
        (!query || item.dataset.search.includes(query));
      item.hidden = !match;
      if (match) shown += 1;
    });
    chips.forEach((chip) => chip.classList.toggle('is-active', chip.dataset.topic === topic));
    status.textContent = shown ? '' : 'No articles match your filter.';
  };

  const setTopic = (value, updateHash = true) => {
    topic = chips.some((chip) => chip.dataset.topic === value) ? value : 'all';
    if (updateHash) history.replaceState(null, '', topic === 'all' ? location.pathname : `#${topic}`);
    apply();
  };

  chips.forEach((chip) => chip.addEventListener('click', () => setTopic(chip.dataset.topic)));
  search.addEventListener('input', apply);
  const allLink = document.querySelector('.blog-filter__all');
  allLink?.addEventListener('click', (event) => {
    event.preventDefault();
    search.value = '';
    setTopic('all');
  });

  // Selecting a topic chip also opens that topic in the sidebar tree.
  chips.forEach((chip) => chip.addEventListener('click', () => {
    const summaries = [...document.querySelectorAll('.blog-filter > .blog-filter__tree > .blog-filter__tree-item > details')];
    summaries.forEach((d) => { d.open = chip.dataset.topic === 'all' ? false : d.dataset.topic === chip.dataset.topic; });
  }));

  setTopic(location.hash.slice(1), false);
})();
