(() => {
  const filter = document.querySelector('.blog-filter');
  const categories = Array.from(document.querySelectorAll('[data-blog-category]'));

  if (!filter || categories.length === 0) return;

  filter.addEventListener('click', (event) => {
    const button = event.target.closest('[data-filter-category]');
    if (!button) return;

    const categoryId = button.dataset.filterCategory;
    const sectionId = button.dataset.filterSection;

    categories.forEach((category) => {
      const categoryMatches = categoryId === 'all' || category.dataset.blogCategory === categoryId;
      category.hidden = !categoryMatches;

      if (categoryMatches && categoryId !== 'all') {
        category.querySelector('.blog-tree__details').open = true;
      }

      category.querySelectorAll('[data-blog-section]').forEach((section) => {
        section.hidden = Boolean(sectionId) && section.dataset.blogSection !== sectionId;
      });
    });

    filter.querySelectorAll('[data-filter-category]').forEach((item) => {
      item.classList.toggle('is-active', item === button);
    });
  });
})();
