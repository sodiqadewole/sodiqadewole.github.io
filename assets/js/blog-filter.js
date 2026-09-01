(() => {
  const filter = document.querySelector('.blog-filter');
  const categories = Array.from(document.querySelectorAll('[data-blog-category]'));

  if (!filter) return;

  const getTopicLabel = (topic) => topic.querySelector('[data-filter-label]').dataset.filterLabel;
  const toggleTopic = (toggle) => {
    const subtopics = document.getElementById(toggle.getAttribute('aria-controls'));
    const expanded = toggle.getAttribute('aria-expanded') === 'true';
    const itemLabel = toggle.dataset.toggleLabel || 'subtopics';

    toggle.setAttribute('aria-expanded', String(!expanded));
    toggle.setAttribute('aria-label', `${expanded ? 'Expand' : 'Collapse'} ${getTopicLabel(toggle.closest('.blog-filter__topic'))} ${itemLabel}`);
    toggle.textContent = expanded ? '+' : '-';
    subtopics.hidden = expanded;
  };

  filter.addEventListener('click', (event) => {
    const toggle = event.target.closest('.blog-filter__toggle');
    if (toggle) {
      toggleTopic(toggle);
      return;
    }

    const topicButton = event.target.closest('[data-topic-toggle]');
    if (topicButton) {
      const topicToggle = topicButton.closest('.blog-filter__topic').querySelector('.blog-filter__toggle');
      if (topicToggle) toggleTopic(topicToggle);
      return;
    }

    const button = event.target.closest('[data-filter-category]');
    if (!button) return;

    if (categories.length === 0) return;

    const categoryId = button.dataset.filterCategory;
    const sectionId = button.dataset.filterSection;

    if (categoryId !== 'all') {
      const topic = button.closest('.blog-filter__topic');
      const toggle = topic && topic.querySelector('.blog-filter__toggle');
      const subtopics = topic && topic.querySelector('.blog-filter__subtopics');
      if (toggle && subtopics) {
        toggle.setAttribute('aria-expanded', 'true');
        toggle.setAttribute('aria-label', `Collapse ${getTopicLabel(topic)} ${toggle.dataset.toggleLabel || 'subtopics'}`);
        toggle.textContent = '-';
        subtopics.hidden = false;
      }
    }

    categories.forEach((category) => {
      const categoryMatches = categoryId === 'all' || category.dataset.blogCategory === categoryId;
      category.hidden = !categoryMatches;

      if (categoryMatches && categoryId !== 'all') {
        category.querySelector('.blog-tree__details').open = true;
      } else if (categoryId === 'all') {
        category.querySelector('.blog-tree__details').open = false;
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
