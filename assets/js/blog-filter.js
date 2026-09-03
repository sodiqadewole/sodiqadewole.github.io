(() => {
  const currentPage = document.querySelector('.blog-filter__link[aria-current="page"]');

  if (!currentPage) return;

  currentPage.closest('.blog-filter')?.querySelectorAll('details').forEach((details) => {
    if (details.contains(currentPage)) details.open = true;
  });
})();
