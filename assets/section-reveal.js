(() => {
  'use strict';
  const sections = [...document.querySelectorAll('.content-site > section:not(.hero):not(.page-masthead)')];
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  if (!sections.length || reduced.matches || !('IntersectionObserver' in window)) return;
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.classList.add('is-visible');
      observer.unobserve(entry.target);
    }
  }, { rootMargin: '0px 0px -35px 0px', threshold: 0 });

  for (const section of sections) {
    // Keep already visible sections stable on load and at restored scroll positions.
    if (section.getBoundingClientRect().top < innerHeight - 35) section.classList.add('is-visible');
    section.classList.add('section-reveal');
    if (!section.classList.contains('is-visible')) observer.observe(section);
  }
  document.addEventListener('focusin', event => {
    const section = event.target.closest('.section-reveal');
    if (!section) return;
    section.classList.add('is-visible');
    observer.unobserve(section);
  });
  reduced.addEventListener('change', event => {
    if (!event.matches) return;
    observer.disconnect();
    sections.forEach(section => section.classList.add('is-visible'));
  });
  window.addEventListener('pagehide', () => observer.disconnect());
  window.addEventListener('pageshow', event => {
    if (!event.persisted || reduced.matches) return;
    sections.filter(section => !section.classList.contains('is-visible')).forEach(section => observer.observe(section));
  });
})();
