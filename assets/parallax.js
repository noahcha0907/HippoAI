(() => {
  'use strict';
  const hero = document.querySelector('.hero');
  const title = document.querySelector('#hero-title');
  if (!hero || !title) return;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let pending = false;

  function update() {
    pending = false;
    const bounds = hero.getBoundingClientRect();
    const distance = Math.min(Math.max(-bounds.top, 0), bounds.height);
    // Move the title more slowly than its surrounding grid as the hero exits.
    const offset = reduced.matches ? 0 : distance * 0.3;
    title.style.setProperty('--hero-parallax', `${offset.toFixed(2)}px`);
  }

  function schedule() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(update);
  }

  window.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('resize', schedule);
  window.addEventListener('pageshow', schedule);
  reduced.addEventListener('change', schedule);
  update();
})();
