(() => {
  'use strict';
  const header = document.querySelector('.site-header');
  if (!header) return;
  let previous = Math.max(0, window.scrollY);
  let pending = false;

  function update() {
    pending = false;
    const current = Math.max(0, window.scrollY);
    const delta = current - previous;
    if (current <= 24 || header.contains(document.activeElement)) {
      header.classList.remove('is-hidden');
    } else if (Math.abs(delta) >= 6) {
      header.classList.toggle('is-hidden', delta > 0);
    } else {
      return;
    }
    previous = current;
  }

  window.addEventListener('scroll', () => {
    if (!pending) {
      pending = true;
      requestAnimationFrame(update);
    }
  }, { passive: true });
  header.addEventListener('focusin', () => {
    header.classList.remove('is-hidden');
    previous = Math.max(0, window.scrollY);
  });
})();
