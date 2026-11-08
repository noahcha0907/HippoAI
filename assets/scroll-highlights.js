/* Adapted from INFO-SCROLL-HIGHLIGHT-INSTRUCTIONS.md.
   Each heading wipes independently as it travels through the viewport. */
(() => {
  'use strict';
  window.sectionHighlightLifecycle?.abort();
  window.sectionHighlightListeners?.abort();
  const lifecycle = new AbortController();
  window.sectionHighlightLifecycle = lifecycle;

  const CHANNEL_STEPS = 256;
  const COLOR_COUNT = CHANNEL_STEPS ** 3;
  function characterSeed(character) {
    let hash = 2166136261;
    for (const symbol of character) hash = Math.imul(hash ^ symbol.codePointAt(0), 16777619);
    hash = Math.imul(hash ^ (hash >>> 16), 2246822507);
    return (hash ^ (hash >>> 13)) >>> 0;
  }
  function hexFromIndex(index) {
    return '#' + [0, 1, 2].map(power =>
      (Math.floor(index / CHANNEL_STEPS ** power) % CHANNEL_STEPS).toString(16).padStart(2, '0')
    ).join('');
  }
  function characterColors(characters) {
    const occurrences = new Map(), groups = new Map(), reserved = new Set();
    return characters.map(character => {
      const occurrence = occurrences.get(character) ?? 0;
      occurrences.set(character, occurrence + 1);
      const batch = Math.floor(occurrence / 5);
      const key = JSON.stringify([character, batch]);
      if (groups.has(key)) return groups.get(key);
      if (reserved.size >= COLOR_COUNT) throw new Error('Character highlight palette exhausted');
      let index = (characterSeed(character) + batch * 3 * (1 + CHANNEL_STEPS + CHANNEL_STEPS ** 2)) % COLOR_COUNT;
      let hex = hexFromIndex(index);
      while (reserved.has(hex)) {
        index = (index + 1) % COLOR_COUNT;
        hex = hexFromIndex(index);
      }
      reserved.add(hex);
      groups.set(key, hex);
      return hex;
    });
  }

  function wrapText(heading) {
    if (heading.querySelector('[data-reading-character]')) return;
    const walker = document.createTreeWalker(heading, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      const fragment = document.createDocumentFragment();
      for (const word of node.textContent.split(/(\s+)/).filter(Boolean)) {
        const wrapper = document.createElement('span');
        wrapper.className = /^\s+$/.test(word) ? 'reading-space' : 'reading-word';
        for (const character of Array.from(word)) {
          const span = document.createElement('span');
          span.className = 'reading-character';
          span.dataset.readingCharacter = '';
          span.textContent = character;
          wrapper.append(span);
        }
        fragment.append(wrapper);
      }
      node.replaceWith(fragment);
    }
  }

  window.initializeSectionHighlights = function () {
    window.sectionHighlightListeners?.abort();
    const listeners = new AbortController();
    window.sectionHighlightListeners = listeners;
    const headings = [...document.querySelectorAll('[data-scroll-highlight]')];
    if (!headings.length) return;
    headings.forEach(wrapText);
    const sections = headings.map(heading => {
      const characters = [...heading.querySelectorAll('[data-reading-character]')];
      const colors = characterColors(characters.map(character => character.textContent));
      characters.forEach((character, index) => character.style.setProperty('--marker-color', colors[index]));
      return { heading, characters, previousFill: new Array(characters.length).fill(-1) };
    });
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    let frame = 0;

    function update() {
      frame = 0;
      if (listeners.signal.aborted) return;
      // Start near the bottom of the viewport and finish above its center.
      // A heading's state never depends on document length or other headings.
      const start = window.innerHeight * 0.85;
      const end = window.innerHeight * 0.35;
      const travel = Math.max(1, start - end);
      const progress = sections.map(({ heading }) => reduced.matches ? 1 :
        Math.max(0, Math.min(1, (start - heading.getBoundingClientRect().top) / travel)));
      sections.forEach(({ characters, previousFill }, sectionIndex) => {
        const read = progress[sectionIndex] * characters.length;
        characters.forEach((character, index) => {
          const fill = Math.max(0, Math.min(1, read - index));
          if (previousFill[index] === fill) return;
          previousFill[index] = fill;
          character.style.setProperty('--read', `${fill * 100}%`);
          character.style.color = fill < 1 ? '#ffffff' : '#000000';
        });
      });
    }
    function schedule() {
      if (!frame && !listeners.signal.aborted) frame = requestAnimationFrame(update);
    }
    const observer = new ResizeObserver(schedule);
    // Re-measure positions when responsive layouts or the walkthrough change size.
    observer.observe(document.body);
    headings.forEach(heading => observer.observe(heading));
    window.addEventListener('scroll', schedule, { passive: true, signal: listeners.signal });
    window.addEventListener('resize', schedule, { signal: listeners.signal });
    reduced.addEventListener('change', schedule, { signal: listeners.signal });
    document.fonts?.ready.then(schedule);
    listeners.signal.addEventListener('abort', () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    }, { once: true });
    update();
  };

  window.addEventListener('pagehide', () => window.sectionHighlightListeners?.abort(), { signal: lifecycle.signal });
  window.addEventListener('pageshow', () => {
    if (window.sectionHighlightListeners?.signal.aborted) window.initializeSectionHighlights();
  }, { signal: lifecycle.signal });
  window.initializeSectionHighlights();
})();
