# Recreate the Info page scroll highlight effect

## Handoff to the implementing developer

Recreate the character-by-character colored highlight removal effect from this repository's Info page. At the top of the document, each character has a colored background and white text. Scrolling down wipes the colored background away from left to right, in reading order. Once a character's background is completely cleared, its text becomes black. Scrolling up reverses the effect exactly.

This is an implementation handoff. It does not change the existing website.

## Reference files and scope

- `info.html`: the paragraph at `.page--info .about`, including emphasis and superscripts.
- `java.js`: the block beginning with `// Info-page character highlights follow document scrolling in both directions.` inside `window.initializeClassPage`.
- `styles.css`: the `.reading-word`, `.reading-space`, and `.reading-character` rules.

The effect uses plain HTML, CSS, and JavaScript. No animation library, canvas, images, or build step is required. Picture rain and capsule navigation are separate effects and are not required to recreate this one. When porting it, use the standalone initializer below rather than copying all of `java.js`.

## Visual and interaction requirements

1. Render one copy of the text. Do not overlay duplicate text or use pseudo-elements containing copies of it.
2. Preserve the paragraph's original content, spaces, punctuation, italic emphasis, and superscripts. Wrap text nodes rather than replacing the paragraph's HTML with plain text.
3. Wrap each word in an inline-block so it stays together when lines wrap. Preserve whitespace in separate wrappers. Each character, including whitespace, receives its own span and participates in scroll progress.
4. Start with a fully colored background and white text. Within the active character, move a hard gradient boundary from left to right: transparent on the left, marker color on the right. Keep its text white until the fill reaches 100%, then switch to black.
5. Map the entire document's scroll range to the entire paragraph. This is not a viewport-entry trigger or a timed animation. Scrolling down removes highlights; scrolling back restores them.
6. Use deterministic colors. For each exact character, occurrences 1–5 share one color, occurrences 6–10 share another, and so on. Occurrences are counted across the paragraph; case, punctuation, and spaces are distinct. Different character/batch groups receive distinct colors through collision resolution.
7. Recalculate on scrolling, resizing, content resizing, and font readiness. Batch scroll updates with `requestAnimationFrame` and write only changed character states.
8. Clean up listeners, observers, and pending frames before reinitializing or leaving the page.

## HTML and typography

Retain the existing Info page markup when working in this repository. For another project, use this minimal structure, replacing the example copy with the intended content:

```html
<main class="page page--info">
  <h1>Info</h1>
  <p class="about">Your introductory text with <em>emphasis</em> and references<sup>(01)</sup>.</p>
</main>
```

The reference uses Schibsted Grotesk with regular and italic text. Preserve the destination's intended paragraph sizing and spacing. A light background is needed for the final black text to remain readable. The exact effect can temporarily leave white letter portions on the cleared background while a character is partially wiped; this follows the reference rather than interpolating text color.

## CSS

Add these rules after the paragraph's normal typography rules:

```css
.about .reading-word { display: inline-block; white-space: nowrap; }
.about .reading-space { white-space: pre-wrap; }
.about .reading-character {
  position: relative;
  color: #fff;
  background: linear-gradient(
    to right,
    transparent var(--read, 0%),
    var(--marker-color) var(--read, 0%)
  );
}
```

`--read` is the percentage already cleared, not the percentage still highlighted. At `0%`, the entire character background is colored; at `100%`, it is transparent. Do not add transitions or easing: the appearance should track the current scroll position immediately.

## JavaScript

For a standalone port, place this script after the paragraph or load it with `defer`. The implementation below copies the reference effect and gives it a dedicated lifecycle. In this repository, retain the existing shared `listeners` controller and initializer instead of installing a second copy of the effect.

```js
window.initializeInfoHighlights = function () {
  window.infoHighlightListeners?.abort();
  const listeners = new AbortController();
  window.infoHighlightListeners = listeners;
const about = document.querySelector('.page--info .about');
if (about) {
  const CHANNEL_STEPS = 256;
  const COLOR_COUNT = CHANNEL_STEPS ** 3;
  const characterSeed = character => {
    let hash = 2166136261;
    for (const symbol of character) hash = Math.imul(hash ^ symbol.codePointAt(0), 16777619);
    hash = Math.imul(hash ^ (hash >>> 16), 2246822507);
    return (hash ^ (hash >>> 13)) >>> 0;
  };
  const hexFromIndex = index => `#${[0, 1, 2].map(power =>
    (Math.floor(index / CHANNEL_STEPS ** power) % CHANNEL_STEPS).toString(16).padStart(2, '0')
  ).join('')}`;
  const characterColors = characters => {
    const occurrences = new Map();
    const groups = new Map();
    const reserved = new Set();
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
  };

  // Wrap text nodes only, leaving emphasis and superscripts intact.
  if (!about.querySelector('[data-reading-character]')) {
    const walker = document.createTreeWalker(about, NodeFilter.SHOW_TEXT);
    const textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);
    for (const node of textNodes) {
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
  const characters = Array.from(about.querySelectorAll('[data-reading-character]'));
  const colors = characterColors(characters.map(character => character.textContent));
  characters.forEach((character, index) => character.style.setProperty('--marker-color', colors[index]));
  const previousFill = new Array(characters.length).fill(-1);
  let frame = 0;
  const update = () => {
    frame = 0;
    const runway = document.documentElement.scrollHeight - window.innerHeight;
    const progress = runway > 0 ? Math.max(0, Math.min(1, window.scrollY / runway)) : 0;
    const read = progress * characters.length;
    characters.forEach((character, index) => {
      const fill = Math.max(0, Math.min(1, read - index));
      if (previousFill[index] === fill) return;
      previousFill[index] = fill;
      character.style.setProperty('--read', `${fill * 100}%`);
      character.style.color = fill < 1 ? '#ffffff' : '#000000';
    });
  };
  const schedule = () => {
    if (!frame && !listeners.signal.aborted) frame = requestAnimationFrame(update);
  };
  const observer = new ResizeObserver(schedule);
  observer.observe(document.querySelector('main.page'));
  observer.observe(about);
  window.addEventListener('scroll', schedule, { passive: true, signal: listeners.signal });
  window.addEventListener('resize', schedule, { signal: listeners.signal });
  document.fonts?.ready.then(schedule);
  listeners.signal.addEventListener('abort', () => {
    cancelAnimationFrame(frame);
    observer.disconnect();
  }, { once: true });
  update();
}
};
window.initializeInfoHighlights();
```

For client-side navigation, call `window.infoHighlightListeners?.abort()` before removing the page and call `window.initializeInfoHighlights()` after mounting the new paragraph. In a component-based application, keep the controller local to the component and abort it during cleanup. If replacing paragraph content, supply fresh, unwrapped markup and reinitialize.

The reference splits text with `Array.from`, which iterates Unicode code points. If the destination includes combined emoji or combining marks, consider grapheme segmentation with `Intl.Segmenter` so each visible glyph stays together; that would be an extension to the reference implementation.

## Scroll mapping and layout

The calculation is:

```text
runway = document scroll height - viewport height
progress = clamp(scroll position / runway, 0, 1)
read = progress × number of characters
character fill = clamp(read - character index, 0, 1)
```

For example, with 100 characters at 25.5% document progress, the first 25 are fully cleared and black, the next is half cleared with white text, and the rest remain highlighted.

Ensure the destination has enough document height to scroll. If all content fits in the viewport, the reference deliberately keeps progress at zero and all characters remain highlighted. If the destination needs extra scroll distance, add intentional spacing or a minimum page height appropriate to its layout, then check desktop and mobile. Do not introduce a sticky paragraph or a different scroll container unless changing the design is explicitly intended.

Other content that changes document height changes this effect's mapping. If the destination dynamically resizes content outside `main.page`, observe the relevant layout container as well. A nested scrolling panel requires adapting both the scroll listener and the scroll-range calculation.

## Verification checklist

Serve the destination with its usual development server. For this plain HTML repository, run `python3 -m http.server 8000` from the root and open `http://localhost:8000/info.html`.

- At the top, every character has a colored background and white text.
- Slow downward scrolling clears backgrounds in reading order, including a partial wipe within the active character.
- Completed characters become black; at the document bottom, all highlights are cleared.
- Upward scrolling restores the colors and white text in reverse order.
- Repeated characters share colors in groups of five occurrences, and assignments remain stable when reinitializing unchanged text.
- Italics, superscripts, punctuation, spaces, selection, and copying still work without duplicated text.
- Desktop and narrow mobile widths preserve word wrapping; resizing and font loading recalculate progress.
- A page with no scroll range remains fully highlighted without division errors.
- Direct loading, restored scroll positions, and navigating away and back initialize correctly without nested wrappers or duplicate handlers.
- The console has no errors, and unrelated pages are unaffected.

These are acceptance checks for the implementing developer, not a claim that browser validation has been performed by this document.
