(() => {
  'use strict';
  const root = document.querySelector('.walkthrough');
  if (!root) return;
  const stages = [...root.querySelectorAll('[data-step]')];
  const panels = [...root.querySelectorAll('.walkthrough-panel')];
  const previous = root.querySelector('#previous-stage');
  const next = root.querySelector('#next-stage');
  const status = root.querySelector('#walkthrough-status');
  let selected = 0;

  function select(index) {
    selected = Math.max(0, Math.min(index, stages.length - 1));
    stages.forEach((button, i) => button.setAttribute('aria-pressed', String(i === selected)));
    panels.forEach((panel, i) => { panel.hidden = i !== selected; });
    previous.disabled = selected === 0;
    next.textContent = selected === stages.length - 1 ? 'Restart walkthrough ↺' : 'Next stage →';
    status.textContent = `${stages[selected].textContent} stage selected`;
  }

  stages.forEach((button, index) => {
    button.addEventListener('click', () => select(index));
    button.addEventListener('keydown', event => {
      let destination;
      if (event.key === 'ArrowRight') destination = (index + 1) % stages.length;
      if (event.key === 'ArrowLeft') destination = (index + stages.length - 1) % stages.length;
      if (event.key === 'Home') destination = 0;
      if (event.key === 'End') destination = stages.length - 1;
      if (destination === undefined) return;
      event.preventDefault();
      select(destination);
      stages[destination].focus();
    });
  });
  previous.addEventListener('click', () => select(selected - 1));
  next.addEventListener('click', () => select((selected + 1) % stages.length));
  root.querySelector('.walkthrough-controls').hidden = false;
  root.querySelector('.walkthrough-navigation').hidden = false;
  select(0);
})();
