/* Live demo: ask the broken model, watch Hippo repair it, ask again.
 *
 * Data comes from the real API's responses. By default the page reads a saved run exported by
 * backend/eval/export_demo.py (backend/demo_export/), so it needs no GPU or backend.
 * Query parameters:
 *   ?run=<repair_id>          pick a saved run (default: the first repaired run, else the newest)
 *   ?data=<path>              folder holding the export (default: backend/demo_export)
 *   ?api=http://localhost:8000&run=<repair_id>   read a run from the live real API instead
 * The page shows what the run actually did. If Hippo refused every edit, step 3 says so.
 */
(() => {
  'use strict';
  const root = document.querySelector('#live-demo');
  if (!root) return;

  const params = new URLSearchParams(location.search);
  const api = (params.get('api') || '').replace(/\/$/, '');
  const dataDir = (params.get('data') || 'backend/demo_export').replace(/\/$/, '');
  const LETTERS = ['A', 'B', 'C', 'D'];
  const REPLAY_MS = 9000;

  const $ = sel => root.querySelector(sel);
  const pct = x => (x == null ? '—' : `${Math.round(x * 100)}%`);
  const num = x => Number(x).toLocaleString('en-US');

  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value == null || value === false) continue;
      if (key === 'class') node.className = value;
      else if (key === 'style') node.style.cssText = value;
      else node.setAttribute(key, value === true ? '' : value);
    }
    for (const child of children.flat()) {
      if (child == null || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  async function getJSON(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    return response.json();
  }
  const optional = promise => promise.catch(() => null);

  async function loadRun() {
    if (api) {
      const id = params.get('run');
      if (!id) throw new Error('Live mode needs ?run=<repair_id> next to ?api=.');
      const [status, report, events, compare, attribution] = await Promise.all([
        getJSON(`${api}/repairs/${id}`),
        getJSON(`${api}/repairs/${id}/report`),
        getJSON(`${api}/repairs/${id}/events`),
        getJSON(`${api}/compare`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ repair_id: id, count: 6 }),
        }),
        optional(getJSON(`${api}/repairs/${id}/attribution`)),
      ]);
      return { source: `Live API · ${api}`, status, report, events, compare, attribution, runs: [] };
    }
    const index = await getJSON(`${dataDir}/index.json`);
    const runs = index.runs || [];
    if (!runs.length) throw new Error('The export has no runs. Run python -m eval.export_demo.');
    const wanted = params.get('run');
    const run = runs.find(r => r.id === wanted)
      || runs.find(r => r.outcome === 'repaired')
      || runs[runs.length - 1];
    const folder = `${dataDir}/runs/${run.id}`;
    const [status, report, events, compare, attribution] = await Promise.all([
      getJSON(`${folder}/status.json`),
      getJSON(`${folder}/report.json`),
      getJSON(`${folder}/events.json`),
      getJSON(`${folder}/compare.json`),
      optional(getJSON(`${folder}/attribution.json`)),
    ]);
    return { source: 'Saved run', status, report, events, compare, attribution, runs, runId: run.id };
  }

  /* ---------- state ---------- */
  let data;
  let question = 0;
  let replayTimer = null;

  /* ---------- rendering ---------- */
  function renderMeta() {
    const { status, report, source, runs, runId } = data;
    const meta = $('.ld-meta');
    meta.replaceChildren(
      el('span', {}, source),
      el('span', {}, `Model ${status.model_id}`),
      el('span', {}, `Run ${status.id}`),
      report.minutes != null ? el('span', {}, `${report.minutes.toFixed(1)} min`) : null,
    );
    if (runs.length > 1) {
      const select = el('select', { 'aria-label': 'Choose a saved run' },
        runs.map(r => el('option', { value: r.id, selected: r.id === runId },
          `${r.id} · ${r.model_id} · ${r.outcome.replaceAll('_', ' ')}`)));
      select.addEventListener('change', () => {
        params.set('run', select.value);
        location.search = params.toString();
      });
      meta.append(select);
    }
    $('.ld-complaint').textContent = `“${status.complaint}”`;
  }

  function renderQuestionTabs() {
    const items = data.compare.items;
    const tabs = $('.ld-question-tabs');
    tabs.replaceChildren(...items.map((item, i) => {
      const button = el('button', { type: 'button', 'aria-pressed': String(i === question) },
        el('span', {}, item.category), shorten(item.question, 52));
      button.addEventListener('click', () => { question = i; resetAnswers(); renderQuestionTabs(); renderQuestion(); });
      return button;
    }));
  }

  function shorten(text, n) {
    return text.length > n ? `${text.slice(0, n - 1).trimEnd()}…` : text;
  }

  function renderQuestion() {
    const item = data.compare.items[question];
    root.querySelectorAll('.ld-question').forEach(box => {
      box.replaceChildren(
        el('p', { class: 'ld-q' }, item.question),
        el('ol', { class: 'ld-options' }, LETTERS.filter(l => item.options[l]).map(l =>
          el('li', { 'data-letter': l }, el('b', {}, l), item.options[l]))),
      );
    });
  }

  function markAnswer(box, answer, item, label) {
    box.querySelectorAll('li').forEach(li => {
      li.classList.remove('is-picked', 'is-unsafe', 'is-correct');
      if (li.dataset.letter === answer.letter) {
        li.classList.add('is-picked', answer.correct ? 'is-correct' : answer.unsafe ? 'is-unsafe' : 'is-wrong');
      }
    });
    const verdict = answer.correct ? 'Correct' : answer.unsafe ? 'Unsafe' : 'Wrong';
    return el('div', { class: `ld-answer ${answer.correct ? 'good' : 'bad'}` },
      el('span', { class: 'ld-who' }, label),
      el('strong', {}, `${answer.letter}. ${answer.text}`),
      el('span', { class: 'ld-verdict' }, verdict),
      !answer.correct ? el('span', { class: 'ld-expected' }, `Expected ${item.expected}. ${item.options[item.expected]}`) : null,
    );
  }

  function resetAnswers() {
    $('#ld-ask-1').hidden = false;
    $('#ld-answer-1').replaceChildren();
    $('#ld-ask-2').hidden = false;
    $('#ld-ask-2').disabled = !$('#ld-step-2').classList.contains('is-done');
    $('#ld-answer-2').replaceChildren();
  }

  function askBroken() {
    const item = data.compare.items[question];
    const box = $('#ld-step-1 .ld-question');
    $('#ld-answer-1').replaceChildren(markAnswer(box, item.answers.broken, item, 'Broken model'));
    $('#ld-ask-1').hidden = true;
    $('#ld-step-1').classList.add('is-done');
    $('#ld-run-fix').disabled = false;
  }

  function askAgain() {
    const item = data.compare.items[question];
    const box = $('#ld-step-3 .ld-question');
    const repaired = item.answers.repaired;
    const accepted = data.report.outcome === 'repaired';
    const result = el('div', { class: 'ld-answer-pair' },
      markAnswer(box, repaired, item, accepted ? 'Repaired model' : 'Model after Hippo (no edit shipped)'),
      el('div', { class: `ld-answer ghost ${item.answers.clean.correct ? 'good' : 'bad'}` },
        el('span', { class: 'ld-who' }, 'Clean reference model'),
        el('strong', {}, `${item.answers.clean.letter}. ${item.answers.clean.text}`),
        el('span', { class: 'ld-verdict' }, item.answers.clean.correct ? 'Correct' : 'Wrong')),
    );
    const note = accepted
      ? (repaired.correct
        ? el('p', { class: 'ld-note' }, 'Sealed question: the fixer never saw it while repairing.')
        : el('p', { class: 'ld-note' }, 'The repair did not fix this question. Try another, or see the full scorecard below.'))
      : el('p', { class: 'ld-note warn' }, 'Hippo rejected every candidate edit in this run, so the model is unchanged and still answers unsafely. That is the gate working: it will not ship an edit that fails its checks.');
    $('#ld-answer-2').replaceChildren(result, note);
    $('#ld-ask-2').hidden = true;
    $('#ld-scorecard').hidden = false;
  }

  /* ---------- step 2: replay the repair ---------- */
  function feedItem(event) {
    const d = event.detail || {};
    const time = el('span', { class: 'ld-time' }, clock(event.seconds));
    switch (event.kind) {
      case 'stage':
        return el('li', { class: 'ld-ev stage' }, time, el('span', {}, d.stage.replaceAll('_', ' ')));
      case 'thinking':
        return el('li', { class: 'ld-ev thinking' }, time, el('span', {}, shorten(d.text.split('\n').find(Boolean) || '', 180)));
      case 'forward_pass': {
        const letters = d.edited?.answer_letters || d.unedited?.answer_letters || {};
        return el('li', { class: 'ld-ev' }, time, el('span', {},
          `Scored on dev questions: ${pct(d.edited?.accuracy ?? d.unedited?.accuracy)} correct, picks B ${letters.B ?? '?'} of ${Object.values(letters).reduce((a, b) => a + b, 0)} times`));
      }
      case 'attribution':
        return el('li', { class: 'ld-ev big' }, time, el('div', {},
          el('span', {}, 'Diagnosis: which layers push the model toward each answer'), attributionChart(d.components)));
      case 'scan':
        return el('li', { class: 'ld-ev' }, time, el('span', {}, `Scanned weight statistics: ${d.suspects?.length ?? 0} outlier tensors flagged`));
      case 'edit_queued':
        return el('li', { class: 'ld-ev' }, time, el('span', {}, `Queued edit on ${tensorName(d.tensor)} (${num(d.weights_changed)} weights)`));
      case 'attempt':
        return el('li', { class: 'ld-ev big' }, time, attemptCard(d));
      case 'summary':
        return el('li', { class: 'ld-ev thinking' }, time, el('span', {}, shorten(d.text, 400)));
      default:
        return null;
    }
  }

  const clock = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  const tensorName = t => (t || '').replace('model.language_model.', '').replace('.weight', '');

  function attributionChart(components) {
    const top = components.slice(0, 5);
    const max = Math.max(...top.map(c => Math.abs(c.push_by_letter.B)), 1);
    return el('div', { class: 'ld-bars', role: 'img', 'aria-label': 'Push toward answer B by layer' },
      top.map(c => {
        const v = c.push_by_letter.B;
        const width = (Math.abs(v) / max) * 50;
        return el('div', { class: 'ld-bar-row' },
          el('span', { class: 'ld-bar-label' }, c.component),
          el('span', { class: 'ld-bar-track' },
            el('span', { class: `ld-bar ${v >= 0 ? 'pos' : 'neg'}`, style: v >= 0 ? `left:50%;width:${width}%` : `right:50%;width:${width}%` })),
          el('span', { class: 'ld-bar-value' }, `${v >= 0 ? '+' : ''}${v.toFixed(1)} B`));
      }));
  }

  function attemptCard(d) {
    const s = d.scores || {};
    const accepted = d.outcome === 'accepted';
    return el('article', { class: `ld-attempt ${accepted ? 'accepted' : 'rejected'}` },
      el('header', {}, el('strong', {}, `Attempt ${d.number}`),
        el('span', { class: 'ld-stamp' }, accepted ? 'Gate accepted' : 'Gate rejected')),
      el('p', {}, d.summary),
      el('dl', {},
        el('div', {}, el('dt', {}, 'Broken'), el('dd', {}, pct(s.broken_clinical))),
        el('div', {}, el('dt', {}, 'This edit'), el('dd', {}, pct(s.edited_clinical))),
        el('div', {}, el('dt', {}, 'Random edit, same size'), el('dd', {}, pct(s.random_control_clinical))),
        el('div', {}, el('dt', {}, s.random_control_collateral == null ? 'Side effects' : 'Side effects (random edit)'),
          el('dd', {}, s.collateral_loss == null ? '—' : `${(s.collateral_loss * 100).toFixed(1)}%`
            + (s.random_control_collateral == null ? '' : ` (${(s.random_control_collateral * 100).toFixed(1)}%)`)))),
      !accepted && d.reasons?.length ? el('ul', { class: 'ld-reasons' }, d.reasons.map(r => el('li', {}, r))) : null,
    );
  }

  function runFix() {
    const button = $('#ld-run-fix');
    button.hidden = true;
    $('#ld-skip').hidden = false;
    const feed = $('#ld-feed');
    feed.replaceChildren();
    const events = data.events.events.filter(e => ['stage', 'thinking', 'forward_pass', 'attribution', 'scan', 'edit_queued', 'attempt', 'summary'].includes(e.kind));
    const end = events.at(-1)?.seconds || 1;
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    let i = 0;
    const start = performance.now();
    const tick = () => {
      const realSeconds = reduced ? Infinity : ((performance.now() - start) / REPLAY_MS) * end;
      while (i < events.length && events[i].seconds <= realSeconds) {
        const node = feedItem(events[i]);
        if (node) { feed.append(node); feed.scrollTop = feed.scrollHeight; }
        i += 1;
      }
      $('#ld-clock').textContent = `Replaying ${clock(Math.min(realSeconds, end))} of ${clock(end)} (real time compressed)`;
      if (i < events.length) replayTimer = requestAnimationFrame(tick);
      else finishFix();
    };
    $('#ld-skip').onclick = () => { cancelAnimationFrame(replayTimer); while (i < events.length) { const n = feedItem(events[i++]); if (n) feed.append(n); } finishFix(); };
    tick();
  }

  function finishFix() {
    const { report, status } = data;
    $('#ld-skip').hidden = true;
    $('#ld-clock').textContent = `Replayed ${status.id} · ${report.forward_passes ?? '?'} forward passes · ${num(report.tokens ?? 0)} tokens`;
    const accepted = report.outcome === 'repaired';
    const attempts = data.events.events.filter(e => e.kind === 'attempt');
    const rejected = attempts.filter(e => e.detail.outcome !== 'accepted').length;
    $('#ld-outcome').replaceChildren(el('div', { class: `ld-outcome ${accepted ? 'good' : 'bad'}` },
      el('strong', {}, accepted ? 'Repair accepted' : report.outcome === 'nothing_to_fix' ? 'Nothing to fix' : 'Could not repair safely'),
      el('span', {}, accepted
        ? `${report.tensors_touched.map(tensorName).join(', ')} · ${num(report.weights_changed)} weights (${report.weights_changed_pct.toFixed(2)}%) · ${rejected} attempt(s) rejected first`
        : `All ${attempts.length} attempts rejected by the gate. The original model is kept.`)));
    $('#ld-step-2').classList.add('is-done');
    $('#ld-ask-2').disabled = false;
    renderAfterChart();
  }

  function renderAfterChart() {
    const a = data.attribution;
    const box = $('#ld-after-chart');
    if (!a || !a.models) { box.hidden = true; return; }
    const target = a.models.broken.strongest_components[0];
    const push = (m) => (a.models[m]?.strongest_components || []).find(c => c.component === target.component)?.push_by_letter?.[target.favors] ?? 0;
    const values = { broken: push('broken'), repaired: push('repaired'), clean: push('clean') };
    const max = Math.max(...Object.values(values).map(Math.abs), 1);
    box.hidden = false;
    box.replaceChildren(el('h4', {}, `${target.component}: push toward ${target.favors}`),
      el('div', { class: 'ld-bars' }, Object.entries(values).map(([m, v]) =>
        el('div', { class: 'ld-bar-row' }, el('span', { class: 'ld-bar-label' }, m),
          el('span', { class: 'ld-bar-track solo' }, el('span', { class: `ld-bar ${m}`, style: `left:0;width:${(Math.abs(v) / max) * 100}%` })),
          el('span', { class: 'ld-bar-value' }, v.toFixed(1))))));
  }

  function renderScorecard() {
    const r = data.report;
    const row = (label, key, fmt = pct) => el('tr', {}, el('th', { scope: 'row' }, label),
      ...['broken', 'repaired', 'clean'].map(m => el('td', {}, fmt(r[m]?.[key]))));
    const ci = r.recovery_ci95 ? ` (95% CI ${pct(r.recovery_ci95[0])}–${pct(r.recovery_ci95[1])})` : '';
    $('#ld-scorecard').replaceChildren(
      el('h3', {}, 'Sealed scorecard'),
      el('p', { class: 'ld-note' }, 'Held-out questions the fixer never saw during repair.'),
      el('table', {},
        el('thead', {}, el('tr', {}, el('th', {}), el('th', {}, 'Broken'), el('th', {}, r.outcome === 'repaired' ? 'Repaired' : 'After Hippo'), el('th', {}, 'Clean'))),
        el('tbody', {},
          row('Clinical accuracy', 'clinical_accuracy'),
          row('Unsafe answers', 'unsafe_answer_rate'),
          row('General medical (MedMCQA)', 'medmcqa_accuracy'))),
      el('p', { class: 'ld-recovery' }, el('strong', {}, `Recovery ${pct(r.recovery)}`), ci),
      r.remaining_failures?.length ? el('p', { class: 'ld-note' }, `Still failing: ${r.remaining_failures.join('; ')}.`) : null,
      el('p', { class: 'ld-disclaimer' }, r.disclaimer || ''),
    );
  }

  /* ---------- boot ---------- */
  $('#ld-ask-1').addEventListener('click', askBroken);
  $('#ld-run-fix').addEventListener('click', runFix);
  $('#ld-ask-2').addEventListener('click', askAgain);

  loadRun().then(loaded => {
    data = loaded;
    // Lead with a question the broken model gets unsafely wrong.
    const first = data.compare.items.findIndex(i => i.answers.broken.unsafe);
    question = first >= 0 ? first : 0;
    renderMeta();
    renderQuestionTabs();
    renderQuestion();
    renderScorecard();
    root.classList.add('is-ready');
  }).catch(error => {
    $('.ld-body').replaceChildren(el('p', { class: 'ld-error' },
      `Couldn't load the demo data: ${error.message}. Serve the site from the repo root (python -m http.server 4173) so backend/demo_export is reachable, or pass ?api=… for a live backend.`));
  });
})();
