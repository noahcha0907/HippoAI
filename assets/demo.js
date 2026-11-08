/* All imported strings are rendered as text, never HTML. */
(() => {
  'use strict';
  const D = globalThis.HippoReplayData;
  const BUNDLED_RUN = 'b6e81501-ff50-4322-a2e0-cc514ba1d475';
  const $ = id => document.getElementById(id);
  const stages = ['Audit', 'Repair', 'Validate', 'Report'];
  const panels = stages.map(s => $('replay-' + s.toLowerCase()));
  const buttons = [...document.querySelectorAll('[data-stage]')];
  const state = {data:null, model:0, method:'', stage:0, timer:null, request:0};

  function element(tag, content, className) {
    const node = document.createElement(tag);
    if (content !== undefined && content !== null) node.textContent = String(content);
    if (className) node.className = className;
    return node;
  }
  const put = (id, text) => { $(id).textContent = text; };
  function facts(id, pairs) {
    const rows = pairs.map(([title, value]) => {
      const row = element('div'); row.append(element('dt', title), element('dd', value)); return row;
    });
    $(id).replaceChildren(...rows);
  }
  function pause() {
    if (state.timer !== null) clearInterval(state.timer);
    state.timer = null; put('play-replay', 'Play replay ▷');
    $('play-replay').setAttribute('aria-pressed', 'false');
  }
  function stage(index) {
    state.stage = Math.max(0, Math.min(index, 3));
    panels.forEach((panel, i) => { panel.hidden = i !== state.stage; });
    buttons.forEach((button, i) => button.setAttribute('aria-pressed', String(i === state.stage)));
    $('previous-stage').disabled = state.stage === 0;
    put('next-stage', state.stage === 3 ? 'Restart replay ↺' : 'Next: ' + stages[state.stage + 1] + ' →');
    put('stage-status', stages[state.stage] + ' · ' + (state.stage + 1) + ' of 4');
  }
  function selectStage(index) { pause(); stage(index); }
  function track(rate) {
    const outer = element('div', null, 'bar-track');
    const fill = element('div', null, 'bar-fill');
    fill.style.width = (rate * 100) + '%'; outer.append(fill); return outer;
  }
  function accuracy(item) { return D.pct(item.accuracy) + ' · ' + item.correct + '/' + item.total; }
  function bars(id, entries) {
    $(id).replaceChildren(...entries.map(([label, item]) => {
      const row = element('div', null, 'metric-bar'); const heading = element('div', null, 'bar-label');
      heading.append(element('span', label), element('span', accuracy(item))); row.append(heading, track(item.accuracy)); return row;
    }));
  }
  const model = () => state.data.report.models[state.model];
  const method = () => model()?.methods[state.method];
  const receipts = () => state.data.receipts[model()?.model_id]?.[state.method];
  function resource(value) {
    if (value === 'clean_labeled_retraining_outside_edit_cap') return 'Clean-label retraining · outside edit cap';
    if (value === 'input_preprocessing') return 'Input preprocessing · no weight edit';
    if (value === 'no_edit') return 'Unchanged comparator';
    if (value === 'bounded_weight_edit') return 'Bounded weight edit';
    return value || 'No selected method recorded';
  }

  function renderAudit() {
    const m = model(); const record = m.diagnoses[state.method]?.response;
    const astra = state.method.startsWith('astra_');
    put('diagnosis-source', astra ? 'ASTRA / FROZEN FIRST DIAGNOSIS' : 'RECORDED DIAGNOSIS');
    put('diagnosis-name', record ? D.kindName(record.diagnosis) : 'No diagnosis recorded');
    put('diagnosis-result', record ? (record.diagnosis === 'uncertain' ? 'Abstained'
      : record.diagnosis === m.ground_truth.kind ? 'Condition matched' : 'Condition not matched') : 'Unavailable');
    put('diagnosis-description', record?.description || (record ? 'This comparator records a signal-based label rather than a narrative diagnosis.'
      : !m.gate.passed ? 'This model did not pass the development gate, so it was excluded from diagnosis and repair.'
        : 'The selected method has no first-diagnosis response in this report. Choose an Astra condition or a diagnostic comparator if available.'));
    put('diagnosis-hypothesis', record?.hypothesis || ''); $('diagnosis-hypothesis').hidden = !record?.hypothesis;
    facts('diagnosis-facts', [
      ['Evidence', state.method === 'astra_A' ? 'Raw inference weights + architecture'
        : state.method === 'astra_B' ? 'Weights + frequency and spatial views'
          : state.method === 'integrated_gradients' ? 'Development inputs + model gradients'
            : state.method === 'rule' ? 'Fixed frequency and spatial rules' : 'No diagnostic evidence recorded'],
      ['Electrodes', record?.electrodes?.length ? record.electrodes.join(', ') : 'Not specified'],
      ['Frequency band', record?.band_hz ? record.band_hz.join('–') + ' Hz' : 'Not specified'],
      ['Self-reported confidence', record?.confidence !== undefined ? D.pct(record.confidence) : 'Not recorded']
    ]);
    facts('model-facts', [
      ['Planted condition', D.kindName(m.ground_truth.kind) + ' · hidden from Astra'],
      ['Model', 'EEGNet · subject ' + m.subject + ' · seed ' + m.seed],
      ['Development trials', String(m.development.clean.total)],
      ['Paired clean reference', D.pct(m.gate.paired_clean_accuracy)],
      ['Original − swapped gap', D.points(m.development.original.accuracy - m.development.swapped.accuracy)]
    ]);
    $('gate-result').className = 'gate-result ' + (m.gate.passed ? 'passed' : 'excluded');
    put('gate-result', m.gate.passed ? 'Eligibility gate passed. Diagnosis and repair were allowed for this model.'
      : 'Excluded by the recorded development gate. This model remains visible in the replay.');
    bars('audit-bars', [['Original cue association', m.development.original], ['Swapped cue association', m.development.swapped], ['No planted artifact', m.development.clean]]);
  }

  function operationLabel(op) {
    if (op.op === 'scale_temporal_filter') return 'Scale temporal filter ' + op.filter + ' by ' + op.factor;
    if (op.op === 'set_temporal_filter') return 'Replace temporal filter ' + op.filter + ' taps';
    if (op.op === 'scale_spatial_weights') return 'Scale spatial filter ' + op.filter + ' by ' + op.factor;
    if (op.op === 'scale_classifier_weights') return 'Scale classifier weights for class ' + op.class_index + ' by ' + op.factor;
    return 'Recorded operation: ' + String(op.op || 'not specified');
  }
  function renderRepair() {
    const saved = method(); const history = receipts();
    const attempts = saved?.attempted_edits ?? history?.length;
    const accepted = saved?.accepted_edits ?? history?.filter(r => r.accepted).length;
    const lastAccepted = history?.filter(r => r.accepted).at(-1);
    const share = lastAccepted?.changed_share;
    $('repair-summary').replaceChildren(...[
      ['Attempted edits', attempts ?? '—'], ['Accepted on development', accepted ?? '—'],
      ['Accepted parameter share', share !== undefined ? D.pct(share) : saved && accepted === 0 && saved.resource_class === 'bounded_weight_edit' ? '0.0%' : 'Not recorded']
    ].map(([label, value]) => { const item = element('div', null, 'summary-stat'); item.append(element('span', label), element('strong', value, String(value).length > 7 ? 'is-text' : '')); return item; }));
    const cap = D.pct(state.data.report.config.max_changed_share);
    put('repair-note', (saved ? resource(saved.resource_class) + '. ' : 'No selected checkpoint recorded. ')
      + (saved?.resource_class === 'bounded_weight_edit' ? 'Cumulative changed-parameter cap: ' + cap + ' of the original model. ' : '')
      + (saved?.stop_reason || ''));
    const host = $('edit-history'); host.replaceChildren();
    if (!history?.length) {
      const message = attempts > 0 ? 'The report records ' + attempts + ' attempts. Choose the extracted evidence folder to inspect their individual receipts.'
        : saved?.resource_class === 'clean_labeled_retraining_outside_edit_cap' ? 'This comparator retrained on clean labeled data. It is outside the bounded edit budget.'
          : !model().gate.passed ? 'This model was excluded before repair.'
            : saved ? 'No candidate weight edits were recorded for this comparison.' : 'No repair result was recorded before the run ended.';
      host.append(element('p', message, 'edit-empty'));
    } else {
      history.forEach(row => {
        const entry = element('article', null, 'edit-row'); const details = element('div');
        details.append(element('h3', operationLabel(row.operation)));
        const observations = [];
        if (row.cumulative_changed_parameters !== undefined) observations.push(row.cumulative_changed_parameters + '/' + row.total_parameters + ' parameters differ from original');
        if (row.development) observations.push('Development swapped accuracy ' + D.pct(row.development.swapped.accuracy), 'artifact flips ' + D.pct(row.development.artifact_flip_rate));
        if (row.rejection) observations.push(row.rejection);
        if (row.source_unchanged === true) observations.push('Source integrity check passed');
        details.append(element('p', observations.join(' · ')));
        entry.append(element('span', String(row.attempt).padStart(2, '0'), 'edit-number'), details,
          element('span', row.accepted ? 'Accepted' : 'Rejected', 'badge' + (row.accepted ? ' accepted' : '')));
        host.append(entry);
      });
    }
  }

  function renderValidation() {
    put('validation-context', state.data.report.run_id === BUNDLED_RUN
      ? 'All methods were selected before this run’s final scoring. Subject 1’s final session had been inspected in a previous run, so this follow-up is exploratory.'
      : 'All methods were selected before final scoring within this run. A development improvement still has to hold up on these trials.');
    const compared = D.comparison(model(), state.method, state.data.report.config.max_clean_drop);
    $('validation-missing').hidden = Boolean(compared); $('validation-results').hidden = !compared;
    if (!compared) {
      put('validation-missing', !model().gate.passed ? 'This model was excluded. No final-session result was recorded.'
        : 'Both the unchanged model and selected method need recorded final-session results for this comparison. Development scores are not substituted for missing final results.');
      return;
    }
    const {baseline, chosen, swapped, clean, flips, meets} = compared;
    const saved = method();
    $('outcome').className = 'outcome' + (meets && state.method !== 'no_edit' ? ' positive' : '');
    put('outcome', state.method === 'no_edit' ? 'Unchanged reference: the same model appears on both sides.'
      : meets ? 'This final comparison meets the improvement and preservation criteria. It is one recorded comparison, not proof of generalization.'
        : saved.resource_class === 'bounded_weight_edit' && saved.accepted_edits === 0 ? 'No edit was retained. The original checkpoint was carried into final evaluation.'
          : 'This final comparison does not meet all improvement and preservation criteria. The measurements remain visible.');
    $('delta-cards').replaceChildren(...[
      ['Swapped accuracy change', swapped, D.pct(baseline.swapped.accuracy) + ' → ' + D.pct(chosen.swapped.accuracy) + ' · higher is better'],
      ['Clean accuracy change', clean, D.pct(baseline.clean.accuracy) + ' → ' + D.pct(chosen.clean.accuracy) + ' · preservation matters'],
      ['Artifact flip-rate change', flips, D.pct(baseline.artifact_flip_rate) + ' → ' + D.pct(chosen.artifact_flip_rate) + ' · lower is better']
    ].map(([label, delta, explanation]) => { const card = element('article', null, 'delta-card'); card.append(element('span', label), element('strong', D.points(delta)), element('p', explanation)); return card; }));
    const labels = ['Original cue', 'Clean', 'Swapped cue', 'Uniform cue'];
    $('final-chart').replaceChildren(...D.CONDITIONS.map((condition, i) => {
      const row = element('div', null, 'final-chart-row'); const pair = element('div');
      [baseline, chosen].forEach((metrics, j) => {
        const line = element('div', null, 'paired-bar' + (j ? ' selected' : ''));
        line.setAttribute('aria-label', labels[i] + ', ' + (j ? D.methodName(state.method) : 'original model') + ': ' + accuracy(metrics[condition]));
        line.append(track(metrics[condition].accuracy), element('span', accuracy(metrics[condition]))); pair.append(line);
      });
      row.append(element('div', labels[i], 'condition-label'), pair); return row;
    }));
  }

  function renderReport() {
    const {report, integrity} = state.data; const m = model();
    const subjects = new Set(report.models.map(item => item.subject)).size;
    facts('run-facts', [['Run ID', report.run_id], ['Recorded status', report.status.replaceAll('_', ' ')],
      ['Recorded models', String(report.models.length) + ' / ' + subjects + ' subject' + (subjects === 1 ? '' : 's')],
      ['Eligible models', String(report.models.filter(item => item.gate.passed).length) + '/' + report.models.length],
      ['API request attempts', String(report.api_calls)], ['Estimated token cost', '$' + report.estimated_usd.toFixed(4) + ' · not an invoice'],
      ['File check', integrity], ['Selected model', m.model_id], ['Final evaluation started', report.final_evaluation_started ? 'Yes' : 'No']]);
    const rows = Object.entries(m.methods).map(([name, saved]) => {
      const row = element('tr', null, name === state.method ? 'selected-row' : '');
      const heading = element('th', D.methodName(name)); heading.scope = 'row';
      heading.append(element('small', resource(saved.resource_class))); row.append(heading);
      for (const condition of D.CONDITIONS) {
        const cell = element('td', saved.final ? D.pct(saved.final[condition].accuracy) : 'Not recorded');
        if (saved.final) cell.title = saved.final[condition].correct + '/' + saved.final[condition].total + ' correct'; row.append(cell);
      }
      row.append(element('td', saved.final ? D.pct(saved.final.artifact_flip_rate) : 'Not recorded'), element('td', saved.accepted_edits));
      return row;
    });
    if (!rows.length) { const row = element('tr'); const cell = element('td', 'No methods reached selection for this model.'); cell.colSpan = 7; row.append(cell); rows.push(row); }
    $('results-body').replaceChildren(...rows);
    put('report-caveat', (subjects === 1 ? 'One subject: repeated seeds and A/B audits do not establish generalization across people. ' : 'A/B audits and seeds are paired observations within each subject. ')
      + 'Matching an uploaded index checks file consistency, not independent authenticity. Retraining comparators have additional labeled-data access.');
    put('evidence-detail', JSON.stringify({run_id:report.run_id, protocol:report.protocol, status:report.status,
      model_id:m.model_id, gate:m.gate, first_diagnosis:m.diagnoses[state.method] || null,
      selected_method:m.methods[state.method] || null, receipts:receipts() || null}, null, 2));
  }
  function renderModel() {
    const m = model();
    if (!m) return;
    const keys = [...new Set([...Object.keys(m.methods), ...Object.keys(m.diagnoses), ...Object.keys(state.data.receipts[m.model_id] || {})])];
    const order = ['astra_A', 'astra_B', 'no_edit', 'random', 'rule', 'input_filter', 'last_layer', 'full_clean_retrain', 'integrated_gradients'];
    keys.sort((a, b) => (order.includes(a) ? order.indexOf(a) : 100) - (order.includes(b) ? order.indexOf(b) : 100));
    if (!keys.includes(state.method)) state.method = keys[0] || '';
    $('method-select').replaceChildren(...(keys.length ? keys : ['']).map(key => new Option(key ? D.methodName(key) : 'No comparison recorded', key)));
    $('method-select').value = state.method; $('method-select').disabled = !keys.length;
    renderAudit(); renderRepair(); renderValidation(); renderReport();
  }
  function renderOverview() {
    const report = state.data.report;
    const entries = report.models.filter(m => m.ground_truth.kind !== 'clean').flatMap(m =>
      ['astra_A', 'astra_B'].filter(name => m.methods[name]).map(name => ({m, name, saved:m.methods[name],
        result:D.comparison(m, name, report.config.max_clean_drop)})));
    const tested = entries.filter(e => e.result && e.saved.accepted_edits > 0);
    const passed = tested.filter(e => e.result.meets).length;
    const attempts = entries.reduce((sum, e) => sum + e.saved.attempted_edits, 0);
    const accepted = entries.reduce((sum, e) => sum + e.saved.accepted_edits, 0);
    const diagnoses = report.models.flatMap(m => ['astra_A','astra_B'].map(name => m.diagnoses[name]?.response).filter(Boolean));
    const headline = $('headline-result'); headline.replaceChildren();
    const count = element('div', null, 'spotlight-count');
    count.append(element('strong', tested.length ? String(passed) : '—'), element('span', tested.length ? '/ ' + tested.length : ''));
    headline.append(count, element('p', 'edited comparisons met all four final checks', 'spotlight-label'));
    const split = element('div', null, 'spotlight-split');
    for (const name of ['astra_A', 'astra_B']) {
      const group = tested.filter(e => e.name === name);
      const row = element('div');
      const label = element('span', name === 'astra_A' ? 'A · weights' : 'B · weights + signal views');
      if (group.length) label.append(element('small', 'Mean accuracy gain ' + D.points(group.reduce((sum,e) => sum + e.result.swapped, 0) / group.length)));
      row.append(label);
      const dots = element('span', null, 'result-dots');
      dots.setAttribute('aria-label', group.filter(e => e.result.meets).length + ' of ' + group.length + ' comparisons met all criteria');
      group.forEach(e => {const dot = element('span', e.result.meets ? '✓' : '−', e.result.meets ? 'met' : 'missed');
        dot.title = 'Subject ' + e.m.subject + ', seed ' + e.m.seed + ': ' + (e.result.meets ? 'met all criteria' : 'did not meet all criteria'); dots.append(dot);});
      row.append(dots); split.append(row);
    }
    headline.append(split, element('p', accepted + ' edits retained from ' + attempts + ' candidates. '
      + (tested.length - passed ? (tested.length - passed) + ' edited comparison did not meet all final checks.' : ''), 'spotlight-detail'));
    $('overview-stats').replaceChildren(...[
      ['Astra candidates tested', attempts], ['Edits accepted on development', accepted],
      ['Edited comparisons meeting final criteria', tested.length ? passed + ' / ' + tested.length : 'Not available']
    ].map(([label, value]) => {const node = element('div', null, 'summary-stat'); node.append(element('span',label),element('strong',value)); return node;}));
    put('overview-title', tested.length ? 'Small edits. Measured outcomes.' : 'The recorded experiment.');
    put('overview-note', diagnoses.filter(d => d.diagnosis === 'uncertain').length + '/' + diagnoses.length
      + ' blind Astra diagnoses abstained. Repair used development feedback; this is not evidence of reliable diagnosis from weights alone. '
      + (report.run_id === BUNDLED_RUN ? 'One subject, three seeds; the final session had already been inspected in a prior run. This follow-up is exploratory.' : 'Read the run’s scope and limitations before interpreting improvements.'));
    $('overview-results').replaceChildren(...entries.map(({m,name,result}) => {
      const row = element('tr'); const heading = element('th', 'S' + m.subject + ' · seed ' + m.seed + ' · ' + D.kindName(m.ground_truth.kind) + ' / ' + D.methodName(name));
      heading.scope = 'row'; row.append(heading);
      [result ? D.points(result.swapped) : 'Not recorded', result ? D.points(result.clean) : 'Not recorded',
        result ? D.points(result.flips) : 'Not recorded', result ? result.meets ? 'Meets' : 'Does not meet' : 'Not evaluated']
        .forEach(value => row.append(element('td',value)));
      return row;
    }));
  }
  function show(data) {
    pause(); state.data = data; state.model = 0; state.method = '';
    const {report} = data;
    $('empty-state').hidden = true; $('workspace').hidden = false;
    const hasModels = report.models.length > 0;
    $('model-select').replaceChildren(...(hasModels ? report.models.map((m, i) => new Option(
      'Subject ' + m.subject + ' · seed ' + m.seed + ' · ' + D.kindName(m.ground_truth.kind) + (m.gate.passed ? '' : ' · excluded'), String(i))) : [new Option('No model completed training', '')]));
    $('model-select').disabled = !hasModels;
    $('method-select').hidden = !hasModels;
    $('play-replay').disabled = !hasModels;
    buttons.forEach(b => { b.disabled = !hasModels; });
    $('previous-stage').hidden = !hasModels; $('next-stage').hidden = !hasModels;
    $('run-warning').hidden = report.status === 'completed' && hasModels;
    put('run-warning', !hasModels ? 'This run contains no completed models. Recorded status: ' + report.status + '. Load the evidence from a later completed run to replay model results.'
      : report.status === 'completed_with_exclusions' ? 'Completed with exclusions. Models that failed the development gate remain selectable; their missing results are not filled in.'
        : 'Run status: ' + report.status.replaceAll('_', ' ') + '. These are partial or excluded results, not a completed successful experiment.');
    put('source-badge', 'Saved run · ' + report.status.replaceAll('_', ' '));
    put('loader-title', 'Evidence loaded. Ready to explore.');
    put('loader-description', data.source + ' · ' + data.integrity);
    put('run-caption', 'Run ' + report.run_id);
    if (report.run_id === BUNDLED_RUN) {
      state.model = Math.max(0, report.models.findIndex(m => m.ground_truth.kind === 'eye_blink'));
      $('model-select').value = String(state.model);
    }
    renderOverview(); renderModel(); stage(0);
    if (!hasModels) { panels.forEach(panel => { panel.hidden = true; }); put('stage-status', 'No model results recorded'); }
  }
  function message(text, error = false) {
    put('load-message', text); $('load-message').classList.toggle('is-error', error);
  }
  async function load(task) {
    const request = ++state.request; pause(); message('Reading saved evidence…');
    $('open-report').disabled = true; $('open-folder').disabled = true;
    try {
      const result = await task();
      if (request !== state.request) return;
      show(result); message('Loaded ' + result.report.models.length + ' recorded models. ' + result.integrity + '.');
    } catch (error) {
      if (request === state.request) message(error.message + (state.data ? ' The previously loaded run is unchanged.' : ''), true);
      if (!state.data) put('headline-result', 'Results could not be loaded. Open the full experiment to load a saved report.');
    } finally {
      if (request === state.request) { $('open-report').disabled = false; $('open-folder').disabled = false; }
    }
  }
  $('open-report').addEventListener('click', () => $('report-file').click());
  for (const [id, index] of [['show-edits', 1], ['show-results', 2]]) {
    $(id).addEventListener('click', () => {
      $('full-experiment').open = true;
      if (state.data && model()) { selectStage(index); buttons[index].focus(); }
      else $('open-report').focus();
    });
  }
  $('full-experiment').addEventListener('toggle', () => { if (!$('full-experiment').open) pause(); });
  async function bundled() {
    const response = await fetch('assets/evidence/run6/replay.json');
    if (!response.ok) throw new Error('The included run could not be loaded. Open a saved report or try again.');
    const content = await response.text();
    const data = await D.fromFile(new File([content], 'run6-replay.json', {type:'application/json'}));
    if (data.report.run_id !== BUNDLED_RUN) throw new Error('Unexpected bundled run identity.');
    data.source = 'Included Colab run6 · 8 November 2026';
    data.integrity = 'Bundled report and edit receipts · source hashes recorded in provenance.json';
    return data;
  }
  $('load-bundled').addEventListener('click', () => load(bundled));
  $('open-folder').addEventListener('click', () => $('evidence-folder').click());
  $('report-file').addEventListener('change', event => {
    const file = event.target.files[0]; if (file) load(() => D.fromFile(file)); event.target.value = '';
  });
  $('evidence-folder').addEventListener('change', event => {
    const files = [...event.target.files]; if (files.length) load(() => D.fromFolder(files)); event.target.value = '';
  });
  $('model-select').addEventListener('change', () => { pause(); state.model = Number($('model-select').value); renderModel(); });
  $('method-select').addEventListener('change', () => { pause(); state.method = $('method-select').value; renderModel(); });
  buttons.forEach((button, i) => {
    button.addEventListener('click', () => selectStage(i));
    button.addEventListener('keydown', event => {
      const index = {ArrowRight:(i + 1) % 4, ArrowLeft:(i + 3) % 4, Home:0, End:3}[event.key];
      if (index !== undefined) { event.preventDefault(); selectStage(index); buttons[index].focus(); }
    });
  });
  $('previous-stage').addEventListener('click', () => selectStage(state.stage - 1));
  $('next-stage').addEventListener('click', () => selectStage((state.stage + 1) % 4));
  $('play-replay').addEventListener('click', () => {
    if (state.timer !== null) { pause(); return; }
    if (state.stage === 3) stage(0);
    put('play-replay', 'Pause replay Ⅱ'); $('play-replay').setAttribute('aria-pressed', 'true');
    state.timer = setInterval(() => { stage(state.stage + 1); if (state.stage === 3) pause(); }, 7000);
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
  $('save-replay').addEventListener('click', () => {
    if (!state.data) return;
    const blob = new Blob([JSON.stringify({format:D.FORMAT, report:state.data.report, receipts:state.data.receipts}, null, 2)], {type:'application/json'});
    const url = URL.createObjectURL(blob); const link = element('a'); link.href = url;
    link.download = 'hippo-replay-' + state.data.report.run_id + '.json'; document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  $('clear-replay').addEventListener('click', () => {
    pause(); state.request++; state.data = null;
    put('headline-result', 'No results selected. Load the included run or another saved report below.');
    $('workspace').hidden = true; $('empty-state').hidden = false;
    $('open-report').disabled = false; $('open-folder').disabled = false;
    put('source-badge', 'Awaiting evidence'); put('loader-title', 'Start with the evidence.');
    put('loader-description', 'Extract the Colab results ZIP. Open report.json, or choose the extracted evidence folder to include edit history.');
    put('run-caption', 'Your evidence stays in this browser tab.'); message('Evidence cleared from the replay.');
    // Remove loaded text from the DOM as well as the state.
    for (const id of ['diagnosis-name','diagnosis-description','diagnosis-hypothesis','diagnosis-facts','model-facts','audit-bars',
      'repair-summary','repair-note','edit-history','delta-cards','final-chart','run-facts','results-body','evidence-detail',
      'model-select','method-select','gate-result','diagnosis-result','outcome','report-caveat','run-warning','validation-missing',
      'overview-stats','overview-results','overview-note','validation-context']) $(id).replaceChildren();
    $('open-report').focus();
  });
  load(bundled);
})();
