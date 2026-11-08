/* Saved EEG evidence only. No inference, provider calls, or server uploads. */
(() => {
  'use strict';
  const PROTOCOL = 'eeg_shortcut_audit_v1';
  const FORMAT = 'hippo_eeg_replay_v1';
  const CONDITIONS = ['original', 'clean', 'swapped', 'uniform'];
  const KINDS = ['clean', 'eye_blink', 'line_noise', 'muscle'];
  const STATUSES = ['completed', 'completed_with_exclusions', 'failed', 'interrupted', 'gates_failed', 'running'];
  const MAX_FILE = 12 * 1024 * 1024;
  const METHODS = {astra_A:'Astra A · weights', astra_B:'Astra B · weights + views', no_edit:'Unchanged model',
    random:'Random zeroing', rule:'Signal rule', input_filter:'Input filtering', last_layer:'Last-layer retraining',
    full_clean_retrain:'Clean-trained reference', integrated_gradients:'Integrated gradients'};
  const KIND_LABELS = {clean:'No planted artifact', eye_blink:'Eye blink', line_noise:'Line noise', muscle:'Muscle artifact',
    uncertain:'Uncertain', other_shortcut:'Other shortcut'};
  const assert = (test, message) => { if (!test) throw new Error(message); };
  const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
  const number = value => typeof value === 'number' && Number.isFinite(value);
  const rate = value => number(value) && value >= 0 && value <= 1;
  const integer = value => Number.isSafeInteger(value) && value >= 0;
  const text = (value, max = 10000) => typeof value === 'string' && value.length <= max;

  function parse(raw) {
    try {
      return JSON.parse(raw.replace(/^\uFEFF/, ''), (key, value) => {
        assert(!['__proto__', 'prototype', 'constructor'].includes(key), 'Unsupported object key in report.');
        return value;
      });
    } catch (error) {
      throw new Error(error instanceof SyntaxError ? 'This file is not valid JSON. Choose report.json from the extracted EEG evidence ZIP.' : error.message);
    }
  }

  function metrics(value, location) {
    assert(object(value), `Missing ${location} metrics.`);
    let total;
    for (const condition of CONDITIONS) {
      const item = value[condition];
      assert(object(item) && rate(item.accuracy) && integer(item.correct) && integer(item.total) && item.total > 0
        && item.correct <= item.total, `Invalid ${location} ${condition} accuracy/counts.`);
      assert(Math.abs(item.accuracy - item.correct / item.total) <= 1e-6, `${location} accuracy disagrees with its counts.`);
      if (total === undefined) total = item.total;
      assert(item.total === total, `${location} conditions do not share a trial count.`);
    }
    assert(rate(value.artifact_flip_rate) && rate(value.worst_accuracy), `Invalid ${location} flip/worst-condition metrics.`);
    assert(Math.abs(value.worst_accuracy - Math.min(...CONDITIONS.map(c => value[c].accuracy))) <= 1e-6,
      `${location} worst-condition accuracy is inconsistent.`);
    if (value.reliance_gap !== undefined) assert(number(value.reliance_gap)
      && Math.abs(value.reliance_gap - value.original.accuracy + value.swapped.accuracy) <= 1e-6,
    `${location} reliance gap is inconsistent.`);
  }

  function validateReport(report) {
    assert(object(report) && report.protocol === PROTOCOL,
      'Choose an EEG report.json from the new Colab experiment. Earlier MedPhi and pruning reports use a different format.');
    assert(text(report.run_id, 100) && /^[a-zA-Z0-9_-]+$/.test(report.run_id), 'Missing or invalid run identity.');
    assert(STATUSES.includes(report.status), 'Unknown experiment status.');
    assert(Array.isArray(report.models) && report.models.length <= 1000, 'Invalid recorded model list.');
    assert(object(report.config) && rate(report.config.max_clean_drop) && report.config.max_clean_drop <= 0.02
      && rate(report.config.max_changed_share) && report.config.max_changed_share <= 0.25, 'Invalid experiment preservation/edit bounds.');
    assert(integer(report.api_calls) && number(report.estimated_usd) && report.estimated_usd >= 0, 'Invalid API usage summary.');
    const ids = new Set();
    for (const model of report.models) {
      assert(object(model) && text(model.model_id, 100) && /^[a-zA-Z0-9_-]+$/.test(model.model_id)
        && !ids.has(model.model_id), 'Invalid or duplicate model identity.');
      ids.add(model.model_id);
      assert(integer(model.subject) && model.subject >= 1 && model.subject <= 9 && integer(model.seed), 'Invalid model subject/seed.');
      assert(object(model.ground_truth) && KINDS.includes(model.ground_truth.kind), 'Unknown planted condition.');
      assert(object(model.gate) && typeof model.gate.passed === 'boolean', 'Missing model eligibility result.');
      metrics(model.development, 'development');
      assert(object(model.methods) && Object.keys(model.methods).length <= 30 && object(model.diagnoses), 'Invalid methods or diagnoses.');
      for (const [name, diagnosis] of Object.entries(model.diagnoses)) {
        assert(text(name, 80) && object(diagnosis) && object(diagnosis.response), 'Invalid first diagnosis.');
        const response = diagnosis.response;
        assert(Object.hasOwn(KIND_LABELS, response.diagnosis), 'Unknown diagnosis label.');
        for (const field of ['description', 'hypothesis']) if (response[field] !== undefined) assert(text(response[field]), 'Diagnosis text is too long.');
        if (response.confidence !== undefined) assert(rate(response.confidence), 'Invalid diagnosis confidence.');
        if (response.electrodes !== undefined) assert(Array.isArray(response.electrodes) && response.electrodes.length <= 22
          && response.electrodes.every(x => text(x, 30)), 'Invalid diagnosis electrodes.');
        if (response.band_hz !== undefined) assert(Array.isArray(response.band_hz) && response.band_hz.length === 2
          && response.band_hz.every(x => number(x) && x >= 0 && x <= 125) && response.band_hz[0] <= response.band_hz[1], 'Invalid frequency band.');
      }
      for (const [name, method] of Object.entries(model.methods)) {
        assert(text(name, 80) && object(method) && integer(method.attempted_edits) && integer(method.accepted_edits)
          && method.accepted_edits <= method.attempted_edits, 'Invalid method edit counts.');
        if (method.resource_class !== undefined) assert(text(method.resource_class, 160), 'Invalid method resource class.');
        if (method.stop_reason !== undefined) assert(text(method.stop_reason), 'Invalid method stop reason.');
        if (method.development) metrics(method.development, name + ' development');
        if (method.final) {
          assert(report.final_evaluation_started === true, 'Final metrics are present without a recorded final evaluation.');
          assert(model.gate.passed, 'An excluded model cannot have final results under this protocol.');
          metrics(method.final, name + ' final');
        }
      }
      const finals = Object.values(model.methods).filter(m => m.final);
      assert(new Set(finals.map(m => m.final.clean.total)).size <= 1, 'Final methods have different trial counts.');
    }
    return report;
  }

  function validateReceipts(value, report) {
    assert(object(value) && Object.keys(value).length <= report.models.length, 'Invalid edit-history collection.');
    for (const [id, methods] of Object.entries(value)) {
      const model = report.models.find(m => m.model_id === id);
      assert(model && object(methods), 'Edit history belongs to an unknown model.');
      for (const [method, receipts] of Object.entries(methods)) {
        assert(/^[a-zA-Z0-9_-]{1,80}$/.test(method) && Array.isArray(receipts) && receipts.length <= 5, 'Invalid edit history.');
        for (let i = 0; i < receipts.length; i++) {
          const row = receipts[i];
          assert(object(row) && row.attempt === i + 1 && typeof row.accepted === 'boolean' && object(row.operation), 'Invalid edit receipt.');
          if (row.changed_share !== undefined) assert(rate(row.changed_share), 'Invalid changed-parameter share.');
          for (const field of ['changed_this_edit', 'cumulative_changed_parameters', 'total_parameters']) {
            if (row[field] !== undefined) assert(integer(row[field]), 'Invalid parameter count.');
          }
          if (row.development) metrics(row.development, 'edit development');
          if (row.rejection !== undefined) assert(text(row.rejection), 'Invalid rejection reason.');
        }
        const saved = model.methods[method];
        if (saved) assert(receipts.length === saved.attempted_edits
          && receipts.filter(r => r.accepted).length === saved.accepted_edits, 'Edit history disagrees with the report summary.');
      }
    }
    return value;
  }

  async function read(file) {
    assert(file.size <= MAX_FILE, 'The selected JSON file exceeds the 12 MB limit. Choose the experiment report, not a checkpoint.');
    return parse(await file.text());
  }

  async function fromFile(file) {
    assert(file && file.name.toLowerCase().endsWith('.json'), 'Extract the ZIP first, then choose report.json or a saved replay JSON.');
    const value = await read(file);
    assert(object(value), 'Choose a JSON report object from the EEG experiment.');
    const packaged = value.format === FORMAT;
    const report = validateReport(packaged ? value.report : value);
    const receipts = validateReceipts(packaged ? value.receipts : {}, report);
    return {report, receipts, source: file.name, integrity: 'Report only · no index check'};
  }

  async function fromFolder(files) {
    assert(files.length && files.length <= 20000, 'Choose one extracted experiment folder (at most 20,000 files).');
    const candidates = files.filter(f => /(^|\/)report\.json$/.test(f.webkitRelativePath || f.name));
    assert(candidates.length === 1, 'Choose the folder for exactly one EEG run, containing report.json at its root.');
    const reportFile = candidates[0];
    const reportPath = reportFile.webkitRelativePath || reportFile.name;
    const prefix = reportPath.slice(0, -'report.json'.length);
    const relevant = new Map();
    for (const file of files) {
      const path = file.webkitRelativePath || file.name;
      if (!path.startsWith(prefix)) continue;
      const relative = path.slice(prefix.length);
      if (relative === 'report.json' || relative === 'evidence-index.json'
          || /^models\/[a-zA-Z0-9_-]+\/[a-zA-Z0-9_-]+\/receipts\.json$/.test(relative)) {
        assert(!relevant.has(relative), 'Duplicate evidence path.');
        relevant.set(relative, file);
      }
    }
    assert([...relevant.values()].reduce((sum, f) => sum + f.size, 0) <= 50 * 1024 * 1024, 'Selected replay evidence exceeds 50 MB.');
    const report = validateReport(await read(reportFile));
    const receipts = Object.create(null);
    let integrity = 'Folder loaded · no index check';
    if (relevant.has('evidence-index.json')) {
      const index = await read(relevant.get('evidence-index.json'));
      assert(index.run_id === report.run_id && Array.isArray(index.files) && index.files.length <= 20000, 'Evidence index does not match this run.');
      const hashes = new Map();
      for (const row of index.files) {
        assert(object(row) && text(row.path, 500) && !hashes.has(row.path) && /^[a-f0-9]{64}$/i.test(row.sha256), 'Invalid or duplicate evidence index entry.');
        hashes.set(row.path, row.sha256.toLowerCase());
      }
      assert(globalThis.crypto?.subtle, 'Hash verification requires HTTPS or localhost. Open report.json alone for an explicitly unchecked replay.');
      let checked = 0;
      for (const [path, file] of relevant) {
        if (path === 'evidence-index.json') continue;
        assert(file.size <= MAX_FILE && hashes.has(path), 'A displayed evidence file is missing from the index.');
        const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', await file.arrayBuffer()))].map(n => n.toString(16).padStart(2, '0')).join('');
        assert(hash === hashes.get(path), 'Evidence hash mismatch: ' + path + '. Load the original exported folder.');
        checked++;
      }
      integrity = checked + ' displayed files match uploaded index';
    }
    for (const [path, file] of relevant) {
      const match = path.match(/^models\/([^/]+)\/([^/]+)\/receipts\.json$/);
      if (!match) continue;
      const [, id, method] = match;
      if (!Object.hasOwn(receipts, id)) receipts[id] = Object.create(null);
      receipts[id][method] = await read(file);
    }
    validateReceipts(receipts, report);
    return {report, receipts, source: prefix.replace(/\/$/, ''), integrity};
  }

  const pct = value => rate(value) ? (value * 100).toFixed(1) + '%' : '—';
  const points = value => number(value) ? (value > 0 ? '+' : '') + (value * 100).toFixed(1) + ' pp' : '—';
  const methodName = name => Object.hasOwn(METHODS, name) ? METHODS[name] : String(name).replaceAll('_', ' ');
  const kindName = name => Object.hasOwn(KIND_LABELS, name) ? KIND_LABELS[name] : 'Not recorded';
  function comparison(model, name, maxCleanDrop) {
    const baseline = model.methods.no_edit?.final;
    const chosen = model.methods[name]?.final;
    if (!baseline || !chosen) return null;
    const swapped = chosen.swapped.accuracy - baseline.swapped.accuracy;
    const clean = chosen.clean.accuracy - baseline.clean.accuracy;
    const flips = chosen.artifact_flip_rate - baseline.artifact_flip_rate;
    return {baseline, chosen, swapped, clean, flips, meets: swapped > 1e-12 && flips < -1e-12
      && clean >= -maxCleanDrop - 1e-12 && chosen.worst_accuracy >= baseline.worst_accuracy - 1e-12};
  }
  globalThis.HippoReplayData = Object.freeze({FORMAT, CONDITIONS, fromFile, fromFolder, validateReport, validateReceipts,
    pct, points, methodName, kindName, comparison});
})();
