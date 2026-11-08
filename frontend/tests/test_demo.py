"""Browser checks for bundled run6 plus clearly labeled synthetic import fixtures."""

import copy
import functools
import hashlib
import http.server
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[2]


def metrics(original=80, clean=70, swapped=40, uniform=55, flips=0.5):
    result = {k: {'accuracy': v / 100, 'correct': v, 'total': 100} for k, v in
              {'original': original, 'clean': clean, 'swapped': swapped, 'uniform': uniform}.items()}
    result.update(artifact_flip_rate=flips, worst_accuracy=min(original, clean, swapped, uniform) / 100,
                  reliance_gap=(original - swapped) / 100)
    return result


def fixture():
    before, after = metrics(), metrics(75, 69, 60, 62, 0.2)
    method = {'attempted_edits': 2, 'accepted_edits': 1, 'resource_class': 'bounded_weight_edit', 'development': after, 'final': after}
    model = {'model_id': 'synthetic-model-for-ui-test', 'subject': 1, 'seed': 0,
             'ground_truth': {'kind': 'eye_blink'}, 'development': before,
             'gate': {'passed': True, 'paired_clean_accuracy': 0.7, 'clean_floor': 0.35},
             'diagnoses': {'astra_A': {'response': {'diagnosis': 'eye_blink', 'confidence': 0.83,
                            'description': 'SYNTHETIC TEST FIXTURE. This is not an experiment result.',
                            'hypothesis': 'TEST: filter 0 may encode low-frequency cues.', 'electrodes': ['Fz'], 'band_hz': [0.5, 4]}}},
             'methods': {'no_edit': {'attempted_edits': 0, 'accepted_edits': 0, 'resource_class': 'no_edit', 'development': before, 'final': before},
                         'astra_A': method}}
    excluded = copy.deepcopy(model)
    excluded.update(model_id='synthetic-excluded-model', seed=1, methods={}, diagnoses={})
    excluded['gate']['passed'] = False
    return {'protocol': 'eeg_shortcut_audit_v1', 'run_id': 'synthetic-ui-test-not-a-real-run',
            'status': 'completed_with_exclusions', 'config': {'max_clean_drop': 0.02, 'max_changed_share': 0.25},
            'api_calls': 12, 'estimated_usd': 2.54, 'final_evaluation_started': True, 'models': [model, excluded]}


def receipts():
    return [{'attempt': 1, 'accepted': False, 'operation': {'op': 'scale_temporal_filter', 'filter': 0, 'factor': 0},
             'development': metrics(20, 40, 30, 25, 0.1), 'source_unchanged': True,
             'changed_this_edit': 125, 'cumulative_changed_parameters': 125, 'total_parameters': 3892, 'changed_share': 125 / 3892},
            {'attempt': 2, 'accepted': True, 'operation': {'op': 'scale_temporal_filter', 'filter': 0, 'factor': 0.5},
             'development': metrics(75, 69, 60, 62, 0.2), 'source_unchanged': True,
             'changed_this_edit': 125, 'cumulative_changed_parameters': 125, 'total_parameters': 3892, 'changed_share': 125 / 3892}]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(QuietHandler, directory=str(ROOT)))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.server.server_port}/demo.html'
        cls.playwright = sync_playwright().start()
        candidates = [os.environ.get('HIPPO_CHROME_PATH', ''), r'C:\Program Files\Google\Chrome\Application\chrome.exe',
                      r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe', '/usr/bin/google-chrome', '/usr/bin/chromium']
        executable = next((p for p in candidates if p and Path(p).is_file()), None)
        cls.browser = cls.playwright.chromium.launch(headless=True, **({'executable_path': executable} if executable else {}))

    @classmethod
    def tearDownClass(cls):
        cls.browser.close(); cls.playwright.stop(); cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hippo-demo-test-')
        self.base = Path(self.temp.name)
        self.context = self.browser.new_context(viewport={'width': 1440, 'height': 1000}, reduced_motion='reduce')
        self.page = self.context.new_page()
        self.errors = []
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))
        self.page.goto(self.url)
        expect(self.page.locator('#run-caption')).to_contain_text('b6e81501-ff50-4322-a2e0-cc514ba1d475')
        self.page.locator('#full-experiment > summary').click()
        self.page.locator('#clear-replay').click()

    def tearDown(self):
        self.context.close(); self.temp.cleanup()
        self.assertEqual(self.errors, [], 'Browser errors')

    def upload(self, report=None, name='report.json'):
        path = self.base / name
        path.write_text(json.dumps(report if report is not None else fixture()), encoding='utf-8')
        self.page.locator('#report-file').set_input_files(path)
        return path

    def folder(self):
        root = self.base / 'evidence'; root.mkdir()
        (root / 'report.json').write_text(json.dumps(fixture()), encoding='utf-8')
        receipt_path = root / 'models/synthetic-model-for-ui-test/astra_A/receipts.json'
        receipt_path.parent.mkdir(parents=True)
        receipt_path.write_text(json.dumps(receipts()), encoding='utf-8')
        rows = [{'path': p.relative_to(root).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
                for p in (root / 'report.json', receipt_path)]
        (root / 'evidence-index.json').write_text(json.dumps({'run_id': fixture()['run_id'], 'files': rows}))
        return root

    def screenshot(self, name):
        # Optional development artifacts; never ship synthetic fixtures as real evidence.
        destination = os.environ.get('HIPPO_DEMO_SCREENSHOTS')
        if destination:
            target = Path(destination); target.mkdir(parents=True, exist_ok=True)
            self.page.evaluate('window.scrollTo(0, 0)')
            self.page.wait_for_timeout(150)
            self.page.screenshot(path=str(target / name), full_page=True)

    def test_empty_state_is_honest_and_responsive(self):
        expect(self.page.locator('#empty-state')).to_be_visible()
        expect(self.page.locator('#empty-state')).to_contain_text('Load the included run6 results')
        expect(self.page.locator('#workspace')).to_be_hidden()
        self.screenshot('empty-desktop.png')
        self.page.set_viewport_size({'width': 390, 'height': 844})
        self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
        self.screenshot('empty-mobile.png')

    def test_report_import_and_final_deltas(self):
        self.upload()
        expect(self.page.locator('#workspace')).to_be_visible()
        expect(self.page.locator('#diagnosis-name')).to_have_text('Eye blink')
        expect(self.page.locator('#run-warning')).to_contain_text('Completed with exclusions')
        self.page.locator('[data-stage="2"]').click()
        expect(self.page.locator('#delta-cards')).to_contain_text('+20.0 pp')
        expect(self.page.locator('#delta-cards')).to_contain_text('-1.0 pp')
        expect(self.page.locator('#delta-cards')).to_contain_text('-30.0 pp')
        expect(self.page.locator('#final-chart')).to_contain_text('60/100')
        expect(self.page.locator('#outcome')).to_contain_text('meets the improvement')
        self.screenshot('loaded-desktop.png')
        self.page.set_viewport_size({'width': 390, 'height': 844})
        self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'))
        self.screenshot('loaded-mobile.png')

    def test_real_run6_autoload_all_outcomes_and_receipts(self):
        self.page.reload()
        expect(self.page.locator('#run-caption')).to_contain_text('b6e81501-ff50-4322-a2e0-cc514ba1d475')
        self.page.locator('#full-experiment > summary').click()
        expect(self.page.locator('#overview-stats')).to_contain_text('5 / 6')
        expect(self.page.locator('#overview-stats')).to_contain_text('30')
        expect(self.page.locator('#overview-note')).to_contain_text('12/12 blind Astra diagnoses abstained')
        expect(self.page.locator('#overview-note')).to_contain_text('exploratory')
        expect(self.page.locator('#overview-results tr')).to_have_count(6)
        expect(self.page.locator('#overview-results')).to_contain_text('Does not meet')
        expect(self.page.locator('#model-select option')).to_have_count(6)
        expect(self.page.locator('#diagnosis-name')).to_have_text('Uncertain')
        self.page.locator('[data-stage="1"]').click()
        expect(self.page.locator('#edit-history .edit-row')).to_have_count(5)
        expect(self.page.locator('#edit-history')).to_contain_text('Scale temporal filter 0 by 0.8')
        self.page.locator('#model-select').select_option('5')
        self.page.locator('#method-select').select_option('astra_B')
        self.page.locator('[data-stage="2"]').click()
        expect(self.page.locator('#outcome')).to_contain_text('does not meet all')
        expect(self.page.locator('#delta-cards')).to_contain_text('-0.7 pp')
        expect(self.page.locator('#validation-context')).to_contain_text('previous run')
        self.screenshot('run6-desktop.png')
        for width in (320, 390, 768):
            self.page.set_viewport_size({'width':width, 'height':844})
            self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'), width)
        self.screenshot('run6-mobile.png')

    def test_bundled_provenance_and_preserved_clinical_demo(self):
        folder = ROOT / 'assets/evidence/run6'
        provenance = json.loads((folder / 'provenance.json').read_text())
        self.assertEqual(hashlib.sha256((folder / 'replay.json').read_bytes()).hexdigest(), provenance['replay_sha256'])
        self.assertEqual(provenance['source_files_verified_at_packaging'], 570)
        self.assertEqual(len(provenance['included_source_files']), 25)
        self.page.locator('#clinical-experiment > summary').click()
        expect(self.page.locator('#live-demo .ld-meta')).not_to_contain_text('Loading run')
        self.page.locator('#ld-ask-1').click()
        expect(self.page.locator('#ld-answer-1')).not_to_be_empty()
        expect(self.page.locator('#ld-run-fix')).to_be_enabled()
        self.page.locator('#ld-run-fix').click()
        # Reduced-motion replay finishes immediately, so the skip button is hidden.
        expect(self.page.locator('#ld-ask-2')).to_be_enabled()
        self.page.locator('#ld-ask-2').click()
        expect(self.page.locator('#ld-answer-2')).not_to_be_empty()

    def test_concise_intro_keeps_real_outcomes_and_opens_details(self):
        self.page.reload()
        expect(self.page.locator('#headline-result .spotlight-count')).to_have_text('5/ 6')
        expect(self.page.locator('#headline-result')).to_contain_text('6 edits retained from 30 candidates')
        expect(self.page.locator('#headline-result')).to_contain_text('1 edited comparison did not meet')
        expect(self.page.locator('#headline-result')).to_contain_text('Mean accuracy gain +1.5 pp')
        expect(self.page.locator('#headline-result')).to_contain_text('Mean accuracy gain +0.7 pp')
        expect(self.page.locator('#replay')).to_be_hidden()
        expect(self.page.locator('#live-demo')).to_be_hidden()
        self.screenshot('concise-desktop.png')
        for width in (320,390,768):
            self.page.set_viewport_size({'width':width,'height':844})
            self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'),width)
        self.page.set_viewport_size({'width':390,'height':844})
        self.screenshot('concise-mobile.png')
        self.page.locator('#show-edits').click()
        expect(self.page.locator('#replay-repair')).to_be_visible()
        expect(self.page.locator('#edit-history .edit-row')).to_have_count(5)
        self.page.locator('#full-experiment > summary').click()
        self.page.locator('#show-results').click()
        expect(self.page.locator('#replay-validate')).to_be_visible()

    def test_partial_does_not_substitute_development_scores(self):
        report = fixture(); report['status'] = 'failed'; report['final_evaluation_started'] = False
        for method in report['models'][0]['methods'].values():
            method.pop('final')
        self.upload(report)
        expect(self.page.locator('#run-warning')).to_contain_text('partial')
        self.page.locator('[data-stage="2"]').click()
        expect(self.page.locator('#validation-results')).to_be_hidden()
        expect(self.page.locator('#validation-missing')).to_contain_text('Development scores are not substituted')

    def test_excluded_model_has_no_invented_diagnosis(self):
        self.upload(); expect(self.page.locator('#workspace')).to_be_visible()
        self.page.locator('#model-select').select_option('1')
        expect(self.page.locator('#diagnosis-name')).to_have_text('No diagnosis recorded')
        expect(self.page.locator('#method-select')).to_be_disabled()
        self.page.locator('[data-stage="2"]').click()
        expect(self.page.locator('#validation-missing')).to_contain_text('excluded')

    def test_invalid_file_keeps_previous_run_and_text_never_executes(self):
        report = fixture()
        report['models'][0]['diagnoses']['astra_A']['response']['description'] = '<img src=x onerror="window.injected=true">'
        self.upload(report)
        expect(self.page.locator('#diagnosis-description')).to_contain_text('<img')
        self.assertIsNone(self.page.evaluate('window.injected'))
        self.assertEqual(self.page.locator('#diagnosis-description img').count(), 0)
        self.upload({'protocol': 'single_unit_cue_pruning_v1'}, 'old-report.json')
        expect(self.page.locator('#load-message')).to_contain_text('different format')
        expect(self.page.locator('#workspace')).to_be_visible()
        expect(self.page.locator('#run-caption')).to_contain_text('synthetic-ui-test')

    def test_rejects_mismatched_counts_and_bad_json(self):
        report = fixture(); report['models'][0]['methods']['astra_A']['final']['swapped']['correct'] = 99
        self.upload(report)
        expect(self.page.locator('#load-message')).to_contain_text('disagrees with its counts')
        expect(self.page.locator('#workspace')).to_be_hidden()
        path = self.base / 'bad.json'; path.write_text('invalid json')
        self.page.locator('#report-file').set_input_files(path)
        expect(self.page.locator('#load-message')).to_contain_text('not valid JSON')

    def test_folder_integrity_receipts_and_replay_roundtrip(self):
        root = self.folder()
        self.page.locator('#evidence-folder').set_input_files(root)
        expect(self.page.locator('#load-message')).to_contain_text('2 displayed files match')
        self.page.locator('[data-stage="1"]').click()
        expect(self.page.locator('#edit-history')).to_contain_text('Rejected')
        expect(self.page.locator('#edit-history')).to_contain_text('Accepted')
        expect(self.page.locator('#edit-history')).to_contain_text('Scale temporal filter 0 by 0.5')
        with self.page.expect_download() as download:
            self.page.locator('#save-replay').click()
        path = self.base / 'saved-replay.json'; download.value.save_as(path)
        saved = json.loads(path.read_text())
        self.assertEqual(saved['format'], 'hippo_eeg_replay_v1')
        self.page.locator('#clear-replay').click()
        expect(self.page.locator('#workspace')).to_be_hidden()
        expect(self.page.locator('#evidence-detail')).to_be_empty()
        self.page.locator('#report-file').set_input_files(path)
        expect(self.page.locator('#workspace')).to_be_visible()
        expect(self.page.locator('#load-message')).to_contain_text('no index check')
        self.page.locator('[data-stage="1"]').click()
        expect(self.page.locator('#edit-history')).to_contain_text('Accepted')

    def test_tampered_index_blocks_import(self):
        root = self.folder()
        with (root / 'report.json').open('a') as stream:
            stream.write(' ')
        self.page.locator('#evidence-folder').set_input_files(root)
        expect(self.page.locator('#load-message')).to_contain_text('hash mismatch')
        expect(self.page.locator('#workspace')).to_be_hidden()

    def test_zero_model_run_and_keyboard_navigation(self):
        empty = fixture(); empty['models'] = []; empty['status'] = 'failed'
        self.upload(empty)
        expect(self.page.locator('#run-warning')).to_contain_text('no completed models')
        expect(self.page.locator('#play-replay')).to_be_disabled()
        self.upload()
        self.page.locator('[data-stage="0"]').focus()
        self.page.keyboard.press('ArrowRight')
        expect(self.page.locator('#replay-repair')).to_be_visible()
        self.page.keyboard.press('End')
        expect(self.page.locator('#replay-report')).to_be_visible()
        self.page.locator('#next-stage').click()
        expect(self.page.locator('#replay-audit')).to_be_visible()

    def test_replay_play_pause_and_no_network_uploads(self):
        self.page.wait_for_load_state('networkidle')
        network = []
        self.page.on('request', lambda request: network.append((request.method, request.url)))
        self.upload(); expect(self.page.locator('#workspace')).to_be_visible()
        self.page.clock.install()
        self.page.locator('#play-replay').click()
        self.page.clock.fast_forward(7100)
        expect(self.page.locator('#replay-repair')).to_be_visible()
        self.page.locator('#play-replay').click()
        self.page.clock.fast_forward(15000)
        expect(self.page.locator('#replay-repair')).to_be_visible()
        self.page.locator('#play-replay').click()
        self.page.clock.run_for(15000)
        expect(self.page.locator('#replay-report')).to_be_visible()
        expect(self.page.locator('#play-replay')).to_have_text('Play replay ▷')
        self.assertEqual(network, [], 'Import/replay must not upload files or contact a provider')

    def test_all_stages_fit_small_mobile_viewports(self):
        self.upload(); expect(self.page.locator('#workspace')).to_be_visible()
        for width in (320, 390, 768):
            self.page.set_viewport_size({'width': width, 'height': 844})
            for index in range(4):
                self.page.locator(f'[data-stage="{index}"]').click()
                self.assertTrue(self.page.evaluate('document.documentElement.scrollWidth <= innerWidth'), (width, index))


if __name__ == '__main__':
    unittest.main()
