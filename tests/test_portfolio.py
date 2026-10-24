"""Public-clone demo, missing-data handling and repository safety checks."""

import json
import shutil
import subprocess
import unittest
import uuid
from unittest.mock import patch

from src.dashboard_data import dashboard_dataset, filter_data, load_dataset
from src.run_step7 import verify_evidence
from tools.build_demo import DESTINATION, ROOT, build
from tools.check_repository import LOCAL_PATH, notebook_issues, secret_kinds


class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.temp = ROOT / 'tests' / ('.test-portfolio-' + uuid.uuid4().hex)
        self.temp.mkdir()

    def tearDown(self):
        self.assertTrue(self.temp.resolve().is_relative_to((ROOT / 'tests').resolve()))
        shutil.rmtree(self.temp)

    def test_demo_is_small_fictional_and_source_grounded(self):
        jobs, facts = load_dataset(DESTINATION)
        self.assertEqual(len(jobs), 6)
        self.assertTrue(jobs.advertisement_id.str.startswith('demo-').all())
        self.assertTrue(jobs.employer.str.startswith('Fictional Example').all())
        self.assertTrue(jobs.ai_model.eq('synthetic_fixture_no_model').all())
        self.assertTrue(jobs.original_advertisement_url.eq('').all())
        verify_evidence(jobs, facts)
        self.assertIn('required', set(facts.modality))
        self.assertIn('preferred', set(facts.modality))
        self.assertIn('mentioned_only', set(facts.modality))

    def test_demo_rebuild_is_deterministic_and_offline(self):
        with patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')):
            build(self.temp / 'demo')
        for original in DESTINATION.iterdir():
            if original.is_file():
                self.assertEqual((self.temp/'demo'/original.name).read_bytes(), original.read_bytes())

    def test_missing_research_uses_demo_but_override_is_respected(self):
        self.assertEqual(dashboard_dataset(self.temp / 'missing'), DESTINATION)
        explicit = self.temp / 'explicit-missing'
        self.assertEqual(dashboard_dataset(self.temp, str(explicit)), explicit)

    def test_completed_local_publication_keeps_priority(self):
        directory = self.temp / 'local'
        directory.mkdir()
        for name in ['jobs_requirements.jsonl', 'requirements_evidence.csv', 'step7_summary.json']:
            (directory/name).write_text('', encoding='utf-8')
        (directory/'manifest.json').write_text(json.dumps({'stage':'step7-ai-relevant'}), encoding='utf-8')
        self.assertEqual(dashboard_dataset(self.temp), directory)

    def test_incomplete_manifest_does_not_break_fallback(self):
        directory = self.temp / 'broken'
        directory.mkdir()
        (directory/'step7_summary.json').write_text('{}', encoding='utf-8')
        (directory/'manifest.json').write_text('incomplete JSON', encoding='utf-8')
        self.assertEqual(dashboard_dataset(self.temp), DESTINATION)

    def test_source_and_quality_filters_on_demo(self):
        jobs, facts = load_dataset(DESTINATION)
        core, evidence = filter_data(jobs, facts, analysis_scope=['core_industry'])
        self.assertEqual(len(core), 3)
        self.assertTrue(set(evidence.advertisement_id) <= set(core.advertisement_id))
        strict, _ = filter_data(core, evidence, strict_audit_approved=[True])
        self.assertEqual(len(strict), 1)

    def test_demo_app_launches_and_filters_without_network(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict('os.environ', {'JOBS_DASHBOARD_DATA':str(DESTINATION)}), patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')):
            app = AppTest.from_file(str(ROOT/'dashboard/app.py')).run(timeout=60)
            self.assertFalse(app.exception, str(app.exception))
            self.assertTrue(any('FICTIONAL DEMO' in item.value for item in app.warning))
            self.assertEqual(int(app.metric[3].value), 3)
            app.selectbox(key='inclusion_quality').set_value('Audit-approved only').run(timeout=60)
            self.assertFalse(app.exception, str(app.exception))
            self.assertEqual(int(app.metric[3].value), 1)

    def test_missing_or_partial_explicit_dataset_shows_help(self):
        from streamlit.testing.v1 import AppTest
        (self.temp/'jobs_requirements.jsonl').write_text('{}\n', encoding='utf-8')
        with patch.dict('os.environ', {'JOBS_DASHBOARD_DATA':str(self.temp)}):
            app = AppTest.from_file(str(ROOT/'dashboard/app.py')).run(timeout=60)
            self.assertFalse(app.exception, str(app.exception))
            self.assertTrue(app.error)
            self.assertIn('examples/demo_dataset', app.error[0].value)

    def test_invalid_dataset_shows_help_without_modifying_it(self):
        from streamlit.testing.v1 import AppTest
        (self.temp/'jobs_requirements.jsonl').write_text('not JSON', encoding='utf-8')
        (self.temp/'requirements_evidence.csv').write_text('advertisement_id\n', encoding='utf-8')
        with patch.dict('os.environ', {'JOBS_DASHBOARD_DATA':str(self.temp)}):
            app = AppTest.from_file(str(ROOT/'dashboard/app.py')).run(timeout=60)
            self.assertFalse(app.exception, str(app.exception))
            self.assertTrue(app.error)
        self.assertEqual((self.temp/'jobs_requirements.jsonl').read_text(encoding='utf-8'), 'not JSON')

    def test_secret_detector_reports_types_without_values(self):
        value = ('sk-' + 'X'*32).encode()
        self.assertIn('provider_token', secret_kinds(value))
        self.assertIn('exact_local_credential', secret_kinds(value, [value]))
        self.assertEqual(secret_kinds(b'OPENAI_API_KEY='), [])
        self.assertNotIn(value.decode(), json.dumps(secret_kinds(value)))

    def test_public_notebooks_have_no_outputs_or_execution_state(self):
        for path in (ROOT/'notebooks').rglob('*.ipynb'):
            if '.ipynb_checkpoints' not in path.parts:
                self.assertFalse(notebook_issues(path.read_text(encoding='utf-8')), str(path))
        sample = json.dumps(dict(cells=[dict(cell_type='code', execution_count=1, outputs=[])]))
        self.assertTrue(notebook_issues(sample))

    @unittest.skipUnless((ROOT/'.git').exists(), 'Git metadata is not included in a deployment bundle')
    def test_sensitive_and_generated_files_are_ignored(self):
        for name in ['.env','.env.local','.streamlit/secrets.toml','data/analysis/report.csv',
                     'data/processed/run/checkpoint.json','.local-archive/original.ipynb','session.log']:
            result = subprocess.run(['git','check-ignore','--no-index',name], cwd=ROOT, capture_output=True)
            self.assertEqual(result.returncode,0,name)
        result = subprocess.run(['git','check-ignore','--no-index','.env.example'],cwd=ROOT,capture_output=True)
        self.assertEqual(result.returncode,1)

    def test_home_path_detector(self):
        windows = 'C:' + chr(92) + 'Users' + chr(92) + 'example' + chr(92) + 'project'
        self.assertTrue(LOCAL_PATH.search(windows))
        self.assertFalse(LOCAL_PATH.search('data/processed/example'))


if __name__ == '__main__':
    unittest.main()
