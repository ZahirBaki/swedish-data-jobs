import json
import shutil
import uuid
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

from src import step6
from src.dashboard_data import latest_dataset, load_dataset, filter_data, coverage
from src.llm_classifier import Config
from src.recover_large_pilot import resilient_atomic_json as atomic_json
from src.validate_classification import sha256
from tests.test_step3d import response
from tests.test_step3f import TITLE, BODY, prediction


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).parent/('.test-step6-'+uuid.uuid4().hex);self.root.mkdir()
        self.addCleanup(lambda:shutil.rmtree(self.root))
        self.prepared=self.root/'prepared';self.prepared.mkdir()
        self.cache=self.root/'cache'
        self.frame=pd.DataFrame([dict(advertisement_id='new1',job_title=TITLE,description_text=BODY)])
        self.frame.to_csv(self.prepared/'selected_jobs.csv',index=False)
        # Unit tests use committed fictional fixtures, not private research runs.
        self.addCleanup(patch.stopall)
        patch.object(step6, 'SOURCE', self.prepared/'selected_jobs.csv').start()
        patch.object(step6, 'BASE', Path(__file__).resolve().parents[1]/'examples/demo_dataset').start()
        self.baseline, self.baseline_facts = load_dataset(step6.BASE)
        registry = self.baseline.assign(step4_disposition='eligible_relevant')
        patch.object(step6, 'load_existing', return_value=(registry, [], [])).start()
        atomic_json(self.prepared/'manifest.json',dict(identities=step6.identities(),
            input_sha256=sha256(self.prepared/'selected_jobs.csv'),config=step6.asdict(Config('mock',.4,1.6,max_retries=0))))
        self.session=Mock();self.session.post.return_value=response(prediction())
        patch.object(step6,'CACHE',self.cache).start()
        patch.object(step6,'history_guard',return_value=(set(),set(),set(),{})).start()
        patch.dict('os.environ',{'OPENAI_API_KEY':'mock'}).start()

    def run_step(self,**kw):
        return step6.run(self.prepared,self.root/'run',session=self.session,**kw)

    def test_dry_run_no_network_then_live_and_resume_once(self):
        self.run_step()
        self.session.post.assert_not_called()
        self.run_step(execute=True,resume=True)
        before=(self.root/'run/checkpoint.json').read_bytes()
        self.run_step(execute=True,resume=True)
        self.assertEqual(self.session.post.call_count,1)
        state=json.loads(before)
        self.assertEqual(state['results'],json.loads((self.root/'run/checkpoint.json').read_bytes())['results'])

    def test_unknown_timeout_is_never_retried(self):
        self.session.post.side_effect=requests.exceptions.Timeout('mock')
        self.run_step(execute=True)
        self.run_step(execute=True,resume=True)
        self.assertEqual(self.session.post.call_count,1)
        state=json.loads((self.root/'run/checkpoint.json').read_text())
        self.assertEqual(state['results']['new1']['ai_status'],'request_outcome_unknown')

    def test_invalid_response_is_not_retried(self):
        self.session.post.return_value=response(prediction(),finish='length')
        self.run_step(execute=True);self.run_step(execute=True,resume=True)
        self.assertEqual(self.session.post.call_count,1)

    def test_history_guard_prevents_duplicate_request(self):
        with patch.object(step6,'history_guard',return_value=({'new1'},set(),set(),{})):
            report=self.run_step(execute=True)
        self.assertEqual(report['actions'],{'previously_processed':1})
        self.session.post.assert_not_called()

    def test_input_change_refuses_resume(self):
        self.run_step()
        self.frame.assign(job_title='changed').to_csv(self.prepared/'selected_jobs.csv',index=False)
        with self.assertRaisesRegex(ValueError,'changed'):self.run_step(resume=True)
        self.session.post.assert_not_called()

    def test_budget_prevents_request(self):
        self.run_step(execute=True,budget=.0000001)
        self.session.post.assert_not_called()

    def test_merge_preserves_baseline_evidence_unique_ids(self):
        jobs,facts=load_dataset(step6.BASE)
        merged,evidence=step6.merge_frames(jobs,facts,jobs.copy(),facts.copy())
        self.assertEqual(len(merged),len(self.baseline))
        pd.testing.assert_frame_equal(evidence,facts)
        self.assertFalse(merged.advertisement_id.duplicated().any())

    def test_publish_requires_duties_audit_and_refreshes(self):
        self.run_step(execute=True)
        output=self.root/'published'
        report=step6.publish(self.prepared,self.root/'run',output)
        self.assertEqual(report['new_analyzed'],0)
        self.assertEqual(report['expansion_statuses'],{'uncertain':1})
        self.assertEqual(latest_dataset(self.root),output)
        jobs,facts=load_dataset(output)
        selected,evidence=filter_data(jobs,facts,reporting_scope=['core_industry'])
        self.assertTrue(coverage(selected,evidence).denominator_jobs.eq(len(selected)).all())
        # A half-written publication is never selected.
        incomplete=self.root/'incomplete';incomplete.mkdir()
        atomic_json(incomplete/'step6_summary.json',{})
        self.assertEqual(latest_dataset(self.root),output)

    def test_audited_new_job_is_extracted_and_combined(self):
        self.run_step(execute=True)
        audit=self.root/'audit.csv'
        pd.DataFrame([dict(advertisement_id='new1',duties_assessment='clear',category_assessment='clear',
            duties_assessor='test auditor',inclusion_audit_notes='Source explicitly describes data engineering duties',
            inclusion_review_resolved='true')]).to_csv(audit,index=False)
        report=step6.publish(self.prepared,self.root/'run',self.root/'combined',audit)
        self.assertEqual(report['new_analyzed'],1)
        jobs,facts=load_dataset(self.root/'combined')
        self.assertEqual(len(jobs),len(self.baseline)+1)
        self.assertTrue(coverage(jobs,facts).denominator_jobs.eq(len(self.baseline)+1).all())
        self.assertEqual(jobs.set_index('advertisement_id').loc['new1','ai_prompt_version'],'step3f-1.1')

if __name__=='__main__':unittest.main()
