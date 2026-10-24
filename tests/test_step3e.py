"""Step 3E tests use mocked API sessions only."""
import io
import json
import os
import shutil
import unittest
import uuid
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock,patch

import pandas as pd

from src.llm_classifier import Config,ClassificationError,validate_prediction,atomic_json
from src.prepare_large_pilot import proportional_sample
from src.run_large_pilot import run,validate_prepared
from src.step3e_classifier import Step3EClassifier,POLICY_PATH,PROMPT_PATH,multi_role_candidate
from src.validate_classification import sha256
from tests.test_step3d import response,prediction,BODY


class Step3ETests(unittest.TestCase):
    def setUp(self):
        self.temp=Path(__file__).parent/('.test-step3e-'+uuid.uuid4().hex);self.temp.mkdir()
        self.addCleanup(lambda:shutil.rmtree(self.temp))
        self.frame=pd.DataFrame([
            dict(advertisement_id=str(i),job_title=f'Data Engineer {i}',description_text=BODY,
                 relevance=(['irrelevant']*3+['relevant']*6+['uncertain']*11)[i],primary_category='') for i in range(20)])
        self.excluded={'0','3','9'}
        self.sample,self.breakdown=proportional_sample(self.frame,self.excluded,size=6,seed=52)
        self.prepared=self.temp/'prepared';self.prepared.mkdir()
        path=self.prepared/'representative_200.csv';self.sample.to_csv(path,index=False,encoding='utf-8-sig')
        atomic_json(self.prepared/'sampling_manifest.json',dict(stage='step3e',sample_size=6,
            sample_sha256=sha256(path),sample_ids=self.sample.advertisement_id.tolist(),excluded_ids=sorted(self.excluded),
            sample_scope='Test eligible frame'))
        self.config=Config(model='mock-model',input_usd_per_million=.4,output_usd_per_million=1.6,
            api_key_env='STEP3E_TEST_KEY',max_retries=0,min_interval=0)
        self.environment=patch.dict(os.environ,{'STEP3E_TEST_KEY':'fake-never-sent'})
        self.environment.start();self.addCleanup(self.environment.stop)

    def runner(self,**kwargs):
        args=dict(prepared=self.prepared,config=self.config,budget_usd=1,output=self.temp/'run',
                  cache_dir=self.temp/'cache',max_ads=6,sleep=lambda seconds:None)
        args.update(kwargs)
        with redirect_stdout(io.StringIO()):return run(**args)

    def test_policy_covers_pilot_boundaries_without_changing_categories(self):
        text=POLICY_PATH.read_text(encoding='utf-8')
        for phrase in ['ORM/entity modeling','HR/recruitment','Analyst versus scientist','Generic titles',
                       'Internships','Multi-role advertisements','not independent human labels']:
            self.assertIn(phrase,text)
        prompt=PROMPT_PATH.read_text(encoding='utf-8')
        for phrase in ['substantive statistical/model development','source']:
            self.assertIn(phrase,prompt)

    def test_proportional_sampling_is_deterministic_and_order_independent(self):
        a,b=proportional_sample(self.frame,self.excluded,6,52)
        c,d=proportional_sample(self.frame.sample(frac=1,random_state=9),self.excluded,6,52)
        pd.testing.assert_frame_equal(a,c);self.assertEqual(b,d)
        self.assertEqual(a.relevance.value_counts().to_dict(),{'uncertain':3,'relevant':2,'irrelevant':1})

    def test_no_duplicate_ids_or_previous_pilot_and_full_descriptions_preserved(self):
        self.assertEqual(self.sample.advertisement_id.nunique(),6)
        self.assertFalse(set(self.sample.advertisement_id)&self.excluded)
        self.assertTrue(self.sample.description_text.eq(BODY).all())
        bad=self.frame.copy();bad.loc[1,'advertisement_id']='0'
        with self.assertRaises(ValueError):proportional_sample(bad,self.excluded,6)
        with self.assertRaises(ValueError):proportional_sample(self.frame,self.excluded,201)

    def test_multi_role_is_a_review_candidate_not_forced_category(self):
        self.assertTrue(multi_role_candidate('Data Architects and Scientists -2026',''))
        self.assertTrue(multi_role_candidate('Consultants','The following positions: engineers'))
        self.assertFalse(multi_role_candidate('Backend developer','Works with data scientists'))

    def test_new_prompt_identity_dry_run_has_zero_api_calls(self):
        session=Mock()
        with patch('requests.sessions.Session.request',side_effect=AssertionError('Network forbidden')):
            out,s=self.runner(session=session)
        session.post.assert_not_called()
        self.assertEqual(s['prompt_version'],'step3e-1.0')
        self.assertEqual(s['accounted_cost_usd'],0)
        self.assertIsNone(s['technical_success_rate'])
        self.assertEqual(s['forecast']['uncached_requests_needed'],6)
        self.assertEqual(s['forecast']['cached'],0)
        self.assertFalse((out/'checkpoint.json').exists())

    def test_budget_stops_before_network_and_resume_preserves_completed(self):
        session=Mock();session.post.return_value=response()
        out,s=self.runner(execute=True,session=session,budget_usd=.000001)
        self.assertEqual(s['status'],'budget_stopped');session.post.assert_not_called()
        self.assertEqual(s['accounted_cost_usd'],0)
        out,s=self.runner(execute=True,session=session,output=None,resume=out)
        self.assertEqual(s['accepted'],6);self.assertEqual(session.post.call_count,6)
        self.runner(execute=True,session=session,output=None,resume=out)
        self.assertEqual(session.post.call_count,6)
        self.assertGreater(s['provider_reported_usage_estimated_cost_usd'],0)
        self.assertEqual(s['unreported_usage_reserved_usd'],0)

    def test_resume_rejects_changed_model_and_sample(self):
        out,_=self.runner()
        with self.assertRaises(ValueError):self.runner(output=None,resume=out,config=replace(self.config,model='different'))
        path=self.prepared/'representative_200.csv';path.write_bytes(path.read_bytes()+b'\n')
        with self.assertRaises(ValueError):self.runner(output=None,resume=out)

    def test_exact_evidence_schema_and_safe_whitespace_are_preserved(self):
        for bad in [prediction(ai_evidence='invented'),prediction(ai_primary_category='Not a category'),{'ai_relevance':'relevant'}]:
            with self.assertRaises(ClassificationError):validate_prediction(bad,'Title',BODY,allow_formatting=True)
        session=Mock();session.post.return_value=response(prediction(ai_evidence='build data pipelines'))
        client=Step3EClassifier(self.config,POLICY_PATH.read_text(encoding='utf-8'),self.temp/'cache',
            self.temp/'cp.json',1,execute=True,session=session)
        result=client.classify('1','Data Architects and Scientists','You build\ndata pipelines.')
        self.assertEqual(result['ai_evidence_method'],'whitespace_only')
        self.assertTrue(result['ai_review_required'])
        self.assertTrue(result['ai_multi_role_review_candidate'])
        self.assertEqual(result['ai_prompt_version'],'step3e-1.0')
        self.assertNotIn('assistant_suggested',json.dumps(session.post.call_args.kwargs['json']))

    def test_successful_outputs_use_step3e_provenance_and_no_human_verification(self):
        session=Mock();session.post.return_value=response()
        out,s=self.runner(execute=True,session=session)
        self.assertEqual(s['accepted'],6);self.assertEqual(s['technical_success_rate'],1)
        rows=[json.loads(l) for l in (out/'pilot_results.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(all(r['ai_prompt_version']=='step3e-1.0' and r['ai_human_verified'] is False for r in rows))
        self.assertEqual(len(s['by_sampling_group']),3)

    def test_runner_refuses_full_dataset_labels_and_overwrite(self):
        with self.assertRaises(ValueError):self.runner(max_ads=5023)
        self.runner()
        with self.assertRaises(FileExistsError):self.runner()
        f=self.sample.copy();f['assistant_reference']='irrelevant'
        path=self.prepared/'representative_200.csv';f.to_csv(path,index=False,encoding='utf-8-sig')
        m=json.loads((self.prepared/'sampling_manifest.json').read_text());m['sample_sha256']=sha256(path)
        atomic_json(self.prepared/'sampling_manifest.json',m)
        with self.assertRaisesRegex(ValueError,'suggestions or human'):validate_prepared(self.prepared)

    def test_auth_failure_stops_and_retains_all_rows(self):
        session=Mock();session.post.return_value=response(status=401,usage=False)
        out,s=self.runner(execute=True,session=session)
        self.assertEqual(session.post.call_count,1);self.assertEqual(s['selected'],6)
        self.assertEqual(s['accepted'],0)
        self.assertGreater(s['unreported_usage_reserved_usd'],0)
        self.assertEqual(s['provider_reported_usage_estimated_cost_usd'],0)


if __name__=='__main__':unittest.main()
