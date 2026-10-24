"""Step 3F safeguards: all model calls use mocked sessions and fake keys."""
import json
import os
import shutil
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

from src.llm_classifier import Config, ClassificationError, BudgetExceeded, LLMClassifier, stable_hash
from src.recover_large_pilot import resilient_atomic_json as atomic_json
from src.run_step3f import run, validate_prepared, unique_ids, review_analysis, CODE_PATHS
from src.step3f_classifier import (Step3FClassifier, source_spans, validate_span_prediction,
    POLICY_PATH, PROMPT_PATH, POLICY_VERSION, PROMPT_VERSION, VALIDATION_VERSION, SCHEMA)
from src.validate_classification import ROOT, sha256, read_csv
from tests.test_step3d import response


TITLE='Data Engineer'
BODY='Du bygger datapipelines på svenska. Förbättra datakvalitet.\n\u2028Fullständig text.'


def prediction(**changes):
    result=dict(ai_relevance='relevant',ai_primary_category='Data Engineer',
        ai_secondary_categories=[],ai_reasoning='Dedicated data pipeline responsibilities.',
        ai_evidence_span_ids=['D00000'],ai_confidence='high',ai_review_required=False,
        ai_scope='core_industry',ai_opportunity_types=[],ai_multi_role=False)
    result.update(changes)
    return result


class Step3FTests(unittest.TestCase):
    def setUp(self):
        self.temp=Path(__file__).parent/('.test-step3f-'+uuid.uuid4().hex);self.temp.mkdir()
        self.addCleanup(lambda:shutil.rmtree(self.temp))
        self.config=Config('mock-model',.4,1.6,api_key_env='STEP3F_FAKE_KEY',max_retries=0,min_interval=0)
        environment=patch.dict(os.environ,{'STEP3F_FAKE_KEY':'fake-never-sent'})
        environment.start();self.addCleanup(environment.stop)
        self.session=Mock();self.session.post.return_value=response(prediction())
        self.prepared=self.temp/'prepared';self.prepared.mkdir()
        self.frame=pd.DataFrame([dict(advertisement_id=str(i),job_title=TITLE+str(i),description_text=BODY,
            relevance='uncertain',primary_category='') for i in range(3)])
        self.frame.to_csv(self.prepared/'prepared_jobs.csv',index=False,encoding='utf-8-sig')
        (self.prepared/'policy.md').write_bytes(POLICY_PATH.read_bytes())
        (self.prepared/'prompt.txt').write_bytes(PROMPT_PATH.read_bytes())
        atomic_json(self.prepared/'schema.json',SCHEMA)
        self.manifest=dict(stage='step3f',total_advertisements=3,policy_version=POLICY_VERSION,
            prompt_version=PROMPT_VERSION,validation_version=VALIDATION_VERSION,
            policy_sha256=sha256(POLICY_PATH),prompt_sha256=sha256(PROMPT_PATH),
            schema_sha256=stable_hash(SCHEMA),prepared_sha256=sha256(self.prepared/'prepared_jobs.csv'),
            code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in CODE_PATHS})
        atomic_json(self.prepared/'manifest.json',self.manifest)

    def client(self,**changes):
        args=dict(config=self.config,policy_text=POLICY_PATH.read_text(encoding='utf-8'),
            cache_dir=self.temp/'cache',checkpoint_path=self.temp/'checkpoint.json',
            budget_usd=2,execute=True,session=self.session,sleep=lambda seconds:None)
        args.update(changes)
        return Step3FClassifier(**args)

    def runner(self,**changes):
        args=dict(prepared=self.prepared,output=self.temp/'run',config=self.config,budget_usd=2,
            cache_dir=self.temp/'cache',session=self.session)
        args.update(changes)
        return run(**args)

    def test_full_unicode_source_coverage_and_offsets(self):
        title='AI-utvecklare – Göteborg 😀';description=BODY*20
        spans=source_spans(title,description)
        for field,text in [('job_title',title),('description_text',description)]:
            selected=[s for s in spans if s['source_field']==field]
            self.assertEqual(''.join(s['text'] for s in selected),text)
            for s in selected:self.assertEqual(text[s['start']:s['end']],s['text'])

    def test_valid_span_is_exact_original_without_normalization(self):
        result=validate_span_prediction(prediction(),TITLE,BODY)
        self.assertEqual(result['ai_evidence'],[BODY])
        self.assertFalse(result['ai_review_required'])

    def test_unknown_duplicate_and_fabricated_evidence_rejected(self):
        for ids in [['D99999'],['D00000','D00000'],['build invented pipelines'],['T00001']]:
            with self.subTest(ids=ids),self.assertRaises(ClassificationError):
                validate_span_prediction(prediction(ai_evidence_span_ids=ids),TITLE,BODY)
        bad=prediction();bad['ai_evidence']='fabricated quote'
        with self.assertRaises(ClassificationError):validate_span_prediction(bad,TITLE,BODY)

    def test_invalid_schema_category_scope_and_types(self):
        cases=[{},prediction(ai_primary_category='Unknown'),prediction(ai_evidence_span_ids='D00000'),
            prediction(ai_scope='unrelated'),prediction(ai_secondary_categories=['Data Engineer']),
            prediction(ai_reasoning=' '),prediction(ai_opportunity_types=['doctoral','doctoral'])]
        for value in cases:
            with self.subTest(value=value),self.assertRaises(ClassificationError):
                validate_span_prediction(value,TITLE,BODY)

    def test_irrelevant_is_supported_unrelated_not_missing_evidence(self):
        value=prediction(ai_relevance='irrelevant',ai_primary_category=None,ai_scope='unrelated',
            ai_evidence_span_ids=['T00000'],ai_reasoning='Teaching responsibilities are unrelated.')
        result=validate_span_prediction(value,'Specialpedagog', 'Undervisa elever på skolan.')
        self.assertEqual(result['ai_evidence'],['Specialpedagog'])
        value['ai_evidence_span_ids']=[]
        with self.assertRaises(ClassificationError):validate_span_prediction(value,TITLE,BODY)

    def test_uncertain_can_abstain_but_cannot_claim_categories(self):
        value=prediction(ai_relevance='uncertain',ai_primary_category=None,ai_scope='unresolved',ai_evidence_span_ids=[])
        self.assertTrue(validate_span_prediction(value,TITLE,BODY)['ai_review_required'])
        value['ai_secondary_categories']=['AI Engineer']
        with self.assertRaises(ClassificationError):validate_span_prediction(value,TITLE,BODY)

    def test_governance_academia_management_and_opportunities_separate(self):
        for category,scope in [('Data Governance / Data Quality','data_ai_governance'),
            ('AI / ML Research','academic_ai_ml_research'),('Data / AI Management','adjacent_borderline')]:
            value=prediction(ai_primary_category=category,ai_scope=scope,ai_opportunity_types=['doctoral'])
            self.assertTrue(validate_span_prediction(value,TITLE,BODY)['ai_review_required'])
        with self.assertRaises(ClassificationError):
            validate_span_prediction(prediction(ai_scope='academic_ai_ml_research'),TITLE,BODY)

    def test_multi_role_allows_relevant_null_primary_and_requires_review(self):
        value=prediction(ai_primary_category=None,ai_secondary_categories=['Data Engineer','Data Scientist'],ai_multi_role=True)
        self.assertTrue(validate_span_prediction(value,TITLE,BODY)['ai_review_required'])

    def test_payload_preserves_full_source_and_ignores_prior_labels(self):
        payload=self.client(execute=False).payload(TITLE,BODY)
        user=json.loads(payload['messages'][1]['content'])
        self.assertEqual(''.join(s['text'] for s in user['source_spans'] if s['source_field']=='description_text'),BODY)
        self.assertEqual(set(user),{'source_spans'})
        self.assertTrue(payload['response_format']['json_schema']['strict'])

    def test_mock_success_cost_provenance_and_utf8_diagnostics(self):
        client=self.client();result=client.classify('1',TITLE,BODY)
        self.assertEqual(result['classification_status'],'Relevant')
        self.assertEqual(result['ai_prompt_version'],PROMPT_VERSION)
        self.assertEqual(result['ai_evidence_spans'][0]['text'],BODY)
        self.assertFalse(result['ai_human_verified'])
        self.assertEqual(self.session.post.return_value.encoding,'utf-8')
        self.assertGreater(client.cost_breakdown()['provider_reported_usage_estimated_cost_usd'],0)
        self.assertEqual(client.cost_breakdown()['unreported_usage_reserved_usd'],0)

    def test_rejection_is_technical_not_irrelevant_and_not_automatically_retried(self):
        self.session.post.return_value=response(prediction(ai_evidence_span_ids=['invented']))
        client=self.client();first=client.classify('1',TITLE,BODY)
        self.assertEqual(first['classification_status'],'Technical failure / requires retry')
        self.assertIsNone(first['ai_relevance'])
        self.assertIn('Unknown evidence span ID',client.state['ledger'][0]['model_response_diagnostic']['validation_error'])
        self.client().classify('1',TITLE,BODY)
        self.assertEqual(self.session.post.call_count,1)

    def test_checkpoint_and_content_cache_skip_paid_requests(self):
        self.client().classify('1',TITLE,BODY)
        self.client().classify('1',TITLE,BODY)
        cached=self.client().classify('2',TITLE,BODY)
        self.assertEqual(cached['ai_status'],'cache_hit')
        self.assertEqual(cached['ai_estimated_cost_usd'],0)
        self.assertEqual(self.session.post.call_count,1)

    def test_cache_receipt_recovers_checkpoint_after_commit_failure(self):
        client=self.client();original_save=client.save
        def fail_after_receipt():
            if client.state['results']:raise PermissionError('Mock crash after cache commit')
            original_save()
        client.save=fail_after_receipt
        with self.assertRaises(PermissionError):client.classify('1',TITLE,BODY)
        recovered=self.client().classify('1',TITLE,BODY)
        self.assertEqual(recovered['ai_status'],'cache_hit')
        self.assertEqual(self.session.post.call_count,1)

    def test_unknown_network_outcome_holds_reservation_and_blocks_repayment(self):
        self.session.post.side_effect=requests.Timeout('mock')
        client=self.client();result=client.classify('1',TITLE,BODY)
        self.assertEqual(result['ai_status'],'request_outcome_unknown')
        self.assertGreater(client.cost_breakdown()['unreported_usage_reserved_usd'],0)
        self.client().classify('2',TITLE,BODY)
        self.assertEqual(self.session.post.call_count,1)

    def test_no_retries_even_when_requested(self):
        with self.assertRaises(ValueError):self.client(config=replace(self.config,max_retries=1))
        with self.assertRaises(ValueError):self.client(retry_unknown=True)

    def test_budget_enforced_before_post(self):
        with self.assertRaises(BudgetExceeded):self.client(budget_usd=.0000001).classify('1',TITLE,BODY)
        self.session.post.assert_not_called()

    def test_missing_description_is_local_uncertain_and_resumable(self):
        result=self.client().classify('1',TITLE,'')
        self.assertEqual(result['classification_status'],'Uncertain')
        self.client().classify('1',TITLE,'')
        self.session.post.assert_not_called()

    def test_old_schema_cache_identity_is_not_reused(self):
        client=self.client(execute=False)
        old=LLMClassifier(self.config,client.policy_text,self.temp/'oldcache',self.temp/'old.json',2)
        self.assertNotEqual(client.cache_key(TITLE,BODY),old.cache_key(TITLE,BODY))
        newer=self.client(execute=False,config=replace(self.config,model='different'))
        self.assertNotEqual(client.cache_key(TITLE,BODY),newer.cache_key(TITLE,BODY))

    def test_dry_run_permissions_forecast_and_zero_requests(self):
        report=self.runner()
        self.assertEqual(report['total_advertisements'],3)
        self.assertEqual(report['unique_uncached_requests'],3)
        self.assertGreater(report['conservative_cost_usd'],report['expected_cost_usd'])
        self.assertFalse((self.temp/'run/checkpoint.json').exists())
        self.session.post.assert_not_called()

    def test_runner_resume_skips_successes_and_preserves_source(self):
        before=sha256(self.prepared/'prepared_jobs.csv')
        self.runner(execute=True)
        self.runner(execute=True,resume=True)
        self.assertEqual(self.session.post.call_count,3)
        self.assertEqual(before,sha256(self.prepared/'prepared_jobs.csv'))
        rows=read_csv(self.temp/'run/classification_results.csv')
        self.assertEqual(set(rows.classification_status),{'Relevant'})

    def test_runner_rejects_changed_config_source_and_checkpoint_before_post(self):
        self.runner(execute=True)
        self.session.post.reset_mock()
        with self.assertRaises(ValueError):self.runner(execute=True,resume=True,config=replace(self.config,model='other'))
        path=self.temp/'run/checkpoint.json';state=json.loads(path.read_text(encoding='utf-8'))
        state['identity']['stage']='different';atomic_json(path,state)
        with self.assertRaises(ValueError):self.runner(execute=True,resume=True)
        self.session.post.assert_not_called()

    def test_permissions_failure_happens_before_post(self):
        with patch('src.run_step3f.writable_probe',side_effect=PermissionError('Mock read-only folder')):
            with self.assertRaises(PermissionError):self.runner(execute=True)
        self.session.post.assert_not_called()

    def test_prepared_tampering_and_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):unique_ids(pd.concat([self.frame,self.frame]))
        with (self.prepared/'prepared_jobs.csv').open('a',encoding='utf-8') as file:file.write('\n')
        with self.assertRaises(ValueError):validate_prepared(self.prepared)

    def test_provisional_review_never_becomes_ground_truth(self):
        pilot=self.frame.copy();pilot['ai_status']=['completed','invalid_model_response','cache_hit']
        pilot['ai_relevance']=['relevant','','irrelevant'];pilot['ai_primary_category']=['Data Engineer','','']
        review=pilot.copy();review['assistant_suggested_relevance']=['irrelevant','relevant','irrelevant']
        review['assistant_suggested_primary_category']=['','Data Engineer','']
        summary=review_analysis(pilot,review,self.temp)
        self.assertEqual(summary['relevance_disagreements'],1)
        self.assertEqual(summary['human_verified_labels_used'],0)
        self.assertFalse(summary['labels_are_ground_truth'])


if __name__=='__main__':unittest.main()
