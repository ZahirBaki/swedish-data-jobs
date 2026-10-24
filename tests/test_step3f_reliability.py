"""Mock-only regressions for the four completed-pilot failure modes."""
import json
from pathlib import Path

import pandas as pd
import requests

from src.llm_classifier import ClassificationError,stable_hash
from src.recover_large_pilot import resilient_atomic_json as atomic_json
from src.step3f_classifier import Step3FClassifier
from src.step3f_reliable_classifier import (Step3FReliableClassifier,schema_for,validate_reliable_prediction,
    source_hash,PROMPT_PATH,PROMPT_VERSION,VALIDATION_VERSION)
from src.prepare_step3f_reliability import prepare_human_reserve
from src.validate_classification import read_csv,sha256
from tests import test_step3f as fixtures
from tests.test_step3d import response
import unittest


class ReliableSpanTests(unittest.TestCase):
    def setUp(self):fixtures.Step3FTests.setUp(self)

    def client(self,**changes):
        args=dict(config=self.config,policy_text=fixtures.POLICY_PATH.read_text(encoding='utf-8'),
            cache_dir=self.temp/'cache',checkpoint_path=self.temp/'checkpoint.json',budget_usd=1,
            execute=True,session=self.session,sleep=lambda seconds:None,blocked_source_hashes=[])
        args.update(changes)
        return Step3FReliableClassifier(**args)

    def test_per_ad_enum_allows_only_actual_atomic_ids(self):
        schema=schema_for(fixtures.TITLE,fixtures.BODY)
        self.assertEqual(schema['properties']['ai_evidence_span_ids']['items']['enum'],['T00000','D00000'])
        self.assertEqual(schema['properties']['ai_evidence_span_ids']['maxItems'],3)
        for ids in [["D00005','D00006','D00008"],['D00000,D00001'],['D99999']]:
            with self.assertRaises(ClassificationError):
                validate_reliable_prediction(fixtures.prediction(ai_evidence_span_ids=ids),fixtures.TITLE,fixtures.BODY)

    def test_exact_historical_combined_id_rejected_not_split_or_repaired(self):
        value=fixtures.prediction(ai_evidence_span_ids=['D00003','D00004',"D00005','D00006','D00008"])
        with self.assertRaises(ClassificationError):validate_reliable_prediction(value,'Senior data engineer','Data '*1200)
        self.session.post.return_value=response(value)
        result=self.client().classify('bad','Senior data engineer','Data '*1200)
        self.assertEqual(result['ai_status'],'invalid_model_response')
        self.assertIsNone(result['ai_relevance'])

    def test_duplicate_id_and_more_than_three_still_rejected(self):
        for ids in [['D00000','D00000'],['D00000','D00001','D00002','D00003']]:
            with self.assertRaises(ClassificationError):
                validate_reliable_prediction(fixtures.prediction(ai_evidence_span_ids=ids),fixtures.TITLE,'Data '*1000)

    def test_management_core_scope_rejected_and_adjacent_accepted(self):
        value=fixtures.prediction(ai_primary_category='Data / AI Management',ai_scope='core_industry')
        with self.assertRaisesRegex(ClassificationError,'Statistics/management require separately tracked adjacent scope'):
            validate_reliable_prediction(value,fixtures.TITLE,fixtures.BODY)
        value['ai_scope']='adjacent_borderline'
        self.assertTrue(validate_reliable_prediction(value,fixtures.TITLE,fixtures.BODY)['ai_review_required'])
        text=PROMPT_PATH.read_text(encoding='utf-8')
        self.assertIn('"adjacent_borderline", NEVER "core_industry"',text)

    def test_primary_scope_rule_does_not_reclassify_secondary_management(self):
        value=fixtures.prediction(ai_secondary_categories=['Data / AI Management'])
        self.assertEqual(validate_reliable_prediction(value,fixtures.TITLE,fixtures.BODY)['ai_scope'],'core_industry')

    def test_length_finish_is_rejected_even_if_json_looks_complete(self):
        self.session.post.return_value=response(fixtures.prediction(),finish='length',usage=dict(prompt_tokens=2403,completion_tokens=1024))
        client=self.client();result=client.classify('1',fixtures.TITLE,fixtures.BODY)
        self.assertEqual(result['ai_status'],'invalid_model_response')
        diagnostic=client.state['ledger'][0]['model_response_diagnostic']
        self.assertEqual(diagnostic['finish_reason'],'length')
        self.assertTrue(diagnostic['incomplete_response'])
        self.assertEqual(diagnostic['configured_output_token_cap'],1024)
        self.assertEqual(diagnostic['reported_completion_tokens'],1024)
        self.assertEqual(len(list((self.temp/'cache').glob('*.json'))),0)

    def test_salesperson_runaway_whitespace_is_preserved_as_diagnostic_not_label(self):
        reply=response(fixtures.prediction(),finish='length')
        body=reply.json.return_value
        body['choices'][0]['message']['content']='{"ai_relevance":"irrelevant","ai_reasoning":"Sales ("'+' \r\n '*1000
        self.session.post.return_value=reply
        client=self.client();result=client.classify('sales','Marknadsföring säljare',fixtures.BODY)
        self.assertIsNone(result['ai_relevance'])
        diagnostic=client.state['ledger'][0]['model_response_diagnostic']
        self.assertGreater(diagnostic['captured_whitespace_fraction'],.9)
        self.assertEqual(diagnostic['validation_stage'],'completion_status')

    def test_explanation_last_short_and_output_cap_not_silently_raised(self):
        payload=self.client(execute=False).payload(fixtures.TITLE,fixtures.BODY)
        schema=payload['response_format']['json_schema']['schema']
        self.assertEqual(list(schema['properties'])[-1],'ai_reasoning')
        self.assertEqual(schema['properties']['ai_reasoning']['maxLength'],400)
        self.assertEqual(payload['max_completion_tokens'],1024)
        with self.assertRaises(ClassificationError):
            validate_reliable_prediction(fixtures.prediction(ai_reasoning='x'*401),fixtures.TITLE,fixtures.BODY)

    def test_valid_sweden_unicode_offsets_cache_resume_and_provenance(self):
        client=self.client();result=client.classify('1',fixtures.TITLE,fixtures.BODY)
        self.assertEqual(result['ai_evidence'],[fixtures.BODY])
        self.assertEqual(result['ai_prompt_version'],PROMPT_VERSION)
        self.assertEqual(result['ai_validation_version'],VALIDATION_VERSION)
        self.assertEqual(result['ai_response_schema_sha256'],stable_hash(schema_for(fixtures.TITLE,fixtures.BODY)))
        self.client().classify('1',fixtures.TITLE,fixtures.BODY)
        cached=self.client().classify('2',fixtures.TITLE,fixtures.BODY)
        self.assertEqual(cached['ai_status'],'cache_hit')
        self.assertEqual(self.session.post.call_count,1)

    def test_historical_unknown_blocks_new_version_and_alternate_ad_ids(self):
        marker=self.temp/'cache';marker.mkdir()
        pending=marker/'old-key.pending.json';atomic_json(pending,dict(attempt_id='old'))
        before=sha256(pending)
        client=self.client(blocked_source_hashes=[source_hash(fixtures.TITLE,fixtures.BODY)])
        for identifier in ['old','copy-id']:
            result=client.classify(identifier,fixtures.TITLE,fixtures.BODY)
            self.assertEqual(result['ai_status'],'historical_request_outcome_unknown')
            self.assertIsNone(result['ai_relevance'])
        self.assertEqual(client.spent,0)
        self.assertEqual(sha256(pending),before)
        self.session.post.assert_not_called()

    def test_new_ssl_outcome_stays_unknown_and_never_retries(self):
        self.session.post.side_effect=requests.exceptions.SSLError('mock SSL')
        client=self.client();result=client.classify('1',fixtures.TITLE,fixtures.BODY)
        self.assertEqual(result['ai_status'],'request_outcome_unknown')
        self.assertGreater(client.cost_breakdown()['unreported_usage_reserved_usd'],0)
        self.client().classify('1',fixtures.TITLE,fixtures.BODY)
        self.assertEqual(self.session.post.call_count,1)

    def test_old_checkpoint_and_old_cache_not_misrepresented_as_new(self):
        old=Step3FClassifier(self.config,fixtures.POLICY_PATH.read_text(encoding='utf-8'),self.temp/'cache',
            self.temp/'old-checkpoint.json',1,execute=True,session=self.session)
        old.classify('1',fixtures.TITLE,fixtures.BODY)
        new=self.client(execute=False)
        self.assertNotEqual(old.cache_key(fixtures.TITLE,fixtures.BODY),new.cache_key(fixtures.TITLE,fixtures.BODY))
        self.assertIsNone(new.cached(fixtures.TITLE,fixtures.BODY))
        with self.assertRaisesRegex(ClassificationError,'Historical checkpoint is incompatible'):
            self.client(checkpoint_path=self.temp/'old-checkpoint.json')

    def test_enum_api_limit_fails_before_any_request_without_source_truncation(self):
        with self.assertRaisesRegex(ClassificationError,'enum exceeds API limit'):
            self.client(execute=False).payload('Title','x'*240000)
        self.session.post.assert_not_called()

    def reserve_fixture(self):
        frame=pd.DataFrame([dict(advertisement_id='human'+str(i),job_title='Unseen '+str(i),description_text='Original text '+str(i),
            human_relevance='',human_primary_category='',human_review_notes='',human_verified='',reviewer='') for i in range(2)])
        source=self.temp/'reserve.csv';frame.to_csv(source,index=False,encoding='utf-8-sig')
        manifest=self.temp/'reserve.json';atomic_json(manifest,dict(reserved_ids=list(frame.advertisement_id),
            content_keys=[source_hash(r.job_title,r.description_text) for r in frame.itertuples()]))
        return frame,source,manifest

    def test_blind_human_sheet_has_empty_labels_and_preserves_full_sources(self):
        frame,source,manifest=self.reserve_fixture();before=sha256(source)
        out=prepare_human_reserve(source,manifest,self.frame,self.temp)
        self.assertEqual(len(out),2)
        self.assertTrue(out.human_verified.eq('').all())
        self.assertTrue(out.human_relevance.eq('').all())
        self.assertTrue(out.human_role_scope.eq('').all())
        self.assertEqual(out.description_text.tolist(),frame.description_text.tolist())
        self.assertNotIn('ai_relevance',out)
        self.assertEqual(sha256(source),before)

    def test_reserve_leakage_and_prediction_contamination_rejected(self):
        frame,source,manifest=self.reserve_fixture()
        with self.assertRaisesRegex(ValueError,'leakage'):prepare_human_reserve(source,manifest,frame,self.temp)
        frame['ai_relevance']='relevant';frame.to_csv(source,index=False,encoding='utf-8-sig')
        with self.assertRaisesRegex(ValueError,'prediction labels'):prepare_human_reserve(source,manifest,self.frame,self.temp)


if __name__=='__main__':unittest.main()
