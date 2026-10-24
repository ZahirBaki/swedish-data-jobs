"""Offline targeted-pilot coverage, holdout separation and post-run audit tests."""
import json
import os
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from src.analyze_step3f_span_pilot import analyze
from src.prepare_step3f_span_pilot import (SELECTION, select_pilot, evaluation_reserve,
    coverage_audit, formatting_audit, content_key)
from src.run_step3f import CODE_PATHS, run
from src.step3f_classifier import (Step3FClassifier, POLICY_PATH, PROMPT_PATH, SCHEMA,
    POLICY_VERSION, PROMPT_VERSION, VALIDATION_VERSION, validate_span_prediction)
from src.llm_classifier import Config, stable_hash, ClassificationError
from src.recover_large_pilot import resilient_atomic_json as atomic_json
from src.validate_classification import ROOT, sha256, read_csv
from tests.test_step3d import response
from tests.test_step3f import prediction


class SpanPilotTests(unittest.TestCase):
    def setUp(self):
        self.temp=Path(__file__).parent/('.test-span-pilot-'+uuid.uuid4().hex);self.temp.mkdir()
        self.addCleanup(lambda:shutil.rmtree(self.temp))
        self.config=Config('mock-model',.4,1.6,api_key_env='SPAN_PILOT_FAKE_KEY',max_retries=0,min_interval=0)
        environment=patch.dict(os.environ,{'SPAN_PILOT_FAKE_KEY':'fake-never-sent'})
        environment.start();self.addCleanup(environment.stop)
        self.rows=[];self.checkpoint={'ledger':[]}
        for i,(identifier,group,reason) in enumerate(SELECTION):
            self.rows.append(dict(advertisement_id=identifier,job_title='Roll '+str(i),
                description_text='Förbättra data.\r\nNästa rad – åäö!\u2028',
                relevance=['relevant','irrelevant','uncertain'][i%3],primary_category='',
                previous_step3e_ai_status='invalid_model_response' if i<6 else '',previous_step3d_ai_status=''))
            if i<6:self.checkpoint['ledger'].append(dict(advertisement_id=identifier,status='invalid_model_response',
                model_response_diagnostic=dict(validation_error='Evidence is not a verbatim excerpt of title/description')))
        self.frame=pd.DataFrame(self.rows)
        self.prepared=self.temp/'prepared';self.prepared.mkdir()
        self.frame.to_csv(self.prepared/'prepared_jobs.csv',index=False,encoding='utf-8-sig')
        (self.prepared/'policy.md').write_bytes(POLICY_PATH.read_bytes())
        (self.prepared/'prompt.txt').write_bytes(PROMPT_PATH.read_bytes())
        atomic_json(self.prepared/'schema.json',SCHEMA)
        self.manifest=dict(stage='step3f',purpose='targeted_source_span_technical_pilot_not_evaluation',total_advertisements=20,
            policy_version=POLICY_VERSION,prompt_version=PROMPT_VERSION,validation_version=VALIDATION_VERSION,
            policy_sha256=sha256(POLICY_PATH),prompt_sha256=sha256(PROMPT_PATH),schema_sha256=stable_hash(SCHEMA),
            prepared_sha256=sha256(self.prepared/'prepared_jobs.csv'),
            code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in CODE_PATHS})
        atomic_json(self.prepared/'manifest.json',self.manifest)
        self.session=Mock();self.session.post.return_value=response(prediction())

    def runner(self,**kwargs):
        return run(self.prepared,self.temp/'run',self.config,1,cache_dir=self.temp/'cache',session=self.session,**kwargs)

    def test_selection_has_20_ids_six_confirmed_failures_and_full_original_text(self):
        selected=select_pilot(self.frame,self.checkpoint)
        self.assertEqual(selected.advertisement_id.nunique(),20)
        self.assertEqual(selected.pilot_selection_group.eq('previous_evidence_failure').sum(),6)
        pd.testing.assert_frame_equal(selected[self.frame.columns],self.frame)
        self.assertEqual(selected.advertisement_id.tolist(),select_pilot(self.frame,self.checkpoint).advertisement_id.tolist())

    def test_missing_diagnostic_or_nonnew_case_is_rejected(self):
        with self.assertRaises(ValueError):select_pilot(self.frame,{'ledger':[]})
        bad=self.frame.copy();bad.loc[7,'previous_step3d_ai_status']='completed'
        with self.assertRaises(ValueError):select_pilot(bad,self.checkpoint)

    def test_coverage_reconciles_ids_without_dropping_missing_or_copies(self):
        client=Step3FClassifier(self.config,POLICY_PATH.read_text(encoding='utf-8'),self.temp/'cache',self.temp/'cp.json',1)
        rows=self.rows[:2]+[dict(self.rows[0],advertisement_id='copy'),dict(self.rows[0],advertisement_id='missing',description_text='')]
        routes,summary=coverage_audit(pd.DataFrame(rows),client)
        self.assertEqual(summary['total_advertisements'],4)
        self.assertEqual(summary['missing_descriptions'],1)
        self.assertEqual(summary['distinct_nonempty_inputs'],2)
        self.assertEqual(summary['additional_same_content_advertisements'],1)
        self.assertEqual(summary['omitted_advertisements'],0)
        self.assertEqual({r['advertisement_id'] for r in routes},{r['advertisement_id'] for r in rows})

    def test_reserve_is_deterministic_blind_and_disjoint_by_id_and_content(self):
        pilot=select_pilot(self.frame,self.checkpoint)
        fresh=[dict(self.rows[i%20],advertisement_id='fresh'+str(i),job_title='Fresh '+str(i),
            previous_step3e_ai_status='',previous_step3d_ai_status='') for i in range(40)]
        duplicate=dict(self.rows[0],advertisement_id='prior-copy',previous_step3e_ai_status='')
        full=pd.concat([self.frame,pd.DataFrame(fresh+[duplicate])],ignore_index=True)
        blind,reserve=evaluation_reserve(full,pilot,size=12)
        again,reserve2=evaluation_reserve(full.sample(frac=1,random_state=8),pilot,size=12)
        self.assertEqual(reserve,reserve2)
        pd.testing.assert_frame_equal(blind.reset_index(drop=True),again.reset_index(drop=True))
        self.assertFalse(set(blind.advertisement_id)&set(pilot.advertisement_id))
        self.assertFalse(set(reserve['content_keys'])&{content_key(r) for r in self.rows})
        self.assertNotIn('relevance',blind)
        self.assertNotIn('ai_relevance',blind)
        self.assertTrue(blind.human_verified.eq('').all())

    def test_source_span_roundtrip_swedish_linebreaks_and_punctuation(self):
        report=formatting_audit(self.frame)
        self.assertTrue(report['all_offsets_exact'])
        self.assertEqual(report['ads_with_swedish_characters'],20)
        self.assertEqual(report['ads_with_line_breaks'],20)
        self.assertEqual(report['ads_with_punctuation'],20)
        result=validate_span_prediction(prediction(),self.rows[0]['job_title'],self.rows[0]['description_text'])
        self.assertEqual(result['ai_evidence'],[self.rows[0]['description_text']])
        with self.assertRaises(ClassificationError):
            validate_span_prediction(prediction(ai_evidence_span_ids=['Förbättra data']),self.rows[0]['job_title'],self.rows[0]['description_text'])

    def test_20_row_dry_run_is_unpaid_and_under_one_dollar(self):
        report=self.runner()
        self.assertEqual(report['selected_advertisements'],20)
        self.assertLess(report['conservative_cost_usd'],1)
        self.session.post.assert_not_called()

    def test_runner_keeps_every_id_including_same_content_and_missing_description(self):
        frame=self.frame.copy()
        frame.loc[1,['job_title','description_text']]=frame.loc[0,['job_title','description_text']].values
        frame.loc[2,'description_text']=''
        path=self.prepared/'prepared_jobs.csv'
        frame.to_csv(path,index=False,encoding='utf-8-sig')
        self.manifest['prepared_sha256']=sha256(path)
        atomic_json(self.prepared/'manifest.json',self.manifest)
        self.runner(execute=True)
        result=read_csv(self.temp/'run/classification_results.csv')
        self.assertEqual(set(result.advertisement_id),set(frame.advertisement_id))
        self.assertEqual(result.advertisement_id.nunique(),20)
        self.assertEqual(result.ai_status.value_counts().to_dict(),{'completed':18,'cache_hit':1,'missing_description':1})
        self.assertEqual(self.session.post.call_count,18)
        missing=result.loc[result.advertisement_id.eq(frame.iloc[2].advertisement_id)].iloc[0]
        self.assertEqual(missing.classification_status,'Uncertain')
        self.assertEqual(sha256(path),self.manifest['prepared_sha256'])

    def test_after_pilot_analyzer_checks_all_spans_disagreements_and_separate_costs(self):
        self.runner(execute=True)
        before=sha256(self.temp/'run/checkpoint.json')
        report=analyze(self.prepared,self.temp/'run',self.temp/'analysis')
        self.assertEqual(report['accepted_and_source_span_valid'],20)
        self.assertEqual(report['rule_ai_relevance_disagreements'],13)
        self.assertFalse(report['classification_accuracy_calculated'])
        self.assertGreater(report['provider_reported_usage_estimated_cost_usd'],0)
        self.assertEqual(report['unreported_usage_reserved_usd'],0)
        self.assertEqual(before,sha256(self.temp/'run/checkpoint.json'))
        review=read_csv(self.temp/'analysis/pilot_technical_review.csv')
        self.assertEqual(len(review),20)
        self.assertTrue(review.human_verified.eq('').all())

    def test_analyzer_detects_corrupted_saved_evidence_without_accepting_replacement(self):
        self.runner(execute=True)
        path=self.temp/'run/checkpoint.json';state=json.loads(path.read_text(encoding='utf-8'))
        state['results'][self.rows[0]['advertisement_id']]['ai_evidence']=['invented']
        atomic_json(path,state)
        report=analyze(self.prepared,self.temp/'run',self.temp/'analysis')
        self.assertEqual(report['accepted_and_source_span_valid'],19)
        self.assertEqual(report['failed_or_invalid_saved_results'],1)

    def test_analyzer_refuses_to_invent_acceptance_from_dry_run(self):
        self.runner()
        with self.assertRaisesRegex(ValueError,'No executed checkpoint'):
            analyze(self.prepared,self.temp/'run',self.temp/'analysis')


if __name__=='__main__':unittest.main()
