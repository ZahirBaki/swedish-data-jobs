"""No paid calls: corpus deduplication, unknown guards and cumulative resume budget."""
import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock,patch
from dataclasses import asdict

import pandas as pd
import requests

from src import corpus_workflow as corpus
from src.llm_classifier import Config
from src.step3f_reliable_classifier import Step3FReliableClassifier,POLICY_PATH
from src.validate_classification import ROOT
from tests.test_step3d import response
from tests.test_step3f import TITLE,BODY,prediction


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).parent/('.test-corpus-'+uuid.uuid4().hex);self.root.mkdir()
        self.addCleanup(lambda:shutil.rmtree(self.root))
        self.cache=self.root/'cache';self.cache.mkdir()
        self.source=pd.DataFrame([dict(advertisement_id='a',job_title=TITLE,description_text=BODY),
            dict(advertisement_id='b',job_title=TITLE,description_text=BODY)])
        self.base=pd.DataFrame(columns=['advertisement_id'])
        self.session=Mock();self.session.post.return_value=response(prediction())
        self.conf=Config('mock',.4,1.6,max_retries=0,min_interval=0)
        self.history=patch.object(corpus,'historical',return_value=({},set(),set(),set(),{}))
        self.history.start();self.addCleanup(self.history.stop)
        self.env=patch.dict('os.environ',{'OPENAI_API_KEY':'mock-only'});self.env.start();self.addCleanup(self.env.stop)

    def engine(self,output=None,budget=60):
        return Step3FReliableClassifier(self.conf,POLICY_PATH.read_text(encoding='utf-8'),self.cache,
            (output or self.root/'engine')/'checkpoint.json',budget,execute=False,blocked_source_hashes=set(),session=self.session)

    def test_one_request_unit_for_identical_source(self):
        coverage,predictions,_=corpus.reconcile(self.source,self.engine(),self.base)
        self.assertEqual(list(coverage.request_action),['new_request','shared_source'])
        self.assertEqual(coverage.iloc[1].representative_id,'a')
        self.assertFalse(predictions)
        self.session.post.assert_not_called()

    def test_compatible_cache_reused_for_all_alias_ids(self):
        engine=self.engine();engine.execute=True
        engine.classify('a',TITLE,BODY)
        coverage,predictions,_=corpus.reconcile(self.source,engine,self.base)
        self.assertEqual(set(coverage.request_action),{'reuse_compatible'})
        self.assertEqual(len(predictions),2)
        self.assertTrue(coverage.compatible_cache.all())
        self.assertEqual(self.session.post.call_count,1)

    def test_unknown_blocks_alternate_id_by_content(self):
        with patch.object(corpus,'historical',return_value=({}, {'a'},{'a'},set(),{})):
            coverage,_,_=corpus.reconcile(self.source,self.engine(),self.base)
        self.assertEqual(set(coverage.request_action),{'blocked_unknown_outcome'})
        self.session.post.assert_not_called()

    def test_failed_source_is_not_automatically_retried(self):
        with patch.object(corpus,'historical',return_value=({}, {'a'},set(),set(),{})):
            coverage,_,_=corpus.reconcile(self.source,self.engine(),self.base)
        self.assertEqual(set(coverage.request_action),{'skip_previous_failure_or_attempt'})
        self.session.post.assert_not_called()

    def test_pending_marker_blocks_all_source_aliases(self):
        engine=self.engine();key=engine.cache_key(TITLE,BODY)
        (self.cache/(key+'.pending.json')).write_text('{}')
        coverage,_,_=corpus.reconcile(self.source,engine,self.base)
        self.assertEqual(set(coverage.request_action),{'blocked_unknown_outcome'})

    def test_completed_request_without_valid_result_never_paid_again(self):
        engine=self.engine();key=engine.cache_key(TITLE,BODY)
        with patch.object(corpus,'historical',return_value=({},set(),set(),{key},{})):
            coverage,_,_=corpus.reconcile(self.source,engine,self.base)
        self.assertEqual(set(coverage.request_action),{'skip_previous_failure_or_attempt'})

    def test_duplicate_collected_ids_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            corpus.reconcile(pd.concat([self.source,self.source]),self.engine(),self.base)

    def run_workflow(self,**kwargs):
        prepared=self.root/'prepared';prepared.mkdir(exist_ok=True)
        (prepared/'manifest.json').write_text('{}')
        output=self.root/'run'
        manifest={'config':asdict(self.conf)}
        with patch.object(corpus,'prepared_inputs',return_value=(manifest,self.source,(self.base,pd.DataFrame()))),\
             patch.object(corpus,'CACHE',self.cache),\
             patch.object(corpus,'client',side_effect=lambda out,budget=60,session=None:self.engine(Path(out),budget)):
            return corpus.run(prepared,output,**kwargs)

    def test_dry_run_then_live_resume_only_one_request(self):
        self.run_workflow()
        self.session.post.assert_not_called()
        self.run_workflow(execute=True,resume=True)
        self.run_workflow(execute=True,resume=True)
        self.assertEqual(self.session.post.call_count,1)

    def test_budget_reservation_prevents_call(self):
        report=self.run_workflow(execute=True,budget=.0000001)
        self.assertEqual(report['stop_reason'],'application_budget_exhausted')
        self.session.post.assert_not_called()

    def test_unknown_resume_never_retries(self):
        self.session.post.side_effect=requests.exceptions.Timeout('mock')
        self.run_workflow(execute=True)
        self.run_workflow(execute=True,resume=True)
        self.assertEqual(self.session.post.call_count,1)

    def test_v3_publication_scope_does_not_use_old_preview_allowlist(self):
        from tests.test_preview_duties_v3 import row
        r=row();original,receipt=corpus.assess(r,{k:v for k,v in r.items() if k.startswith('ai_')})
        self.assertEqual(receipt['inclusion_status'],'included')
        self.assertEqual(receipt['inclusion_policy_version'],'precision-first-v3')
        self.assertFalse(receipt['human_reviewed'])

if __name__=='__main__':unittest.main()

OFFLINE=ROOT/'data/processed/corpus-offline-coverage-20261024'


@unittest.skipUnless((OFFLINE/'manifest.json').exists(),'Offline corpus publication not available')
class CorpusPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.dashboard_data import load_dataset
        cls.jobs,cls.facts=load_dataset(OFFLINE)
        cls.base,cls.base_facts=load_dataset(corpus.V3_BASE)
        cls.coverage=pd.read_csv(OFFLINE/'corpus_coverage.csv',dtype=str,keep_default_na=False)
        cls.report=json.loads((OFFLINE/'step6_summary.json').read_text(encoding='utf-8'))

    def test_all_original_ids_have_distinct_processing_status(self):
        from src.validate_classification import read_csv
        source=read_csv(corpus.SOURCE)
        self.assertEqual(len(self.coverage),5023)
        self.assertFalse(self.coverage.advertisement_id.duplicated().any())
        self.assertEqual(set(self.coverage.advertisement_id),set(source.advertisement_id))
        self.assertTrue(self.coverage.final_processing_status.ne('').all())
        self.assertEqual(sum(self.report['expansion_statuses'].values()),5023)

    def test_original_145_and_evidence_unchanged(self):
        old=self.jobs[self.jobs.advertisement_id.isin(self.base.advertisement_id)][self.base.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(old,self.base,check_dtype=False)
        pd.testing.assert_frame_equal(self.facts[self.facts.advertisement_id.isin(self.base.advertisement_id)].reset_index(drop=True),self.base_facts)

    def test_new_jobs_pass_frozen_v3_without_paid_requests(self):
        new=self.jobs[~self.jobs.advertisement_id.isin(self.base.advertisement_id)]
        self.assertEqual(len(new),self.report['new_analyzed'])
        for r in corpus.json_rows(new):
            self.assertEqual(corpus.duties.decision(r)['preview_decision'],'would_qualify_v3')
            self.assertEqual(r['inclusion_policy_version'],'precision-first-v3')
            self.assertFalse(r['inclusion_human_reviewed'])
        self.assertEqual(self.report['publication_api_requests'],0)
        self.assertEqual(self.report['accounted_classification_cost_this_run_usd'],0)

    def test_all_evidence_offsets_are_source_grounded(self):
        by_id=self.jobs.set_index('advertisement_id').to_dict('index')
        for fact in self.facts.to_dict('records'):
            source=by_id[fact['advertisement_id']][fact['source_field']]
            self.assertEqual(source[int(float(fact['evidence_start'])):int(float(fact['evidence_end']))],fact['evidence_text'])

    def test_requirement_denominators_track_filtered_ids(self):
        from src.dashboard_data import coverage,filter_data
        self.assertTrue(coverage(self.jobs,self.facts).denominator_jobs.eq(len(self.jobs)).all())
        selected,evidence=filter_data(self.jobs,self.facts,ai_primary_category=['Data Engineer'])
        self.assertTrue(coverage(selected,evidence).denominator_jobs.eq(len(selected)).all())
        self.assertTrue(set(evidence.advertisement_id)<=set(selected.advertisement_id))

    def test_dashboard_corpus_refresh_and_download(self):
        from src.dashboard_data import latest_dataset,load_dataset
        from streamlit.testing.v1 import AppTest
        with patch.dict('os.environ',{'JOBS_DASHBOARD_DATA':str(OFFLINE)}):
            app=AppTest.from_file(str(ROOT/'dashboard/app.py')).run(timeout=45)
            self.assertFalse(app.exception,str(app.exception))
            self.assertEqual(int(app.metric[0].value),len(self.jobs))
            self.assertEqual(int(app.metric[1].value),self.report['usable_requirements_jobs'])
            self.assertTrue(any('all collected advertisements' in e.label for e in app.expander))
            app.multiselect(key='inclusion_method').set_value(['automated_deterministic']).run(timeout=45)
            self.assertFalse(app.exception,str(app.exception))
            self.assertEqual(int(app.metric[0].value),int(self.jobs.inclusion_method.eq('automated_deterministic').sum()))
        latest,_=load_dataset(latest_dataset(ROOT/'data/processed'))
        self.assertTrue(set(self.jobs.advertisement_id)<=set(latest.advertisement_id))
