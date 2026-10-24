"""Precision denominators, blind labels and hard-capped reliability runner tests."""
import json
import os
import shutil
import unittest
import uuid
from dataclasses import replace,asdict
from pathlib import Path
from unittest.mock import Mock,patch

import pandas as pd
import requests

from src.precision_inclusion import InclusionConfig,inclusion_decision
from src.evaluate_precision import evaluate,validate_human_labels
from src.run_precision_reliability import run,code_hashes,ALLOWED_IDS
from src.step3f_classifier import span_match
from src.step3f_reliable_classifier import source_hash,POLICY_PATH,PROMPT_PATH
from src.validate_classification import ROOT,sha256
from src.recover_large_pilot import resilient_atomic_json as atomic_json
from src.llm_classifier import Config,BudgetExceeded
from tests.test_step3f import prediction,BODY
from tests.test_step3d import response


def row(identifier='1',**changes):
    value=dict(advertisement_id=identifier,job_title='Data Engineer '+identifier,description_text=BODY,
        **prediction(),ai_status='completed',ai_prompt_version='step3f-1.1',ai_human_verified=False,
        duties_assessment='clear',category_assessment='clear',duties_assessor='inclusion-auditor',
        inclusion_audit_notes='Checked full responsibilities and primary role.',inclusion_review_resolved='')
    value.update(changes)
    spans=span_match(value['ai_evidence_span_ids'],value['job_title'],value['description_text'])['spans']
    value.update(ai_evidence=[s['text'] for s in spans],ai_evidence_spans=spans)
    return value


class PrecisionGateTests(unittest.TestCase):
    def test_only_clear_audited_duties_included_by_default(self):
        self.assertEqual(inclusion_decision(row())['inclusion_status'],'included')
        out=inclusion_decision(row(duties_assessment=''))
        self.assertEqual(out['inclusion_status'],'abstained');self.assertIn('duties_not_confirmed',out['inclusion_reasons'])

    def test_incidental_mentions_excluded_but_insufficient_evidence_abstains(self):
        self.assertEqual(inclusion_decision(row(duties_assessment='incidental_only'))['inclusion_status'],'excluded')
        self.assertEqual(inclusion_decision(row(duties_assessment='insufficient'))['inclusion_status'],'abstained')
        self.assertEqual(inclusion_decision(row(category_assessment='insufficient'))['inclusion_status'],'abstained')

    def test_technical_failure_is_never_semantic_uncertainty(self):
        self.assertEqual(inclusion_decision(row(ai_status='invalid_model_response'))['inclusion_status'],'technical_failure')
        self.assertEqual(inclusion_decision(row(ai_relevance='uncertain',ai_primary_category=None,
            ai_scope='unresolved',ai_evidence_span_ids=[]))['inclusion_status'],'abstained')

    def test_scope_and_opportunity_filters_do_not_declare_irrelevance(self):
        research=row(ai_primary_category='AI / ML Research',ai_scope='academic_ai_ml_research',inclusion_review_resolved='true')
        self.assertEqual(inclusion_decision(research)['inclusion_status'],'out_of_scope')
        config=replace(InclusionConfig(),scopes=('academic_ai_ml_research',))
        self.assertEqual(inclusion_decision(research,config)['inclusion_status'],'included')
        self.assertEqual(inclusion_decision(row(ai_opportunity_types=['thesis']))['inclusion_status'],'out_of_scope')

    def test_confidence_threshold_changes_are_explicit_not_probabilities(self):
        self.assertEqual(inclusion_decision(row(ai_confidence='medium'))['inclusion_status'],'abstained')
        self.assertEqual(inclusion_decision(row(ai_confidence='medium',inclusion_review_resolved='true'),
            replace(InclusionConfig(),confidence_levels=('high','medium')))['inclusion_status'],'included')

    def test_review_resolution_requires_attributed_audit(self):
        self.assertEqual(inclusion_decision(row(ai_review_required=True))['inclusion_status'],'abstained')
        self.assertEqual(inclusion_decision(row(ai_review_required=True,inclusion_review_resolved='true'))['inclusion_status'],'included')
        self.assertEqual(inclusion_decision(row(ai_review_required=True,inclusion_review_resolved='true',duties_assessor=''))['inclusion_status'],'abstained')

    def test_corrupt_offsets_or_invented_evidence_cannot_pass_gate(self):
        value=row();value['ai_evidence_spans'][0]['text']='invented'
        self.assertEqual(inclusion_decision(value)['inclusion_status'],'technical_failure')


class PrecisionEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.predictions=pd.DataFrame([row(str(i)) for i in range(6)])
        self.predictions.loc[2,'ai_confidence']='medium'
        self.predictions.loc[2,'inclusion_review_resolved']='true'
        self.predictions.loc[3,'ai_status']='invalid_model_response'
        uncertain=row('5',ai_relevance='uncertain',ai_primary_category=None,ai_scope='unresolved',ai_evidence_span_ids=[])
        self.predictions=pd.concat([self.predictions.iloc[:5],pd.DataFrame([uncertain])],ignore_index=True)
        self.labels=pd.DataFrame([dict(advertisement_id=str(i),job_title='Data Engineer '+str(i),description_text=BODY,
            human_relevance=['relevant','irrelevant','relevant','irrelevant','irrelevant','uncertain'][i],
            human_primary_category=['Data Engineer','','ML Engineer','','',''][i],
            human_verified='false' if i==4 else 'true',reviewer='independent-gold-reviewer',human_role_scope='') for i in range(6)])
        self.reserve=dict(reserved_ids=list(self.labels.advertisement_id),
            content_keys=[source_hash(r.job_title,r.description_text) for r in self.labels.itertuples()],
            source_by_id={r.advertisement_id:source_hash(r.job_title,r.description_text) for r in self.labels.itertuples()})

    def test_verified_precision_denominators_exclude_unverified_and_uncertain_truth(self):
        result=evaluate(self.predictions,self.labels,self.reserve);s=result['summary']
        self.assertEqual(s['relevance_precision'],.5);self.assertEqual(s['relevance_precision_denominator'],2)
        self.assertEqual(s['category_precision'],.5);self.assertEqual(s['category_precision_denominator'],2)
        self.assertEqual(s['included_without_resolved_truth'],1)
        self.assertEqual(s['inclusion_coverage'],.5)
        self.assertEqual(s['technical_failure_rate'],1/6)
        self.assertFalse(s['confidence_is_calibrated'])

    def test_category_precision_counts_included_irrelevant_and_wrong_category(self):
        config=replace(InclusionConfig(),confidence_levels=('high','medium'))
        result=evaluate(self.predictions,self.labels,self.reserve,config)
        self.assertAlmostEqual(result['summary']['relevance_precision'],2/3)
        self.assertAlmostEqual(result['summary']['category_precision'],1/3)
        self.assertEqual(sum(r['relevance_false_positive'] for r in result['rows']),1)
        self.assertEqual(sum(r['category_false_positive'] for r in result['rows']),2)

    def test_no_verified_labels_yields_no_precision_claim(self):
        self.labels['human_verified']=''
        s=evaluate(self.predictions,self.labels,self.reserve)['summary']
        self.assertIsNone(s['relevance_precision']);self.assertIsNone(s['category_precision'])
        self.assertEqual(s['verified_human_labels'],0)

    def test_ai_assisted_labels_rejected_even_when_flagged_verified(self):
        self.labels['assistant_suggested_relevance']='relevant'
        with self.assertRaisesRegex(ValueError,'AI-assisted'):evaluate(self.predictions,self.labels,self.reserve)

    def test_same_auditor_and_truth_reviewer_rejected(self):
        self.labels.loc[0,'reviewer']='inclusion-auditor'
        with self.assertRaisesRegex(ValueError,'must differ'):evaluate(self.predictions,self.labels,self.reserve)

    def test_source_swaps_duplicate_ids_and_missing_reviewers_rejected(self):
        bad=self.labels.copy();bad.loc[0,'job_title']=bad.loc[1,'job_title'];bad.loc[1,'job_title']=self.labels.loc[0,'job_title']
        with self.assertRaises(ValueError):validate_human_labels(bad,self.reserve)
        bad=self.labels.copy();bad.loc[0,'reviewer']=''
        with self.assertRaises(ValueError):validate_human_labels(bad,self.reserve)
        with self.assertRaises(ValueError):evaluate(pd.concat([self.predictions,self.predictions]),self.labels,self.reserve)

    def test_false_positive_cause_is_not_invented(self):
        result=evaluate(self.predictions,self.labels,self.reserve)
        fp=next(r for r in result['rows'] if r['relevance_false_positive'])
        self.assertEqual(fp['human_false_positive_cause'],'')
        self.assertEqual(fp['cause_status'],'requires_human_cause_review')
        self.assertEqual(result['summary']['false_positive_cause_counts'],{'requires_human_cause_review':1})


class ReliabilityRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=Path(__file__).parent/('.test-precision-'+uuid.uuid4().hex);self.temp.mkdir()
        self.addCleanup(lambda:shutil.rmtree(self.temp))
        env=patch.dict(os.environ,{'PRECISION_FAKE_KEY':'fake-no-network'});env.start();self.addCleanup(env.stop)
        self.prepared=self.temp/'prepared';self.prepared.mkdir()
        frame=pd.DataFrame([row(identifier) for identifier in sorted(ALLOWED_IDS)])
        # Source-only input; existing predictions do not enter the API payload.
        frame=frame[['advertisement_id','job_title','description_text']]
        frame.to_csv(self.prepared/'reliability_cases.csv',index=False,encoding='utf-8-sig')
        history=self.temp/'history.json';atomic_json(history,dict(ledger=[],results={}))
        atomic_json(self.prepared/'unknown_outcome_guard.json',dict(blocked_sources=[dict(source_sha256='ssl-source')]))
        policy=ROOT/'docs/precision_first_inclusion_policy_v1.md'
        for source,name in [(POLICY_PATH,'model_policy.md'),(PROMPT_PATH,'model_prompt.txt'),(policy,'precision_first_policy.md')]:
            shutil.copyfile(source,self.prepared/name)
        manifest=dict(stage='precision-reliability-three',input_sha256=sha256(self.prepared/'reliability_cases.csv'),
            guard_sha256=sha256(self.prepared/'unknown_outcome_guard.json'),code_hashes=code_hashes(),
            policy_sha256=sha256(POLICY_PATH),prompt_sha256=sha256(PROMPT_PATH),precision_policy_sha256=sha256(policy),
            history_checkpoint=str(history),history_checkpoint_sha256=sha256(history),history_accounted_cost_usd=.0430352,
            config=asdict(Config('mock-model',.4,1.6,max_retries=0,min_interval=0,api_key_env='PRECISION_FAKE_KEY')))
        atomic_json(self.prepared/'pilot_manifest.json',manifest)
        self.session=Mock();self.session.post.return_value=response(prediction())

    def runner(self,**changes):
        args=dict(prepared=self.prepared,output=self.temp/'run',session=self.session,cache_dir=self.temp/'cache')
        args.update(changes);return run(**args)

    def test_dry_run_exactly_three_and_historical_hold_in_budget(self):
        r=self.runner()
        self.assertEqual(r['selected'],3)
        self.assertAlmostEqual(r['remaining_budget_usd'],.9569648)
        self.session.post.assert_not_called()
        with self.assertRaises(ValueError):self.runner(output=self.temp/'invalid',max_ads=4)

    def test_live_mock_resume_does_not_repay_completed_results(self):
        self.runner(execute=True);self.runner(execute=True,resume=True)
        self.assertEqual(self.session.post.call_count,3)
        state=json.loads((self.temp/'run/checkpoint.json').read_text(encoding='utf-8'))
        self.assertEqual(len(state['ledger']),3)

    def test_failures_never_automatically_retried_and_ssl_stays_unknown(self):
        self.session.post.side_effect=requests.exceptions.SSLError('mock SSL')
        self.runner(execute=True)
        self.session.post.side_effect=None
        self.runner(execute=True,resume=True)
        self.assertEqual(self.session.post.call_count,3) # one unknown + two untouched, never replay first
        state=json.loads((self.temp/'run/checkpoint.json').read_text(encoding='utf-8'))
        self.assertEqual(state['ledger'][0]['status'],'request_outcome_unknown')
        self.assertEqual(state['ledger'][0]['usage_basis'],'conservative_reservation')

    def test_budget_stops_before_post_with_history_included(self):
        self.runner(execute=True,budget_usd=.0430353)
        self.session.post.assert_not_called()
        with self.assertRaises(BudgetExceeded):self.runner(output=self.temp/'under',budget_usd=.01)

    def test_unknown_source_and_changed_history_block_preflight(self):
        path=self.prepared/'reliability_cases.csv'
        frame=pd.read_csv(path,dtype=str);frame.loc[0,'advertisement_id']='31505124'
        frame.to_csv(path,index=False)
        with self.assertRaises(ValueError):self.runner(execute=True)
        self.session.post.assert_not_called()


if __name__=='__main__':unittest.main()
