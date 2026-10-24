"""Saved-result analysis tests; all source API responses are mocks."""
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock,patch

import pandas as pd

import tests.test_step3e as fixtures
from tests.test_step3d import response,prediction
from src.analyze_step3e import analyze,evidence_root_cause
from src.llm_classifier import ClassificationError,evidence_match


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.Step3ETests('test_new_prompt_identity_dry_run_has_zero_api_calls')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.temp=self.fixture.temp

    def completed_run(self):
        session=Mock();session.post.side_effect=[response(),response(prediction(ai_evidence='invented quote')),
            response(prediction(ai_confidence='low')),
            response(prediction(ai_relevance='uncertain',ai_primary_category=None,ai_secondary_categories=['Data Engineer'])),
            response(),response()]
        return self.fixture.runner(execute=True,session=session)[0]

    def test_analysis_reconciles_coverage_costs_and_keeps_failures_unresolved(self):
        source=self.completed_run();before={p:p.read_bytes() for p in source.iterdir() if p.is_file()}
        with patch('requests.sessions.Session.request',side_effect=AssertionError('Network forbidden')):
            rows,failures,summary=analyze(source,self.temp/'analysis',expected_count=6,cache_dir=self.temp/'cache')
        self.assertEqual((summary['total_ids'],summary['accepted'],summary['unresolved'],summary['api_attempts']),(6,4,2,6))
        self.assertAlmostEqual(summary['technical_acceptance_rate'],4/6)
        self.assertEqual(summary['accepted_low_confidence'],1)
        self.assertEqual(summary['reserved_unreported_usage_usd'],0)
        self.assertGreater(summary['recorded_usage_estimated_cost_usd'],0)
        self.assertGreater(summary['rejected_response_usage_usd'],0)
        self.assertFalse(summary['accuracy_claimed'])
        for row in rows:
            if row['ai_validation_status']=='unresolved':
                self.assertIsNone(row['ai_relevance']);self.assertIsNone(row['ai_reasoning'])
                self.assertTrue(row['rejected_ai_reasoning'])
                self.assertTrue(row['ai_review_required'])
        for path,data in before.items():self.assertEqual(path.read_bytes(),data)

    def test_review_retains_every_ad_and_blank_human_fields_with_priorities(self):
        source=self.completed_run();output=self.temp/'analysis'
        rows,failures,summary=analyze(source,output,expected_count=6,cache_dir=self.temp/'cache')
        review=pd.read_csv(output/'prioritized_human_review.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
        self.assertEqual(review.advertisement_id.nunique(),6)
        for field in ['human_relevance','human_primary_category','human_review_notes','human_verified','reviewer']:
            self.assertTrue(review[field].eq('').all())
        self.assertTrue(review.loc[review.ai_validation_status.eq('unresolved'),'review_priority'].eq('1').all())
        self.assertTrue(review.loc[review.ai_confidence.eq('low'),'review_priority'].eq('1').all())

    def test_formatting_analysis_does_not_change_evidence_acceptance(self):
        self.assertEqual(evidence_root_cause('id','Develop data models','Title','develop data models'),'capitalization_only')
        self.assertEqual(evidence_root_cause('id','develop, data models','Title','develop data models'),'punctuation_only')
        for quote in ['Develop data models','develop, data models']:
            with self.assertRaises(ClassificationError):evidence_match(quote,'Title','develop data models',True)
        self.assertEqual(evidence_root_cause('id','develop ... models','Title','develop data models'),'omitted_or_joined_passages')
        self.assertEqual(evidence_root_cause('id','develop AI models','Title','develop data models'),'wording_spelling_or_grammar')

    def test_category_consistency_failure_and_independent_evidence_check(self):
        source=self.completed_run()
        _,failures,summary=analyze(source,self.temp/'analysis',expected_count=6,cache_dir=self.temp/'cache')
        category=next(f for f in failures if f['root_cause']=='uncertain_with_secondary_categories')
        self.assertEqual(category['ai_relevance'],'uncertain')
        self.assertEqual(category['independent_evidence_check'],'exact')
        self.assertEqual(summary['additional_evidence_failures_among_category_rejections'],0)

    def test_refuses_overwrite_incomplete_results_and_mismatched_exports(self):
        source=self.completed_run()
        with self.assertRaises(ValueError):analyze(source,source,expected_count=6,cache_dir=self.temp/'cache')
        with self.assertRaisesRegex(ValueError,'complete result coverage'):
            analyze(source,self.temp/'wrong-size',expected_count=200,cache_dir=self.temp/'cache')
        path=source/'pilot_results.jsonl';rows=[json.loads(l) for l in path.read_text().splitlines()]
        rows[0]['ai_reasoning']='tampered'
        path.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'exported prediction disagree'):
            analyze(source,self.temp/'analysis',expected_count=6,cache_dir=self.temp/'cache')


if __name__=='__main__':unittest.main()
