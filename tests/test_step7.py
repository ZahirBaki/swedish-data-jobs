"""Offline Step 7 eligibility, immutable reuse, evidence, reconciliation and UI."""
import json
import unittest
from unittest.mock import patch

import pandas as pd
from src.dashboard_data import load_dataset, latest_dataset, filter_data, coverage, count_ads, skill_counts, validate
from src.run_step7 import BASE, DEFAULT_OUTPUT, POLICY, eligible, scope_metadata, verify_evidence, publish
from src.step4_jobs import SOURCE
from src.validate_classification import ROOT, read_csv, sha256


@unittest.skipUnless((DEFAULT_OUTPUT/'manifest.json').exists(), 'Step 7 publication not available')
class Step7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs, cls.facts = load_dataset(DEFAULT_OUTPUT)
        cls.base, cls.base_facts = load_dataset(BASE)
        cls.report=json.loads((DEFAULT_OUTPUT/'step7_summary.json').read_text(encoding='utf-8'))
        cls.predictions=json.loads((DEFAULT_OUTPUT/'classification_provenance.json').read_text(encoding='utf-8'))
        cls.source=read_csv(SOURCE).set_index('advertisement_id',drop=False).to_dict('index')

    def test_every_saved_prediction_accepted_schema_valid_source_grounded_relevant(self):
        for identifier,prediction in self.predictions.items():
            self.assertEqual(eligible(prediction,self.source[identifier]),(True,'accepted_schema_valid_source_grounded_ai_relevant'))
        self.assertEqual(set(self.predictions),set(self.jobs.advertisement_id))
        from src.run_step4 import json_rows
        for row in json_rows(self.jobs):
            self.assertTrue(eligible(row,self.source[row['advertisement_id']])[0])

    def test_uncertain_irrelevant_failure_and_invalid_evidence_rejected(self):
        identifier=next(iter(self.predictions)); original=self.predictions[identifier]
        for changes in [dict(ai_status='request_outcome_unknown'),dict(ai_status='invalid_response'),
                        dict(ai_status='technical_failure'),dict(ai_relevance='uncertain'),dict(ai_relevance='irrelevant'),
                        dict(ai_evidence=['invented evidence'])]:
            self.assertFalse(eligible(dict(original,**changes),self.source[identifier])[0])

    def test_no_confidence_or_duties_gate(self):
        unapproved=self.jobs[~self.jobs.strict_audit_approved]
        self.assertGreater(len(unapproved),0)
        self.assertTrue(unapproved.dashboard_inclusion_policy_version.eq(POLICY).all())
        self.assertTrue(self.jobs.ai_review_required.eq(True).any())
        self.assertTrue(self.jobs.ai_confidence.ne('high').any())
        self.assertEqual(int(self.jobs.strict_audit_approved.sum()),214)

    def test_unique_ids_and_primary_category_total(self):
        self.assertFalse(self.jobs.advertisement_id.duplicated().any())
        self.assertEqual(sum(self.report['categories'].values()),len(self.jobs))
        self.assertEqual(int(count_ads(self.jobs,'primary_category_label',len(self.jobs)).job_count.sum()),len(self.jobs))
        self.assertEqual(self.report['expanded_total'],len(self.jobs))
        self.assertEqual(self.report['expanded_total'],self.report['previous_preserved']+self.report['new_unique_jobs'])

    def test_copies_with_distinct_ids_and_secondary_categories_do_not_inflate(self):
        self.assertGreater(self.jobs.duplicated(['job_title','description_text']).sum(),0)
        self.assertEqual(self.jobs.duplicated(['job_title','description_text']).sum(),self.report['same_content_additional_ids'])
        self.assertGreater(self.jobs.ai_secondary_categories.map(lambda x: len(x or [])).sum(),0)
        self.assertEqual(sum(self.report['categories'].values()),self.jobs.advertisement_id.nunique())

    def test_previous_jobs_and_all_original_columns_preserved(self):
        selected=self.jobs.set_index('advertisement_id').loc[self.base.advertisement_id].reset_index()
        pd.testing.assert_frame_equal(selected[self.base.columns],self.base,check_dtype=False)
        oldfacts=self.facts[self.facts.advertisement_id.isin(self.base.advertisement_id)].reset_index(drop=True)
        pd.testing.assert_frame_equal(oldfacts,self.base_facts)

    def test_original_source_preserved(self):
        for row in self.jobs.to_dict('records'):
            for column,value in self.source[row['advertisement_id']].items():
                self.assertEqual(row[column],value,(row['advertisement_id'],column))

    def test_every_evidence_and_match_offset_validated(self):
        verify_evidence(self.jobs,self.facts)

    def test_scopes_opportunities_and_multi_role_are_filterable(self):
        self.assertEqual(sum(self.report['scopes'].values()),len(self.jobs))
        for scope in self.report['scopes']:
            jobs,facts=filter_data(self.jobs,self.facts,analysis_scope=[scope])
            self.assertEqual(len(jobs),self.report['scopes'][scope])
            self.assertTrue(set(facts.advertisement_id)<=set(jobs.advertisement_id))
        jobs,_=filter_data(self.jobs,self.facts,analysis_opportunity_types=['internship'])
        self.assertGreater(len(jobs),0)
        self.assertTrue(jobs.analysis_opportunity_types.map(lambda x:'internship' in x).all())
        jobs,_=filter_data(self.jobs,self.facts,multi_role_group=['multi_role'])
        self.assertEqual(len(jobs),self.report['multi_role']['multi_role'])
        jobs,facts=filter_data(self.jobs,self.facts,analysis_opportunity_types=[])
        self.assertTrue(jobs.empty and facts.empty)

    def test_scope_reporting_does_not_reclassify_or_hide_special_groups(self):
        for title,category,scope,flags,expected in [
            ('Data Engineer','Data Engineer','core_industry',{},'core_industry'),
            ('Data Engineer internship','Data Engineer','core_industry',{'ai_opportunity_types':['internship']},'opportunities'),
            ('AI researcher','AI / ML Research','academic_ai_ml_research',{},'academic_ai_ml_research'),
            ('Data manager','Data / AI Management','adjacent_borderline',{},'adjacent'),
            ('Governance','Data Governance / Data Quality','data_ai_governance',{},'governance')]:
            self.assertEqual(scope_metadata(dict(job_title=title,ai_primary_category=category,ai_scope=scope,**flags))['analysis_scope'],expected)

    def test_requirement_denominators_follow_selected_ids(self):
        for filters in [dict(analysis_scope=['core_industry']),dict(strict_audit_approved=[True]),dict(primary_category_label=['Data Engineer'])]:
            jobs,facts=filter_data(self.jobs,self.facts,**filters)
            self.assertTrue(coverage(jobs,facts).denominator_jobs.eq(len(jobs)).all())
            counts=skill_counts(jobs,facts)
            self.assertTrue(counts.denominator_jobs.eq(len(jobs)).all())
            self.assertTrue(counts.job_count.le(len(jobs)).all())
        self.assertEqual(self.report['usable_requirements_jobs'],int(self.jobs.requirements_data_usable.sum()))

    def test_unresolved_categories_explicit_without_editing_classification(self):
        unresolved=self.jobs[self.jobs.primary_category_label.eq('Unresolved primary category')]
        self.assertEqual(len(unresolved),self.report['categories']['Unresolved primary category'])
        self.assertTrue(unresolved.ai_primary_category.isna().all())

    def test_coverage_all_5023_no_failures_enter(self):
        cover=read_csv(DEFAULT_OUTPUT/'corpus_coverage.csv')
        self.assertEqual(set(cover.advertisement_id),set(self.source))
        self.assertFalse(cover.advertisement_id.duplicated().any())
        self.assertEqual(set(cover.loc[cover.step7_status.eq('included_ai_relevant'),'advertisement_id']),set(self.jobs.advertisement_id))
        self.assertFalse(cover[cover.advertisement_id.isin(self.jobs.advertisement_id)].classification_relevance.ne('relevant').any())
        self.assertEqual(self.report['paid_api_requests'],0)

    def test_history_and_checkpoints_unchanged(self):
        manifest=json.loads((DEFAULT_OUTPUT/'preservation_manifest.json').read_text(encoding='utf-8'))
        self.assertTrue(manifest['verified_unchanged'])
        for name,digest in manifest['files'].items():
            self.assertEqual(sha256(ROOT/name),digest,name)

    def test_completed_publication_selected_and_overwrite_rejected(self):
        self.assertEqual(latest_dataset(ROOT/'data/processed'),DEFAULT_OUTPUT)
        with self.assertRaises(FileExistsError): publish(DEFAULT_OUTPUT)

    def test_dashboard_defaults_filters_charts_and_evidence(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict('os.environ',{'JOBS_DASHBOARD_DATA':str(DEFAULT_OUTPUT)}), patch('requests.sessions.Session.request',side_effect=AssertionError('Network forbidden')):
            app=AppTest.from_file(str(ROOT/'dashboard/app.py')).run(timeout=60)
            self.assertFalse(app.exception,str(app.exception))
            metrics={m.label:m.value for m in app.metric}
            self.assertEqual(int(metrics['Relevant advertisements in current analysis']),self.report['core_industry_total'])
            self.assertEqual(int(metrics['Jobs with usable requirements evidence']),self.report['core_usable_requirements_jobs'])
            self.assertEqual(app.multiselect(key='analysis_scope').value,['core_industry'])
            self.assertEqual(app.selectbox(key='inclusion_quality').value,'All accepted AI-relevant jobs')
            self.assertTrue(len(app.get('plotly_chart')))
            self.assertTrue(any(e.label=='Full original description' for e in app.expander))
            app.multiselect(key='analysis_scope').set_value(sorted(self.report['scopes'])).run(timeout=60)
            self.assertFalse(app.exception,str(app.exception))
            self.assertEqual(int(app.metric[3].value),len(self.jobs))
            app.selectbox(key='inclusion_quality').set_value('Audit-approved only').run(timeout=60)
            self.assertFalse(app.exception,str(app.exception))
            self.assertEqual(int(app.metric[3].value),self.report['strict_audit_approved'])
            app.selectbox(key='inclusion_quality').set_value('All accepted AI-relevant jobs').run(timeout=60)
            app.multiselect(key='primary_category_label').set_value(['Data Engineer']).run(timeout=60)
            self.assertEqual(int(app.metric[3].value),333)
            app.multiselect(key='analysis_scope').set_value([]).run(timeout=60)
            self.assertFalse(app.exception,str(app.exception))
            self.assertTrue(app.warning)


if __name__=='__main__': unittest.main()
