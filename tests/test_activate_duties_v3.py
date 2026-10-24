"""Approved v3 gates, publication provenance, historical preservation and UI."""
import copy
import json
import unittest
from unittest.mock import patch

import pandas as pd

from src.activate_duties_v3 import (evaluate,approved_inputs,DEFAULT_OUTPUT,POLICY_VERSION)
from src.dashboard_data import (load_dataset,latest_dataset,coverage,filter_data,skill_counts)
from src.step6 import BASE
from src.validate_classification import ROOT,sha256
from tests.test_preview_duties_v3 import row


class ActiveGateTests(unittest.TestCase):
    def test_passes_only_approved_and_recomputed_jobs(self):
        original=row();before=copy.deepcopy(original)
        receipts,included=evaluate({'test':original},{'test'})
        self.assertEqual(len(included),1)
        self.assertEqual(receipts[0]['inclusion_policy_version'],POLICY_VERSION)
        self.assertEqual(receipts[0]['inclusion_method'],'automated_deterministic')
        self.assertFalse(receipts[0]['human_reviewed'])
        self.assertEqual(original,before)
        self.assertNotIn('duties_assessment',original)

    def test_changed_gate_does_not_force_expected_count(self):
        original=row(ai_review_required=True)
        receipts,included=evaluate({'test':original},{'test'})
        self.assertEqual(len(included),0)
        self.assertEqual(receipts[0]['inclusion_status'],'abstention')
        self.assertFalse(receipts[0]['approved'])

    def test_unapproved_new_candidate_is_not_silently_activated(self):
        receipts,included=evaluate({'test':row()},set())
        self.assertEqual(len(included),0)
        self.assertEqual(receipts[0]['inclusion_status'],'not_in_approved_preview')

    def test_unknown_outcome_does_not_invoke_network(self):
        r=row();r['ai_status']='request_outcome_unknown'
        with patch('requests.sessions.Session.request',side_effect=AssertionError('No network')):
            receipts,included=evaluate({'test':r},{'test'})
        self.assertEqual(len(included),0)
        self.assertEqual(receipts[0]['inclusion_status'],'technical_failure')


@unittest.skipUnless((DEFAULT_OUTPUT/'manifest.json').exists(),'V3 publication not generated')
class ActivePublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs,cls.facts=load_dataset(DEFAULT_OUTPUT)
        cls.base,cls.base_facts=load_dataset(BASE)
        cls.report=json.loads((DEFAULT_OUTPUT/'step6_summary.json').read_text(encoding='utf-8'))
        cls.receipts=[json.loads(x) for x in (DEFAULT_OUTPUT/'inclusion_audit_receipts.jsonl').read_text(encoding='utf-8').splitlines()]

    def test_all_passing_receipts_and_unique_ids(self):
        included=[r for r in self.receipts if r['approved']]
        new=self.jobs[~self.jobs.advertisement_id.isin(self.base.advertisement_id)]
        self.assertEqual(set(new.advertisement_id),{r['advertisement_id'] for r in included})
        self.assertEqual(len(self.jobs),len(self.base)+len(new))
        self.assertFalse(self.jobs.advertisement_id.duplicated().any())
        self.assertEqual(int(self.jobs.requirements_data_usable.sum()),self.report['usable_requirements_jobs'])

    def test_original_108_projection_and_extraction_facts_unchanged(self):
        old=self.jobs[self.jobs.advertisement_id.isin(self.base.advertisement_id)][self.base.columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(old,self.base,check_dtype=False)
        old_facts=self.facts[self.facts.advertisement_id.isin(self.base.advertisement_id)].reset_index(drop=True)
        pd.testing.assert_frame_equal(old_facts,self.base_facts)

    def test_all_job_inclusion_provenance_and_original_classifications(self):
        self.assertTrue(self.jobs.inclusion_method.notna().all())
        self.assertTrue(self.jobs.inclusion_policy_version.notna().all())
        self.assertFalse(self.jobs.inclusion_human_reviewed.any())
        checkpoint=json.loads((ROOT/'data/processed/step6-live500-20261024/checkpoint.json').read_text(encoding='utf-8'))
        saved={r['advertisement_id']:r for r in [json.loads(x) for x in (DEFAULT_OUTPUT/'jobs_requirements.jsonl').read_text(encoding='utf-8').splitlines()]}
        for receipt in self.receipts:
            if not receipt['approved']:continue
            job=saved[receipt['advertisement_id']]
            self.assertEqual(job['inclusion_policy_version'],'precision-first-v3')
            for field,value in checkpoint['results'][receipt['advertisement_id']].items():
                self.assertEqual(job[field],value,(receipt['advertisement_id'],field))

    def test_separate_operational_and_reporting_groups(self):
        self.assertEqual(len(self.receipts),500)
        self.assertEqual(sum(self.report['expansion_statuses'].values()),500)
        for group in ['abstention','academic_research','adjacent','opportunity','governance','technical_failure','uncertain']:
            self.assertTrue((DEFAULT_OUTPUT/('group_'+group+'.csv')).exists())
        approved={r['advertisement_id'] for r in self.receipts if r['approved']}
        for receipt in self.receipts:
            if receipt['inclusion_status']!='included':self.assertNotIn(receipt['advertisement_id'],approved)

    def test_fresh_recomputation_matches_saved_receipts(self):
        from src.preview_duties_v3 import PREPARED,RUN
        with patch('requests.sessions.Session.request',side_effect=AssertionError('No network')):
            _,originals,ids,_=approved_inputs(PREPARED,RUN)
            receipts,included=evaluate(originals,ids)
        self.assertEqual(receipts,self.receipts)
        self.assertEqual(len(included),self.report['newly_extracted_jobs'])

    def test_filtered_denominators_and_evidence(self):
        self.assertTrue(coverage(self.jobs,self.facts).denominator_jobs.eq(len(self.jobs)).all())
        new,facts=filter_data(self.jobs,self.facts,inclusion_method=['automated_deterministic'])
        self.assertEqual(len(new),self.report['new_analyzed'])
        self.assertTrue(coverage(new,facts).denominator_jobs.eq(len(new)).all())
        self.assertTrue(set(facts.advertisement_id)<=set(new.advertisement_id))
        self.assertTrue(skill_counts(new,facts).denominator_jobs.eq(len(new)).all())

    def test_all_historical_files_hashes_preserved(self):
        manifest=json.loads((DEFAULT_OUTPUT/'preservation_manifest.json').read_text(encoding='utf-8'))
        for path,digest in manifest['files'].items():self.assertEqual(sha256(ROOT/path),digest,path)

    def test_dashboard_refresh_methods_filters_and_evidence(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict('os.environ',{'JOBS_DASHBOARD_DATA':str(DEFAULT_OUTPUT)}):
            app=AppTest.from_file(str(ROOT/'dashboard/app.py')).run(timeout=45)
            self.assertFalse(app.exception,str(app.exception))
            self.assertEqual(int(app.metric[0].value),len(self.jobs))
            self.assertEqual(int(app.metric[1].value),self.report['usable_requirements_jobs'])
            self.assertTrue(any('precision-first-v3' in item.value for item in app.info))
            app.multiselect(key='inclusion_method').set_value(['automated_deterministic']).run(timeout=45)
            self.assertFalse(app.exception,str(app.exception))
            self.assertEqual(int(app.metric[0].value),self.report['new_analyzed'])
            self.assertEqual(int(app.metric[1].value),self.report['new_usable_requirements_jobs'])
            app.multiselect(key='region').set_value([]).run(timeout=45)
            self.assertFalse(app.exception,str(app.exception));self.assertTrue(app.warning)
        latest_jobs,_=load_dataset(latest_dataset(ROOT/'data/processed'))
        self.assertTrue(set(self.jobs.advertisement_id).issubset(set(latest_jobs.advertisement_id)))

if __name__=='__main__':unittest.main()
