import json
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd

from src.dashboard_data import (count_ads, coverage, filter_data, language_counts,
                                load_dataset, skill_counts, validate)
from src.validate_classification import ROOT, sha256

DATA = ROOT / 'data/processed/step5-available108-20261023'


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.jobs = pd.DataFrame([
            dict(advertisement_id='a', ai_relevance='relevant', classification_source_valid=True,
                 ai_primary_category='Data Engineer', region='Stockholm', seniority='senior',
                 swedish_requirement='required', english_requirement='not_specified'),
            dict(advertisement_id='b', ai_relevance='relevant', classification_source_valid=True,
                 ai_primary_category='Data Analyst', region='Skane', seniority='unspecified',
                 swedish_requirement='not_specified', english_requirement='explicitly_not_required')])
        self.facts = pd.DataFrame([
            dict(advertisement_id='a', kind='programming_language', value='Python', modality=m)
            for m in ['required', 'required', 'preferred']
        ] + [dict(advertisement_id='b', kind='programming_language', value='Python', modality='mentioned_only')])

    def test_distinct_ids_not_mentions_or_modalities(self):
        row = skill_counts(self.jobs, self.facts).iloc[0]
        self.assertEqual(row.job_count, 1)
        self.assertEqual(row.denominator_jobs, 2)
        self.assertEqual(row.percent, 50)

    def test_mention_is_not_requirement(self):
        row = skill_counts(self.jobs, self.facts, modalities=['mentioned_only']).iloc[0]
        self.assertEqual(row.job_count, 1)
        self.assertEqual(skill_counts(self.jobs, self.facts, modalities=[]).shape[0], 0)

    def test_filtered_denominator_and_facts(self):
        jobs, facts = filter_data(self.jobs, self.facts, region=['Stockholm'])
        self.assertEqual(set(facts.advertisement_id), {'a'})
        self.assertEqual(skill_counts(jobs, facts).iloc[0].percent, 100)
        self.assertEqual(set(self.facts.advertisement_id), {'a', 'b'})

    def test_empty_filter_selects_zero(self):
        jobs, facts = filter_data(self.jobs, self.facts, seniority=[])
        self.assertTrue(jobs.empty)
        self.assertTrue(facts.empty)
        self.assertTrue(count_ads(jobs, 'region', 0).empty)

    def test_unspecified_not_negative_language(self):
        rows = language_counts(self.jobs, 'swedish').set_index('swedish_requirement')
        self.assertEqual(rows.loc['required', 'percent'], 50)
        self.assertEqual(rows.loc['not_specified', 'job_count'], 1)
        self.assertEqual(rows.loc['required', 'denominator_specified_jobs'], 1)
        self.assertNotIn('explicitly_not_required', rows.index)

    def test_duplicate_jobs_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unique'):
            validate(pd.concat([self.jobs, self.jobs]), self.facts)

    def test_orphan_fact_rejected(self):
        facts = self.facts.copy()
        facts.loc[0, 'advertisement_id'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'unknown'):
            validate(self.jobs, facts)

    def test_noneligible_rejected(self):
        for column, value in [('ai_relevance', 'uncertain'), ('classification_source_valid', False)]:
            jobs = self.jobs.copy()
            jobs.loc[0, column] = value
            with self.assertRaises(ValueError): validate(jobs, self.facts)


@unittest.skipUnless(DATA.exists(), 'Step 5 generated dataset not present')
class AvailableDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs, cls.facts = load_dataset(DATA)

    def test_all_108_and_no_reclassification(self):
        self.assertEqual(len(self.jobs), 108)
        registry = pd.read_csv(DATA / 'classification_registry.csv', dtype=str, keep_default_na=False)
        expected = set(registry.loc[registry.step4_disposition.eq('eligible_relevant'), 'advertisement_id'])
        self.assertEqual(set(self.jobs.advertisement_id), expected)
        self.assertFalse(self.jobs.classification_human_verified.any())

    def test_all_evidence_grounded_in_original_unicode(self):
        by_id = self.jobs.set_index('advertisement_id').to_dict('index')
        for fact in self.facts.to_dict('records'):
            source = by_id[fact['advertisement_id']][fact['source_field']]
            for prefix, text_column in [('evidence', 'evidence_text'), ('match', 'matched_text')]:
                self.assertEqual(source[int(float(fact[prefix + '_start'])):int(float(fact[prefix + '_end']))], fact[text_column])

    def test_all_original_source_fields_preserved(self):
        from src.step4_jobs import SOURCE
        source = pd.read_csv(SOURCE, dtype=str, keep_default_na=False, encoding='utf-8-sig').set_index('advertisement_id')
        selected = self.jobs.set_index('advertisement_id')
        for identifier in selected.index:
            for column in source.columns:
                self.assertEqual(selected.loc[identifier, column], source.loc[identifier, column], (identifier, column))

    def test_coverage_and_denominators(self):
        rows = coverage(self.jobs, self.facts)
        self.assertTrue(rows.denominator_jobs.eq(108).all())
        self.assertTrue(rows.job_count.between(0, 108).all())
        self.assertEqual(int(rows.set_index('requirement_type').loc['programming_languages', 'job_count']), 36)

    def test_existing_data_preservation(self):
        manifest = json.loads((DATA / 'preservation_manifest.json').read_text(encoding='utf-8'))
        for filename, digest in manifest['files'].items():
            self.assertEqual(sha256(ROOT / filename), digest, filename)

    @patch.dict('os.environ', {'JOBS_DASHBOARD_DATA': str(DATA)})
    def test_dashboard_renders_and_filters(self):
        from streamlit.testing.v1 import AppTest
        with patch.dict('os.environ', {'JOBS_DASHBOARD_DATA': str(DATA)}):
            app = AppTest.from_file(str(ROOT / 'dashboard/app.py')).run(timeout=45)
        self.assertEqual(len(app.exception), 0, str(app.exception))
        self.assertEqual(app.metric[0].value, '108')
        app.multiselect(key='reporting_scope').set_value(['core_industry']).run(timeout=45)
        self.assertEqual(len(app.exception), 0, str(app.exception))
        self.assertEqual(int(app.metric[0].value), int(self.jobs.reporting_scope.eq('core_industry').sum()))
        app.multiselect(key='region').set_value([]).run(timeout=45)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(len(app.warning))


if __name__ == '__main__': unittest.main()
