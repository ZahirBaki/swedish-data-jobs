"""Offline Step 3 behavioral checks; no APIs or changes to collected data."""
import hashlib
import json
import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

from src.clean_jobs import clean_jobs, latest_collection, normalize_text
from src.classify_jobs import (CATEGORIES, IRRELEVANT, UNCERTAIN, classify_job,
                               evaluate_review, review_sample, run_pipeline)


def frame(rows):
    return pd.DataFrame([dict(advertisement_id=str(i), job_title='Data Engineer',
                              description_text='Original\nÅÄÖ', publication_date='2026-06-01', **row)
                         for i, row in enumerate(rows)])


class Step3Tests(unittest.TestCase):
    def test_missing_values(self):
        raw = frame([{}])
        raw.loc[0, ['job_title', 'description_text', 'advertisement_id']] = None
        cleaned = clean_jobs(raw)
        self.assertTrue(cleaned.loc[0, 'missing_title'])
        self.assertTrue(cleaned.loc[0, 'missing_description'])
        self.assertTrue(cleaned.loc[0, 'missing_id'])
        self.assertEqual(classify_job(None, None)['primary_job_category'], UNCERTAIN)
        self.assertEqual(classify_job('Data Engineer', None)['primary_job_category'], 'Data Engineer')

    def test_english_and_swedish(self):
        examples = {'Data Engineer': 'Data Engineer', 'Dataingenjör': 'Data Engineer',
                    'Dataanalytiker': 'Data Analyst', 'Senior Data Scientist': 'Data Scientist',
                    'Datavetare': 'Data Scientist', 'ML-utvecklare': 'ML Engineer',
                    'Maskininlärningsingenjör': 'ML Engineer', 'AI-ingenjör': 'AI Engineer',
                    'AI developer': 'AI Engineer', 'Analytics Engineer': 'Analytics Engineer',
                    'BI-analytiker': 'BI Developer / BI Analyst', 'Power BI-utvecklare': 'BI Developer / BI Analyst',
                    'Datavaruhusutvecklare': 'Data Warehouse / ETL Developer',
                    'ETL Developer': 'Data Warehouse / ETL Developer',
                    'Data Architect': 'Other relevant Data & AI roles'}
        for title, expected in examples.items():
            with self.subTest(title=title):
                self.assertEqual(classify_job(title)['primary_job_category'], expected)

    def test_unicode(self):
        self.assertEqual(normalize_text('  Ｄata\u00a0Engineer\u2028\u200b TEST\t'), 'data engineer test')
        self.assertEqual(normalize_text('A\u030aÄÖ'), 'åäö')

    def test_duplicate_and_original_preservation(self):
        raw = frame([{}, {}, {}])
        raw['advertisement_id'] = ['1', '1', '']
        raw.loc[1, 'job_title'] = 'Different later title'
        before = raw.copy(deep=True)
        cleaned = clean_jobs(raw)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned.iloc[0].job_title, 'Data Engineer')
        self.assertEqual(cleaned.iloc[0].description_text, 'Original\nÅÄÖ')
        pd.testing.assert_frame_equal(raw, before)
        raw['advertisement_id'] = ['', '', '']
        self.assertEqual(len(clean_jobs(raw)), 3)

    def test_ambiguous_and_incidental_ai(self):
        for title in ['Software Engineer', 'Analytiker', 'AI', 'Data Engineer / Data Scientist', 'Säljare Data Engineer']:
            with self.subTest(title=title):
                self.assertEqual(classify_job(title, 'Work with AI Engineer teams')['primary_job_category'], UNCERTAIN)
        self.assertEqual(classify_job('Kock', 'We use AI')['primary_job_category'], IRRELEVANT)
        self.assertEqual(classify_job('Software Engineer', 'AI Engineer') ['classification_method'], 'description_support')

    def test_date_boundaries(self):
        raw = frame([{} for _ in range(7)])
        raw['publication_date'] = ['2026-04-21', '2026-10-21T23:59:59', '2026-04-20',
                                   '2026-10-22', 'invalid', '', '2026-04-20T22:30:00Z']
        cleaned = clean_jobs(raw, '2026-04-21', '2026-10-21')
        self.assertEqual(cleaned.in_analysis_window.tolist(), [True, True, False, False, False, False, True])
        self.assertEqual(len(cleaned), 7)

    def test_review_reproducibility_and_validation(self):
        raw = frame([{} for _ in range(220)])
        raw['primary_job_category'] = ['Data Engineer'] * 160 + [UNCERTAIN] * 55 + [IRRELEVANT] * 5
        a = review_sample(raw)
        pd.testing.assert_frame_equal(a, review_sample(raw))
        self.assertEqual(len(a), 150)
        self.assertEqual(a.advertisement_id.nunique(), 150)
        self.assertEqual(set(a.primary_job_category), {'Data Engineer', UNCERTAIN, IRRELEVANT})
        self.assertIsNone(evaluate_review(a)[0]['accuracy'])
        a.loc[a.index[:2], 'correct_category'] = ['Data Engineer', 'Data Engineer']
        metrics, matrix = evaluate_review(a)
        self.assertEqual(metrics['reviewed'], 2)
        self.assertEqual(metrics['accuracy'], .5)
        self.assertEqual(matrix.to_numpy().sum(), 2)
        a.loc[a.index[0], 'correct_category'] = 'typo'
        with self.assertRaises(ValueError):
            evaluate_review(a)

    def test_pipeline_preserves_source_and_refuses_overwrite(self):
        root = Path(__file__).parent / ('.test-step3-' + uuid.uuid4().hex)
        root.mkdir()
        self.addCleanup(lambda: shutil.rmtree(root))
        collection = root / 'collection'
        collection.mkdir()
        frame([{}, {}]).to_csv(collection / 'jobs.csv', index=False)
        (collection / 'configuration.json').write_text(json.dumps({'start':'2026-04-21', 'end':'2026-10-21'}))
        source = collection / 'jobs.csv'
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        output, summary = run_pipeline(collection, root / 'processed')
        self.assertEqual(before, hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(summary['relevant'], 2)
        self.assertEqual(sum(summary['count_per_job_category'].values()), 2)
        self.assertTrue((output / 'manual_review.csv').exists())
        with self.assertRaises(FileExistsError):
            run_pipeline(collection, output)

    def test_latest_collection_ignores_incomplete(self):
        root = Path(__file__).parent / ('.test-step3-' + uuid.uuid4().hex)
        root.mkdir()
        self.addCleanup(lambda: shutil.rmtree(root))
        for name, status in [('a', 'queries_exhausted'), ('z', 'incomplete')]:
            folder = root / name
            folder.mkdir()
            (folder / 'summary.json').write_text(json.dumps({'status':status, 'finished_at':name}))
            (folder / 'jobs.csv').touch()
        self.assertEqual(latest_collection(root).name, 'a')
