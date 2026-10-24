"""Behavioral tests for two-stage classification and verified-only validation."""
import json
import io
import shutil
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from src.classify_jobs_v2 import CATEGORIES, classify_frame, classify_job, description_evidence, text
from src.run_step3b import evaluate_verified_review, latest_step3, make_samples, run


class Step3BTests(unittest.TestCase):
    def assert_prediction(self, title, description, relevance, category=None):
        result = classify_job(title, description)
        self.assertEqual(result['relevance'], relevance, result)
        self.assertEqual(result['primary_category'], category, result)
        self.assertIn(result['classification_method'], {'title_rule', 'description_rule', 'manual'})
        return result

    def test_missing_evidence_is_uncertain(self):
        for title, body in [(None,None), ('Consultant',''), ('Engineer','We value curiosity.'), ('DE',''),
                            ('CDO (Chief Digital Officer)',''), ('Head of Data Center','')]:
            with self.subTest(title=title):
                r = self.assert_prediction(title, body, 'uncertain')
                self.assertTrue(r['review_required'])

    def test_plural_seniority_punctuation_and_swedish(self):
        cases = {'Senior AI Engineers':'AI Engineer', 'AI—ingenjörer, seniora':'AI Engineer',
                 'Machine-Learning Engineers (Principal)':'ML Engineer', 'ML-ingenjör':'ML Engineer',
                 'Junior DATA ENGINEERS':'Data Engineer', 'Dataanalytiker':'Data Analyst',
                 'Sr. BI-utvecklare':'BI Developer / BI Analyst', 'DWH Developer':'Data Warehouse / ETL Developer',
                 'Data Architect - Senior':'Data Architect', 'Datastyrning':'Data Governance / Data Quality',
                 'Senior IDQ-utvecklare':'Data Governance / Data Quality',
                 'Statistiker / Forskningsassistent':'Statistician', 'Biostatisticians':'Statistician',
                 'Head of Data Engineering':'Data / AI Management',
                 'Doktorand i maskininlärning':'AI / ML Research',
                 'Lead Data Engineer':'Data Engineer', 'Data Governance Lead':'Data Governance / Data Quality',
                 'Sr. AI Eng.':'AI Engineer'}
        for title, expected in cases.items():
            with self.subTest(title=title):
                self.assert_prediction(title, '', 'relevant', expected)

    def test_relevance_without_single_category(self):
        for title in ['Data Engineer / Data Architect', 'AI / ML Engineer', 'Data Analyst / Data Scientist']:
            r = self.assert_prediction(title, '', 'relevant')
            self.assertTrue(r['review_required'])
            self.assertIn('conflicting_primary_categories', r['review_reasons'])

    def test_real_description_duties(self):
        cases = [
            ('Engineer', 'Your role\nBuild data pipelines for ingestion.', 'Data Engineer'),
            ('Snowflake Developer', 'You will build and maintain ETL/ELT workflows and data pipelines.', 'Data Warehouse / ETL Developer'),
            ('Analytiker', 'Arbetsuppgifter\nSammanställa och analysera statistik och trender.', 'Data Analyst'),
            ('Developer', 'Du kommer att utveckla AI-agenter för interna processer.', 'AI Engineer'),
            ('Consultant', 'Responsibilities\nDevelop data governance frameworks.', 'Data Governance / Data Quality'),
            ('Postdoktor i fysik', 'Arbetsuppgifter\nDu kommer att testa verktyg och metoder för maskininlärning.', 'AI / ML Research'),
        ]
        for title, body, category in cases:
            with self.subTest(title=title):
                r=self.assert_prediction(title,body,'relevant',category)
                self.assertEqual(r['classification_method'],'description_rule')
                self.assertTrue(r['evidence']['description_matches'])

    def test_required_experience_needs_review(self):
        r=self.assert_prediction('Engineer','Requirements\nYou have experience building data pipelines.',
                                 'relevant','Data Engineer')
        self.assertIn('required_competence_only',r['review_reasons'])

    def test_analysis_requires_specific_deliverables(self):
        self.assert_prediction('Ingenjör inom produktionsteknik',
            'Arbetsuppgifter\nAnalysera data och flöden i utfört arbete.', 'uncertain')
        self.assert_prediction('Analytiker',
            'Du kommer att analysera data och presentera insikter och rapporter.', 'relevant', 'Data Analyst')

    def test_manager_noun_is_not_management_duty(self):
        result=classify_job('Engineer', 'You will deliver our Master Data Management solutions, supported by our Data Engineering team.')
        self.assertNotEqual(result['primary_category'],'Data / AI Management')
        self.assert_prediction('Enhetschef Analys och Innovation',
            'Du utvecklar vårt datalager.', 'relevant', 'Data / AI Management')
        self.assert_prediction('Master Data Management Solution Engineer', '',
            'relevant', 'Data Governance / Data Quality')

    def test_optional_skills_do_not_establish_duties(self):
        self.assert_prediction('Consultant','Nice to have\nExperience building data pipelines.', 'uncertain')
        self.assert_prediction('Utvecklare','Meriterande\nDu har erfarenhet av att utveckla AI-agenter.', 'uncertain')
        self.assert_prediction('Engineer', 'Preferred qualifications\nBuild data pipelines.\nResponsibilities\nBuild web applications.', 'irrelevant')

    def test_incidental_ai_is_not_ai_engineering(self):
        for body in [
            'We use ChatGPT and Copilot for productivity.',
            'AI supports our recruitment process.',
            'You collaborate with data scientists who build machine learning models.',
            'You will work closely with data engineers who develop data pipelines.',
            'Du samarbetar med data scientists som utvecklar maskininlärningsmodeller.',
            'About us\nWe develop AI agents for clients.',
            'Our services include building data pipelines and AI systems.',
            'Our team has experience building data pipelines.',
            'We have experience building data pipelines for our customers.',
            'You will support data scientists who develop machine learning models.',
        ]:
            with self.subTest(body=body):
                self.assert_prediction('Consultant',body,'uncertain')
        self.assert_prediction('Rekryterare','Du arbetar med en HR Data Analyst. Vi använder AI i rekryteringsprocessen.','irrelevant')

    def test_nondata_duties_with_ai_tools(self):
        self.assert_prediction('Developer', 'Responsibilities\nBuild web applications. We use AI tools to speed up coding.', 'irrelevant')
        self.assert_prediction('Ingenjör Licensiering', 'AI används i rekrytering.', 'irrelevant')
        self.assert_prediction('Senior affärsjurist till AI-bolag', '', 'irrelevant')

    def test_negation_blocks_positive_match(self):
        for body in ['You will not develop AI agents.', 'You do not build data pipelines.',
                     'Du kommer inte att utveckla AI-agenter.']:
            with self.subTest(body=body):
                self.assert_prediction('Engineer',body,'uncertain')

    def test_conflicting_nondata_and_data_evidence(self):
        r=self.assert_prediction('Mechanical Engineer','You will develop AI agents.','uncertain')
        self.assertTrue(r['review_required'])
        self.assert_prediction('Data Engineer','You will build web applications.','uncertain')

    def test_unicode_and_original_frame_preserved(self):
        raw=pd.DataFrame([dict(advertisement_id='001',job_title='Senior\u00a0AI\u2011Engineers',
            description_text='Original\nÅÄÖ',primary_job_category='Uncertain / Needs Review',
            classification_method='ambiguous_title',classification_evidence='original evidence',review_status='needs_review')])
        original=raw.copy(deep=True)
        classified=classify_frame(raw)
        pd.testing.assert_frame_equal(raw,original)
        self.assertEqual(classified.loc[0,'description_text'],'Original\nÅÄÖ')
        self.assertEqual(classified.loc[0,'primary_category'],'AI Engineer')
        self.assertEqual(classified.loc[0,'step3_classification_method'],'ambiguous_title')
        self.assertEqual(classified.loc[0,'step3_classification_evidence'],'original evidence')

    def test_sampling_random_reproducible_and_targeted_disjoint(self):
        raw=pd.DataFrame([dict(advertisement_id=str(i),job_title='Engineer',description_text='',
                              primary_job_category='Uncertain / Needs Review',classification_method='ambiguous_title',
                              classification_evidence='old',review_status='needs_review') for i in range(400)])
        classified=classify_frame(raw)
        classified.at[0,'review_reasons']=['conflicting_primary_categories']
        classified.at[1,'review_reasons']=['conflicting_data_and_nondata_duties_or_title']
        a,b=make_samples(classified)
        c,d=make_samples(classified)
        pd.testing.assert_frame_equal(a,c)
        pd.testing.assert_frame_equal(b,d)
        self.assertEqual(len(a),200)
        self.assertEqual(len(b),150)
        self.assertTrue(set(a.advertisement_id).isdisjoint(b.advertisement_id))
        self.assertTrue(b.relevance.eq('uncertain').all())
        self.assertTrue(a.human_verified.eq('').all())

    def test_verified_humans_only(self):
        review=pd.DataFrame([dict(human_relevance='relevant',human_primary_category='AI Engineer',
            human_verified='',reviewer='',relevance='relevant',primary_category='AI Engineer',
            sampling_design='simple_random_without_replacement')])
        metrics=evaluate_verified_review(review)
        self.assertNotIn('relevance_sample_accuracy',metrics)
        review.at[0,'human_verified']='true'
        with self.assertRaises(ValueError): evaluate_verified_review(review)
        review.at[0,'reviewer']='Human reviewer'
        self.assertEqual(evaluate_verified_review(review)['relevance_sample_accuracy'],1.0)
        review.at[0,'human_primary_category']='typo'
        with self.assertRaises(ValueError): evaluate_verified_review(review)

    def test_pipeline_preserves_inputs_and_no_overwrite(self):
        temp=Path(__file__).parent / ('.test-step3b-'+uuid.uuid4().hex)
        temp.mkdir()
        self.addCleanup(lambda:shutil.rmtree(temp))
        processed=temp/'old'
        processed.mkdir()
        original=pd.DataFrame([dict(advertisement_id='001',job_title='AI Engineers',description_text='You build AI agents.',
            primary_job_category='Uncertain / Needs Review',classification_method='ambiguous_title',
            classification_evidence='saved',review_status='needs_review'),
            dict(advertisement_id='002',job_title='Mystery role',description_text='',
            primary_job_category='Uncertain / Needs Review',classification_method='insufficient_evidence',
            classification_evidence='saved unknown',review_status='needs_review')])
        source=processed/'jobs_classified.csv'
        original.to_csv(source,index=False)
        before=source.read_bytes()
        with redirect_stdout(io.StringIO()):
            output,summary=run(processed,temp/'new',representative_size=1,targeted_size=1)
        self.assertEqual(source.read_bytes(),before)
        self.assertEqual(summary['relevance_after']['relevant'],1)
        entries=[json.loads(line,parse_constant=lambda constant:self.fail('Nonstandard JSON value: '+constant))
                 for line in (output/'classification_comparison.jsonl').read_text().splitlines()]
        entry=entries[0]
        self.assertEqual(entry['before']['primary_job_category'],'Uncertain / Needs Review')
        self.assertEqual(entry['after']['primary_category'],'AI Engineer')
        self.assertIsInstance(entry['after']['review_required'],bool)
        self.assertIsNone(entries[1]['after']['primary_category'])
        self.assertEqual(entries[1]['after']['relevance'],'uncertain')
        with self.assertRaises(FileExistsError):run(processed,output)

    def test_date_flags_review_and_counts(self):
        raw=pd.DataFrame([dict(advertisement_id='1',job_title='Data Engineer',description_text='text',
            primary_job_category='Data Engineer',classification_method='title_rule',classification_evidence='old',
            review_status='pending_validation',date_status='outside_window')])
        classified=classify_frame(raw)
        self.assertTrue(classified.loc[0,'review_required'])
        self.assertIn('invalid_or_out_of_window_date',classified.loc[0,'review_reasons'])


if __name__=='__main__':unittest.main()
