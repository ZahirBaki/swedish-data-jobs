"""Deterministic source-grounded extraction; real model/network calls forbidden."""
import copy
import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.extract_requirements import extract_job
from src.requirements_summary import summarize,evaluate_requirement_review
from src.step4_jobs import validate_classification,select_pilot
from src.run_step4 import run
from src.validate_classification import sha256,read_csv


def job(body,title='Data Engineer',identifier='1'):
    return dict(advertisement_id=identifier,job_title=title,description_text=body,
        region='Stockholms län',municipality='Stockholm',employer='Example',publication_date='2026-06-01',
        ai_primary_category='Data Engineer',ai_relevance='relevant',ai_status='completed',
        reporting_scope='core_industry',step4_disposition='eligible_relevant')


def matching(facts,kind=None,value=None):
    return [r for r in facts if (kind is None or r['kind']==kind) and (value is None or r['value']==value)]


class ExtractionTests(unittest.TestCase):
    def test_required_preferred_and_merely_mentioned_skills_separate(self):
        result,facts=extract_job(job('Python is required. Azure is preferred. We use Tableau.'))
        self.assertIn('Python',result['skills_required']);self.assertIn('Azure',result['skills_preferred'])
        self.assertIn('Tableau',result['skills_mentioned_only'])
        self.assertNotIn('Tableau',result['skills_required'])

    def test_swedish_qualifications_preferred_and_unicode_offsets(self):
        original=job('Kvalifikationer\r\nDu har kunskaper i Python och SQL.\u2028Meriterande:\nDatabricks och dbt.','Senior dataingenjör – Göteborg')
        result,facts=extract_job(original)
        self.assertEqual(result['skills_required'],['Python','SQL'])
        self.assertEqual(result['skills_preferred'],['Databricks','dbt'])
        for r in facts:
            source=original[r['source_field']]
            self.assertEqual(source[r['match_start']:r['match_end']],r['matched_text'])
            self.assertEqual(source[r['evidence_start']:r['evidence_end']],r['evidence_text'])

    def test_glued_html_headings_apply_only_after_the_heading(self):
        body='We use Python. RequirementsSQL knowledge. What We OfferTraining in Azure.'
        result,facts=extract_job(job(body))
        self.assertIn('Python',result['skills_mentioned_only'])
        self.assertIn('SQL',result['skills_required'])
        self.assertIn('Azure',result['skills_mentioned_only'])
        sql=matching(facts,value='SQL')[0]
        self.assertEqual(body[sql['context_heading_start']:sql['context_heading_end']],sql['context_heading_text'])

    def test_preferred_heading_after_bom_and_preceding_sentence(self):
        result,_=extract_job(job('Our platform uses Databricks. \ufeffNice to have:\nPySpark/Apache Spark'))
        self.assertIn('Databricks',result['skills_mentioned_only'])
        self.assertEqual(result['skills_preferred'],['Apache Spark','PySpark'])

    def test_programming_alias_boundaries_and_punctuation(self):
        result,_=extract_job(job('Required: C#/.NET, C++, JavaScript and R.'))
        for value in ['C#','.NET','C++','JavaScript','R']:self.assertIn(value,result['skills_required'])
        self.assertNotIn('Java',result['skills_required']);self.assertNotIn('C',result['skills_required'])
        result,_=extract_job(job('Required:\n.NET'))
        self.assertIn('.NET',result['skills_required'])

    def test_sql_server_is_not_automatically_sql_language(self):
        result,_=extract_job(job('SQL Server is required.'))
        self.assertIn('SQL Server',result['skills_required']);self.assertNotIn('SQL',result['skills_required'])

    def test_cloud_aliases_normalized_but_source_not_modified(self):
        source=job('Required: Microsoft Azure, Amazon Web Services and GCP.')
        result,facts=extract_job(source)
        self.assertTrue({'Azure','AWS','Google Cloud'}<=set(result['skills_required']))
        self.assertEqual(matching(facts,value='Google Cloud')[0]['matched_text'],'GCP')

    def test_skill_alternatives_not_individually_mandatory(self):
        result,_=extract_job(job('Required: Python or R.'))
        self.assertEqual(result['skills_required'],[])
        self.assertEqual(result['skills_alternative_required'],['Python','R'])
        self.assertTrue(result['requirements_review_required'])

    def test_several_of_following_list_retains_alternatives(self):
        result,_=extract_job(job('Requirements\nExperience with several of the following:\nPython\nSQL'))
        self.assertEqual(result['skills_required'],[])
        self.assertEqual(result['skills_alternative_required'],['Python','SQL'])

    def test_sweden_language_required_preferred_and_not_required(self):
        for sentence,expected in [('Swedish is required.','required'),('Swedish is preferred.','preferred'),
            ('Swedish is not required.','explicitly_not_required'),('Svenska är inget krav.','explicitly_not_required')]:
            with self.subTest(sentence=sentence):self.assertEqual(extract_job(job(sentence))[0]['swedish_requirement'],expected)

    def test_english_and_swedish_language_requirement_independent(self):
        result,_=extract_job(job('Swedish is not required. Fluent English is required.'))
        self.assertEqual(result['swedish_requirement'],'explicitly_not_required')
        self.assertEqual(result['english_requirement'],'required')

    def test_language_application_team_and_ad_language_not_requirements(self):
        result,_=extract_job(job('Send your application in English. We have an English-speaking team. Swedish offices.'))
        self.assertEqual(result['english_requirement'],'not_specified')
        self.assertEqual(result['swedish_requirement'],'not_specified')

    def test_language_alternatives_and_conflicts_are_not_forced(self):
        result,_=extract_job(job('Fluent Swedish or English is required.'))
        self.assertEqual(result['swedish_requirement'],'not_specified');self.assertTrue(result['swedish_review_required'])
        result,_=extract_job(job('Swedish is required. Swedish is not required.'))
        self.assertEqual(result['swedish_requirement'],'not_specified');self.assertTrue(result['swedish_review_required'])

    def test_education_required_and_preferred_levels(self):
        result,facts=extract_job(job("A bachelor's degree is required. A master's degree is preferred."))
        self.assertEqual(result['education_levels'],['bachelor','master'])
        self.assertEqual(matching(facts,'education','bachelor')[0]['modality'],'required')
        self.assertEqual(matching(facts,'education','master')[0]['modality'],'preferred')

    def test_swedish_degree_and_degree_without_level(self):
        self.assertEqual(extract_job(job('Krav:\nDoktorsexamen i datavetenskap.'))[0]['education_levels'],['phd'])
        self.assertEqual(extract_job(job('A university degree is required.'))[0]['education_levels'],['degree_unspecified'])
        self.assertEqual(extract_job(job('A degree is required.'))[0]['education_levels'],['degree_unspecified'])
        self.assertEqual(extract_job(job('We require a high degree of independence.'))[0]['education_levels'],['not_specified'])

    def test_education_or_equivalent_experience_not_forced_degree(self):
        result,facts=extract_job(job("A master's degree or equivalent professional experience is required."))
        self.assertEqual(matching(facts,'education')[0]['modality'],'alternative_required')
        self.assertTrue(result['requirements_review_required'])

    def test_phd_position_or_supervising_students_not_degree_requirement(self):
        result,_=extract_job(job('You will supervise PhD students. This master thesis is about AI.','PhD student'))
        self.assertEqual(result['education_levels'],['not_specified'])

    def test_experience_numeric_range_minimum_plus_and_number_words(self):
        for sentence,years,bound,minimum,maximum in [
            ('At least 3 years of professional experience is required.',3,'at_least',3,None),
            ('Required: 3–5 years of experience.',3,'range',3,5),
            ('Required: 5+ years experience.',5,'at_least',5,None),
            ('Required: three years of work experience.',3,'stated_duration',None,None),
            ('Krav:\nMinst fem års erfarenhet av Python.',5,'at_least',5,None)]:
            with self.subTest(sentence=sentence):
                result,_=extract_job(job(sentence));record=result['experience_requirements'][0]
                self.assertEqual((record['stated_years'],record['bound_type'],record['minimum_years'],record['maximum_years']),
                    (years,bound,minimum,maximum))

    def test_years_of_study_and_employment_dates_not_experience(self):
        result,_=extract_job(job('A PhD takes five years. Apply in 2026. A three-year degree is common.'))
        self.assertEqual(result['experience_status'],'not_specified')

    def test_multiple_experience_requirements_preserved_independently(self):
        result,_=extract_job(job('Required: 5 years of experience in industry. 3 years of experience with Python.'))
        self.assertEqual(len(result['experience_requirements']),2)
        self.assertEqual({r['stated_years'] for r in result['experience_requirements']},{3,5})

    def test_seniority_explicit_title_not_years_or_colleagues(self):
        self.assertEqual(extract_job(job('Required: 10 years of experience.','Junior Data Analyst'))[0]['seniority'],'junior')
        self.assertEqual(extract_job(job('Work with senior data scientists.'))[0]['seniority'],'unspecified')
        self.assertEqual(extract_job(job('We are looking for a Senior Data Engineer.'))[0]['seniority'],'senior')

    def test_employment_region_hybrid_platform_does_not_imply_hybrid_work(self):
        result,facts=extract_job(job('Permanent employment. Full-time. A hybrid working arrangement.'))
        self.assertEqual(result['working_time'],['full_time']);self.assertEqual(result['work_arrangement'],['hybrid'])
        self.assertEqual(result['region'],'Stockholms län')
        self.assertTrue(matching(facts,'source_metadata','region'))
        self.assertEqual(extract_job(job('We build hybrid cloud architectures.'))[0]['work_arrangement'],['not_specified'])

    def test_missing_values_do_not_invent_requirements(self):
        source=job(None,title='');before=copy.deepcopy(source)
        result,records=extract_job(source)
        self.assertEqual(result['swedish_requirement'],'not_specified');self.assertEqual(result['education_levels'],['not_specified'])
        self.assertEqual(result['seniority'],'unspecified');self.assertFalse(result['requirements_data_usable'])
        self.assertEqual(source,before)

    def test_no_degree_or_skill_assumed_from_role_title(self):
        result,_=extract_job(job('Welcome to our team.','Senior Python / SQL developer'))
        self.assertEqual(result['skills_required'],[]);self.assertEqual(result['education_levels'],['not_specified'])


class SelectionSummaryTests(unittest.TestCase):
    def test_classification_evidence_must_match_original_source(self):
        source=job('Python is required.')
        p=dict(ai_status='completed',ai_prompt_version='step3e-1.0',ai_relevance='relevant',ai_primary_category='Data Engineer',
            ai_secondary_categories=[],ai_reasoning='Data role',ai_evidence='Python',ai_confidence='high',ai_review_required=False)
        self.assertTrue(validate_classification(p,source)[0])
        p['ai_evidence']='invented quotation'
        self.assertFalse(validate_classification(p,source)[0])

    def test_selection_deterministic_no_irrelevant_uncertain_or_technical_rows(self):
        frame=pd.DataFrame([job('Python is required.',identifier=str(i)) for i in range(60)])
        frame.loc[0,'step4_disposition']='excluded_irrelevant';frame.loc[1,'step4_disposition']='set_aside_uncertain'
        frame.loc[2,'step4_disposition']='set_aside_technical_failure'
        a=select_pilot(frame);b=select_pilot(frame.sample(frac=1,random_state=7))
        self.assertEqual(len(a),50);self.assertEqual(a.advertisement_id.tolist(),b.advertisement_id.tolist())
        self.assertFalse(set(a.advertisement_id)&{'0','1','2'})

    def test_summary_counts_distinct_jobs_not_repeated_mentions_and_denominators(self):
        first,facts1=extract_job(job('Python is required. Python is required.',identifier='1'))
        second,facts2=extract_job(job('SQL is preferred.',identifier='2'))
        third,facts3=extract_job(job('We use Python.',identifier='3'))
        summary=summarize(pd.DataFrame([first,second,third]),pd.DataFrame(facts1+facts2+facts3))
        python=next(r for r in summary['requested_skills'] if r['value']=='Python')
        self.assertEqual(python['job_count'],1);self.assertEqual(python['denominator_selected_jobs'],3)
        self.assertEqual(summary['denominators']['jobs_with_requested_skills'],2)
        self.assertEqual(summary['denominators']['usable_requirements_jobs'],2)

    def test_no_accuracy_before_human_review(self):
        _,facts=extract_job(job('Python is required.'))
        review=pd.DataFrame(facts)
        for field in ['human_valid','human_modality','human_verified','reviewer']:review[field]=''
        result=evaluate_requirement_review(review)
        self.assertIsNone(result['extraction_precision']);self.assertIsNone(result['recall'])

    def test_verified_review_metrics_require_human_reviewer(self):
        _,facts=extract_job(job('Python is required.'))
        review=pd.DataFrame(facts)
        for field in ['human_valid','human_modality','human_verified','reviewer']:review[field]=''
        review.loc[0,['human_valid','human_modality','human_verified','reviewer']]=['true','required','true','reviewer-1']
        self.assertEqual(evaluate_requirement_review(review)['extraction_precision'],1)
        review.loc[0,'reviewer']=''
        with self.assertRaises(ValueError):evaluate_requirement_review(review)


class Step4PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=Path(__file__).parent/('.test-step4-'+uuid.uuid4().hex);self.temp.mkdir()
        self.addCleanup(lambda:shutil.rmtree(self.temp))
        self.registry=pd.DataFrame([job('Python is required. English is required.',identifier=str(i)) for i in range(3)])
        self.source=self.temp/'original.csv';self.registry.to_csv(self.source,index=False)

    def test_offline_pipeline_preserves_source_and_emits_normalized_tables(self):
        before=sha256(self.source)
        with patch('src.run_step4.load_existing',return_value=(self.registry,[],[])),patch('src.run_step4.SOURCE',self.source),\
            patch('requests.sessions.Session.request',side_effect=AssertionError('No network')) as network:
            result=run(self.temp/'output',size=50)
            network.assert_not_called()
        self.assertEqual(result['selected_pilot_jobs'],3);self.assertEqual(result['expected_api_cost_usd'],0)
        for name in ['jobs_requirements.csv','job_skills.csv','job_language_requirements.csv','job_education_requirements.csv',
            'job_experience_requirements.csv','requirements_summary.json','manual_review_requirements.csv']:
            self.assertTrue((self.temp/'output'/name).exists())
        rows=read_csv(self.temp/'output/jobs_requirements.csv')
        self.assertTrue(rows.description_text.eq('Python is required. English is required.').all())
        self.assertEqual(sha256(self.source),before)
        with self.assertRaises(FileExistsError):run(self.temp/'output')

    def test_dry_run_generates_no_model_calls_or_extraction_outputs(self):
        with patch('src.run_step4.load_existing',return_value=(self.registry,[],[])),patch('src.run_step4.SOURCE',self.source),\
            patch('requests.sessions.Session.request',side_effect=AssertionError('No network')) as network:
            result=run(self.temp/'dry',dry_run=True)
            network.assert_not_called()
        self.assertEqual(result['classifier_requests'],0)
        self.assertFalse((self.temp/'dry/jobs_requirements.csv').exists())


if __name__=='__main__':unittest.main()
