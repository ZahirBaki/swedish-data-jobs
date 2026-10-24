"""Synthetic duplicate cases and hand-calculable verified-label metrics."""
import io
import json
import shutil
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from src.classification_policy import add_policy_flags,policy_flags
from src.detect_duplicate_jobs import Settings,detect_duplicates,normalize,run
from src.validate_classification import (ABSTAIN,diagnostic_sample,evaluate_review,
    latest_step3b,prepare_review_files,read_csv,representative_sample,save_review_edits)


BODY = ' '.join(f'word{i}' for i in range(100))


def ad(identifier,**updates):
    row = dict(advertisement_id=str(identifier),employer='Acme AB',job_title='Data Engineer',
        description_text=BODY,publication_date='2026-06-01T10:00:00',municipality='Stockholm',
        region='Stockholms län',number_of_vacancies='1',relevance='relevant',primary_category='Data Engineer',
        classification_method='title_rule',evidence='{}',candidate_categories='["Data Engineer"]',
        review_reasons='[]',review_required='False')
    row.update(updates)
    return row


def review_row(identifier,truth='relevant',prediction='relevant',category='Data Engineer',pred_category='Data Engineer',verified='true'):
    return dict(advertisement_id=str(identifier),human_relevance=truth,human_primary_category=category,
        human_verified=verified,reviewer='Synthetic human fixture',review_notes='',relevance=prediction,
        primary_category=pred_category,sampling_design='simple_random_without_replacement')


class Step3CDuplicateTests(unittest.TestCase):
    def test_exact_normalization_and_original_preservation(self):
        frame = pd.DataFrame([ad(1),ad(2,employer=' ACME\u00a0AB ',job_title='DATA\u2028ENGINEER',
                                            description_text=BODY.upper().replace(' ','\u00a0'))])
        original=frame.copy(deep=True)
        out,pairs,groups,summary=detect_duplicates(frame)
        self.assertEqual(len(out),2)
        self.assertEqual(summary['exact_text_groups'],1)
        self.assertEqual(summary['exact_pairs'],1)
        self.assertEqual(summary['records_deleted'],0)
        self.assertEqual(pairs.iloc[0].duplicate_review_status,'exact_repeat_candidate_needs_review')
        pd.testing.assert_frame_equal(frame,original)
        pd.testing.assert_frame_equal(out[frame.columns],original)

    def test_different_ids_near_text(self):
        changed=BODY.replace('word50','changed50')
        out,pairs,groups,summary=detect_duplicates(pd.DataFrame([ad(1),ad(2,description_text=changed)]))
        self.assertEqual(summary['near_pairs'],1)
        self.assertEqual(summary['exact_pairs'],0)
        self.assertGreater(pairs.iloc[0].description_similarity,.85)
        self.assertEqual(out.duplicate_group_id.nunique(),1)

    def test_separate_vacancy_signals(self):
        cases=[dict(publication_date='2026-06-10'),dict(municipality='Göteborg'),dict(number_of_vacancies='2')]
        for changes in cases:
            with self.subTest(changes=changes):
                out,pairs,_,_=detect_duplicates(pd.DataFrame([ad(1),ad(2,**changes)]))
                self.assertEqual(pairs.iloc[0].duplicate_review_status,'possible_separate_vacancies_or_repost_needs_review')
                self.assertTrue(json.loads(pairs.iloc[0].separate_vacancy_signals))
                self.assertEqual(len(out),2)

    def test_missing_and_placeholder_descriptions(self):
        frame=pd.DataFrame([ad(1,description_text=''),ad(2,description_text=None)])
        self.assertEqual(detect_duplicates(frame)[3]['duplicate_review_groups'],0)
        short=pd.DataFrame([ad(1,description_text='Apply now'),ad(2,description_text='Apply now')])
        self.assertEqual(detect_duplicates(short)[1].iloc[0].duplicate_review_status,'limited_context_needs_review')
        placeholder='För att se annonsen i sin helhet '+BODY
        frame=pd.DataFrame([ad(1,description_text=placeholder),ad(2,description_text=placeholder)])
        self.assertTrue(detect_duplicates(frame)[1].iloc[0].limited_description_context)

    def test_missing_metadata_not_assumed_equal(self):
        out,pairs,_,_=detect_duplicates(pd.DataFrame([ad(1,municipality=''),ad(2,municipality='')]))
        self.assertEqual(pairs.iloc[0].duplicate_review_status,'metadata_incomplete_needs_review')
        self.assertIsNone(pairs.iloc[0].same_known_location)

    def test_different_employer_and_title_are_not_exact(self):
        frame=pd.DataFrame([ad(1),ad(2,employer='Other AB'),ad(3,job_title='Nurse')])
        self.assertEqual(detect_duplicates(frame)[3]['duplicate_review_groups'],0)
        self.assertNotEqual(normalize('C# Engineer'),normalize('C++ Engineer'))

    def test_group_ids_stable_and_components_not_cliques(self):
        b=BODY
        for n in [20,21,22]:b=b.replace(f'word{n} ',f'change{n} ')
        c=b
        for n in [60,61,62]:c=c.replace(f'word{n} ',f'change{n} ')
        frame=pd.DataFrame([ad('a'),ad('b',description_text=b),ad('c',description_text=c)])
        out,pairs,groups,_=detect_duplicates(frame)
        self.assertEqual(len(groups),1)
        self.assertEqual(len(pairs),2)
        shuffled=detect_duplicates(frame.sample(frac=1,random_state=7))[0]
        self.assertEqual(out.set_index('advertisement_id').duplicate_group_id.to_dict(),
                         shuffled.set_index('advertisement_id').duplicate_group_id.to_dict())

    def test_duplicate_ids_and_invalid_settings_rejected(self):
        with self.assertRaises(ValueError):detect_duplicates(pd.DataFrame([ad(1),ad(1)]))
        for identifier in ['',None,'   ']:
            with self.subTest(identifier=identifier),self.assertRaises(ValueError):
                detect_duplicates(pd.DataFrame([ad(1,advertisement_id=identifier)]))
        with self.assertRaises(ValueError):Settings(near_threshold=1.1)

    def test_swedish_calendar_day_comparison(self):
        frame=pd.DataFrame([ad(1,publication_date='2026-06-01T23:00:00Z'),
                            ad(2,publication_date='2026-06-02T01:00:00+02:00')])
        self.assertEqual(detect_duplicates(frame)[1].iloc[0].publication_day_gap,0)


class Step3CEvaluationTests(unittest.TestCase):
    def test_no_unverified_ground_truth(self):
        data=pd.DataFrame([review_row(1,verified=''),review_row(2,verified='false')])
        result=evaluate_review(data)
        self.assertIsNone(result['relevance'])
        self.assertIsNone(result['primary_category'])
        self.assertEqual(result['counts']['unreviewed_records'],2)
        self.assertEqual(result['counts']['entered_but_unverified'],2)

    def test_relevance_precision_recall_f1_by_hand(self):
        rows=[review_row(1),review_row(2,truth='irrelevant',category=''),
              review_row(3,prediction='uncertain',pred_category='')]
        result=evaluate_review(pd.DataFrame(rows))
        scores=result['relevance']['per_class'].set_index('label').loc['relevant']
        self.assertEqual(scores.precision,.5)
        self.assertEqual(scores.recall,.5)
        self.assertEqual(scores.f1,.5)
        matrix=result['relevance']['confusion_matrix']
        self.assertEqual(matrix.loc['relevant','uncertain'],1)
        self.assertEqual(matrix.loc['irrelevant','relevant'],1)

    def test_category_metrics_include_missed_relevance_as_abstention(self):
        rows=[review_row(1),review_row(2,pred_category='Data Scientist'),
              review_row(3,prediction='irrelevant',pred_category='')]
        result=evaluate_review(pd.DataFrame(rows))
        scores=result['primary_category']['per_class'].set_index('label').loc['Data Engineer']
        self.assertEqual(scores.precision,1)
        self.assertAlmostEqual(scores.recall,1/3)
        self.assertEqual(scores.f1,.5)
        self.assertEqual(result['primary_category']['confusion_matrix'].loc['Data Engineer',ABSTAIN],1)

    def test_unresolved_truth_excluded_and_reported(self):
        rows=[review_row(1,truth='uncertain',category=''),review_row(2,category=''),
              review_row(3,verified='')]
        result=evaluate_review(pd.DataFrame(rows))
        self.assertEqual(result['counts']['unresolved_human_relevance'],1)
        self.assertEqual(result['counts']['unresolved_human_category'],1)
        self.assertEqual(result['counts']['relevance_evaluable'],1)
        self.assertIsNone(result['primary_category'])

    def test_verified_label_integrity_and_mixed_designs(self):
        for updates in [dict(human_primary_category='Typo'),dict(reviewer=''),dict(human_verified='yes'),
                        dict(human_relevance='irrelevant',human_primary_category='Data Engineer')]:
            row=review_row(1)
            row.update(updates)
            with self.subTest(updates=updates),self.assertRaises(ValueError):evaluate_review(pd.DataFrame([row]))
        with self.assertRaises(ValueError):evaluate_review(pd.DataFrame([review_row(1),review_row(1)]))
        b=review_row(2);b['sampling_design']='targeted_error_audit_excluding_representative'
        with self.assertRaises(ValueError):evaluate_review(pd.DataFrame([review_row(1),b]))

    def test_position_flags_are_independent_of_prediction(self):
        flags=policy_flags('PhD Student - AI internship','uncertain','')
        self.assertTrue(flags['is_doctoral'])
        self.assertTrue(flags['is_internship'])
        self.assertFalse(flags['core_dashboard_candidate'])
        self.assertFalse(policy_flags('Data Scientist (PhD preferred)','relevant','Data Scientist')['is_doctoral'])
        self.assertEqual(policy_flags('Statistician','relevant','Statistician')['role_scope'],'statistician')
        self.assertTrue(policy_flags('Data Engineer','relevant','Data Engineer')['core_dashboard_candidate'])


class Step3CPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=Path(__file__).parent/('.test-step3c-'+uuid.uuid4().hex)
        self.temp.mkdir()
        self.addCleanup(lambda:shutil.rmtree(self.temp))

    def test_reuse_representative_and_targeted_strata(self):
        source=self.temp/'source';source.mkdir()
        out=self.temp/'out';out.mkdir()
        rows=[ad(i,relevance=['relevant','irrelevant','uncertain'][i%3],
                 primary_category='Data Engineer' if i%3==0 else '',
                 candidate_categories='["Data Engineer"]' if i%3!=1 else '[]',
                 review_reasons='["conflicting_primary_categories"]' if i%10==0 else '[]') for i in range(500)]
        frame=pd.DataFrame(rows)
        frame.to_csv(source/'jobs_classified_v2.csv',index=False)
        sample=frame.sample(n=200,random_state=42).copy()
        sample['sampling_design']='simple_random_without_replacement';sample['sampling_seed']=42
        sample['sample_order']=range(1,201);sample['inclusion_probability']=200/500
        for col in ['human_relevance','human_primary_category','human_verified','reviewer','review_notes']:sample[col]=''
        sample.to_csv(source/'manual_review_representative.csv',index=False)
        before=(source/'manual_review_representative.csv').read_bytes()
        info=prepare_review_files(source,out,add_policy_flags(frame))
        reused=read_csv(out/'manual_review_representative.csv')
        target=read_csv(out/'manual_review_diagnostic.csv')
        self.assertTrue(info['representative_reused'])
        self.assertEqual(reused.advertisement_id.tolist(),sample.advertisement_id.tolist())
        self.assertTrue(set(reused.advertisement_id).isdisjoint(target.advertisement_id))
        self.assertEqual(set(target.diagnostic_stratum),{'suspected_false_positive','suspected_false_negative','conflicting_categories'})
        self.assertEqual((source/'manual_review_representative.csv').read_bytes(),before)
        (out/'step3c_summary.json').write_text('{}')
        identifier=reused.advertisement_id.iloc[0]
        save_review_edits(out/'manual_review_representative.csv',{identifier:{'review_notes':'User-entered note'}})
        self.assertTrue(read_csv(out/'manual_review_representative.csv').human_verified.eq('').all())
        with self.assertRaises(ValueError):save_review_edits(out/'manual_review_representative.csv',{identifier:{'relevance':'irrelevant'}})
        with self.assertRaises(ValueError):save_review_edits(source/'manual_review_representative.csv',{})

    def test_fallback_strata_reproducible(self):
        frame=pd.DataFrame([ad(i,relevance='relevant' if i<298 else 'uncertain' if i==298 else 'irrelevant') for i in range(300)])
        a=representative_sample(frame)
        pd.testing.assert_frame_equal(a,representative_sample(frame))
        self.assertEqual(len(a),200)
        self.assertEqual(set(a.relevance),{'relevant','irrelevant','uncertain'})

    def test_pipeline_preserves_all_inputs_and_refuses_overwrite(self):
        source=self.temp/'source';source.mkdir()
        frame=pd.DataFrame([ad(1),ad(2,relevance='irrelevant',primary_category=''),
                            ad(3,relevance='uncertain',primary_category='')])
        path=source/'jobs_classified_v2.csv';frame.to_csv(path,index=False)
        before=path.read_bytes()
        with redirect_stdout(io.StringIO()):output,summary=run(source,self.temp/'output')
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(summary['output_rows'],3)
        self.assertEqual(summary['records_deleted'],0)
        saved=read_csv(output/'jobs_duplicate_review.csv')
        pd.testing.assert_frame_equal(saved[frame.columns],frame)
        with redirect_stdout(io.StringIO()),self.assertRaises(FileExistsError):run(source,output)


if __name__=='__main__':unittest.main()
