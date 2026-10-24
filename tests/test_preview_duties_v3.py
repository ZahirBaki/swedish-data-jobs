"""v3 preview policy: occupational evidence, retained flags, no activation."""
import copy
import unittest
from unittest.mock import patch

from src.preview_duties_v3 import decision,evidence,clauses,corroborated
from src.step3f_reliable_classifier import validate_reliable_prediction
from src.step3f_classifier import span_match
from tests.test_step3f import prediction


def row(body='You will build data pipelines for analytical reporting.',**changes):
    p=prediction();p.update(changes)
    p=validate_reliable_prediction(p,'Data Engineer',body)
    return dict(advertisement_id='test',job_title='Data Engineer',description_text=body,
        ai_status='completed',ai_prompt_version='step3f-1.1',ai_human_verified=False,
        ai_evidence_spans=span_match(p['ai_evidence_span_ids'],'Data Engineer',body)['spans'],**p)


class PreviewV3Tests(unittest.TestCase):
    def test_concrete_duty_requires_full_span_coverage(self):
        h=dict(duty_start=200,duty_end=280)
        self.assertTrue(corroborated(h,[dict(source_field='description_text',start=0,end=240),dict(source_field='description_text',start=240,end=480)]))
        self.assertFalse(corroborated(h,[dict(source_field='description_text',start=0,end=240)]))
        self.assertFalse(corroborated(h,[dict(source_field='description_text',start=0,end=200)]))

    def test_generic_sql_and_edi_integration_do_not_support_data_engineering(self):
        for body in ['You will develop SQL backend services.', 'Responsibilities\nBuild integrations between SAP and EDI systems.']:
            self.assertEqual(evidence(body,'Data Engineer'),[])

    def test_governance_category_remains_separate_even_if_saved_scope_is_core(self):
        d=decision(row('You will implement data governance.',ai_primary_category='Data Governance / Data Quality'))
        self.assertEqual(d['reporting_lane'],'governance')
        self.assertEqual(d['preview_decision'],'separate_lane')

    def test_third_party_responsibilities_do_not_create_an_employee_section(self):
        for body in ["Our clients responsibilities include: Build data pipelines.",
                     "Our colleagues have responsibilities: Develop AI agents."]:
            self.assertEqual(evidence(body,'Data Engineer'),[])

    def test_employee_intro_supports_following_imperative_duty(self):
        body='Working as a Data Engineer, you will:\nBuild data pipelines.'
        self.assertTrue(evidence(body,'Data Engineer'))

    def test_one_substantial_duty_can_qualify_but_never_activates(self):
        r=row();original=copy.deepcopy(r);d=decision(r)
        self.assertEqual(d['preview_decision'],'would_qualify_v3')
        self.assertFalse(d['activated']);self.assertFalse(d['approved'])
        self.assertFalse(d['human_reviewed']);self.assertEqual(r,original)

    def test_secondaries_do_not_disqualify_supported_primary(self):
        d=decision(row(ai_secondary_categories=['BI Developer / BI Analyst']))
        self.assertEqual(d['preview_decision'],'would_qualify_v3')

    def test_secondaries_do_not_establish_primary(self):
        d=decision(row('You will build AI agents.',ai_secondary_categories=['AI Engineer']))
        self.assertEqual(d['preview_decision'],'abstain')

    def test_noun_form_role_duties(self):
        d=decision(row('Your responsibilities\nDevelopment of data pipelines.'))
        self.assertEqual(d['preview_decision'],'would_qualify_v3')

    def test_swedish_list_and_inflection(self):
        body='Arbetsuppgifter\nUtveckling av dataplattformar.\nBygga datapipelines.'
        self.assertEqual(len(evidence(body,'Data Engineer')),2)

    def test_qualifications_and_company_context_end_duty_context(self):
        for stop in ['Requirements','Qualifications','About the Company','Kvalifikationer','Din profil','Om oss']:
            body='Responsibilities\nAttend meetings.\n'+stop+'\nBuild data pipelines.'
            with self.subTest(stop=stop):self.assertEqual(evidence(body,'Data Engineer'),[])

    def test_tool_title_and_quotation_alone_insufficient(self):
        self.assertEqual(decision(row('SQL, Python and data pipelines are interesting.'))['preview_decision'],'abstain')

    def test_qualification_optional_negative_and_third_party_not_duties(self):
        for body in ['You have experience building data pipelines.',
                     'You may build data pipelines.','You will not build data pipelines.',
                     "You won't build data pipelines.",
                     'Our company builds data pipelines.',
                     'Responsibilities\nOur clients build data pipelines.',
                     'You will work with data engineers who build data pipelines.']:
            with self.subTest(body=body):self.assertEqual(evidence(body,'Data Engineer'),[])

    def test_flags_are_never_resolved(self):
        for changes in [dict(ai_review_required=True),dict(ai_multi_role=True),dict(ai_confidence='medium')]:
            d=decision(row(**changes));self.assertNotEqual(d['preview_decision'],'would_qualify_v3')

    def test_opportunities_academic_and_adjacent_identifiable(self):
        for changes,lane in [(dict(ai_opportunity_types=['internship']),'opportunity'),
            (dict(ai_primary_category='AI / ML Research',ai_scope='academic_ai_ml_research'),'academic_research'),
            (dict(ai_primary_category='Statistician',ai_scope='adjacent_borderline'),'adjacent')]:
            d=decision(row(**changes));self.assertEqual(d['reporting_lane'],lane)
            self.assertEqual(d['preview_decision'],'separate_lane')

    def test_governance_separate(self):
        d=decision(row('You will implement data governance.',ai_primary_category='Data Governance / Data Quality',ai_scope='data_ai_governance'))
        self.assertEqual(d['reporting_lane'],'governance')
        self.assertEqual(d['preview_decision'],'separate_lane')

    def test_exact_unicode_offsets(self):
        body='Åäö\nArbetsuppgifter\nDu utvecklar datapipelines.'
        for e in evidence(body,'Data Engineer'):
            self.assertEqual(body[e['start']:e['end']],e['text'])
            self.assertEqual(body[e['duty_start']:e['duty_end']],e['duty_text'])
        self.assertTrue(evidence(body,'Data Engineer'))

    def test_invalid_source_and_technical_failures_remain_technical(self):
        for status in ['invalid_model_response','request_outcome_unknown']:
            r=row();r['ai_status']=status
            with patch('requests.sessions.Session.request',side_effect=AssertionError('No network')):
                self.assertEqual(decision(r)['preview_decision'],'technical_failure')
        r=row();r['ai_evidence_spans'][0]['text']='corrupt'
        self.assertEqual(decision(r)['preview_decision'],'technical_failure')

    def test_selected_description_evidence_must_corroborate_duties(self):
        body='General text. '+'x'*250+'. You will build data pipelines.'
        d=decision(row(body))
        self.assertEqual(d['preview_decision'],'abstain')
        self.assertIn('duties_not_corroborated_by_validated_description_evidence',d['reasons'])

    def test_irrelevant_and_uncertain_unchanged(self):
        for relevance,scope in [('irrelevant','unrelated'),('uncertain','unresolved')]:
            r=row(ai_relevance=relevance,ai_primary_category=None,ai_scope=scope)
            d=decision(r)
            self.assertEqual(d['preview_decision'],'not_relevant_candidate')
            self.assertEqual(d['saved_relevance'],relevance)

if __name__=='__main__':unittest.main()
