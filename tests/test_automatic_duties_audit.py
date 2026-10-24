"""Offline candidate rules cannot grant approvals under person-audit policy v1."""
import copy
import unittest
from unittest.mock import patch

from src.automatic_duties_audit import audit_row, duty_matches
from src.step3f_reliable_classifier import validate_reliable_prediction
from src.step3f_classifier import span_match
from tests.test_step3f import prediction

BODY = 'You will build data pipelines for analytics. You will operate data platforms for business reporting.'


def row(body=BODY, **changes):
    value=prediction()
    value.update(changes)
    value=validate_reliable_prediction(value,'Data Engineer',body)
    return dict(advertisement_id='new',job_title='Data Engineer',description_text=body,
        ai_status='completed',ai_prompt_version='step3f-1.1',ai_human_verified=False,
        ai_evidence_spans=span_match(value['ai_evidence_span_ids'],'Data Engineer',body)['spans'],**value)


class AutomaticDutiesTests(unittest.TestCase):
    def test_candidate_is_never_approved_under_v1(self):
        r=row();before=copy.deepcopy(r)
        result=audit_row(r)
        self.assertEqual(result['candidate_disposition'],'candidate_pending_policy_approval')
        self.assertFalse(result['automatic_approval'])
        self.assertFalse(result['human_reviewed'])
        self.assertEqual(result['approval_method'],'none')
        self.assertIn('duties_not_confirmed',result['active_policy_reasons'])
        self.assertEqual(r,before)
        self.assertNotIn('duties_assessment',result)

    def test_concrete_compound_duty_is_only_a_pending_candidate(self):
        result=audit_row(row('You will design and implement big data pipelines.'))
        self.assertEqual(result['candidate_disposition'],'candidate_pending_policy_approval')
        self.assertFalse(result['automatic_approval'])
        result=audit_row(row('You will build data pipelines.'))
        self.assertEqual(result['candidate_disposition'],'human_review')

    def test_unsupported_classifier_version_cannot_enter_automatic_lane(self):
        r=row();r['ai_prompt_version']='step3f-1.0'
        result=audit_row(r)
        self.assertIn('classifier_version_not_supported_by_automation',result['decision_reasons'])
        self.assertEqual(result['candidate_disposition'],'human_review')

    def test_negative_contractions_and_questions_do_not_qualify(self):
        for body in ["You won't build data pipelines.", 'Vill du bygga dataplattformar?']:
            self.assertEqual(duty_matches(body,'Data Engineer'),[])

    def test_exact_unicode_offsets_and_category_match(self):
        body='Åäö — You will build data pipelines. You will operate data platforms.'
        matches=duty_matches(body,'Data Engineer')
        self.assertEqual(len(matches),2)
        for m in matches:
            self.assertEqual(body[m['start']:m['end']],m['text'])
            self.assertEqual(body[m['action_object_start']:m['action_object_end']],m['action_object_text'])

    def test_tools_and_title_only_do_not_establish_duties(self):
        result=audit_row(row('We use Python, SQL and AI. Data Engineer is our preferred title.'))
        self.assertEqual(result['candidate_disposition'],'human_review')
        self.assertEqual(result['duty_evidence'],[])

    def test_qualifications_are_not_duties(self):
        for body in ['You have experience building data pipelines. You have skills to operate data platforms.',
                     'You will build data pipelines if required. You may operate data platforms.',
                     'You will not build data pipelines. You will never operate data platforms.',
                     'Our company will build data pipelines. Our colleagues operate data platforms.',
                     'You will help the team build data pipelines. You will watch colleagues operate data platforms.']:
            with self.subTest(body=body):
                self.assertEqual(duty_matches(body,'Data Engineer'),[])

    def test_duplicate_clauses_do_not_inflate_evidence(self):
        self.assertEqual(len(duty_matches('You will build data pipelines. You will build data pipelines.','Data Engineer')),1)

    def test_duties_bullet_heading_and_qualification_boundary(self):
        body='Responsibilities\n- Build data pipelines\n- Operate data platforms\nRequirements\n- Build data pipelines\n- Operate data platforms'
        self.assertEqual(len(duty_matches(body,'Data Engineer')),2)
        self.assertEqual(duty_matches('- Build data pipelines\n- Operate data platforms','Data Engineer'),[])

    def test_swedish_employee_duties(self):
        body='Du kommer att bygga dataplattformen. Du utvecklar datapipelines.'
        self.assertEqual(len(duty_matches(body,'Data Engineer')),2)

    def test_wrong_category_abstains(self):
        result=audit_row(row(ai_primary_category='AI Engineer'))
        self.assertEqual(result['candidate_disposition'],'human_review')
        self.assertEqual(result['duty_evidence'],[])

    def test_review_flag_cannot_be_resolved_by_rules(self):
        result=audit_row(row(ai_review_required=True))
        self.assertIn('mandatory_review_flag',result['decision_reasons'])
        self.assertEqual(result['candidate_disposition'],'human_review')

    def test_multirole_and_hybrids_remain_separate(self):
        for kwargs in [dict(ai_multi_role=True),dict(ai_secondary_categories=['Data Scientist'])]:
            result=audit_row(row(**kwargs))
            self.assertEqual(result['candidate_disposition'],'human_review')

    def test_low_confidence_cannot_be_promoted(self):
        result=audit_row(row(ai_confidence='medium'))
        self.assertIn('confidence_below_threshold',result['decision_reasons'])
        self.assertFalse(result['automatic_approval'])

    def test_research_and_adjacent_scopes_not_automated(self):
        for cat,scope in [('AI / ML Research','academic_ai_ml_research'),('Statistician','adjacent_borderline')]:
            result=audit_row(row(ai_primary_category=cat,ai_scope=scope))
            self.assertIn('special_scope_requires_human_review',result['decision_reasons'])
            self.assertEqual(result['candidate_disposition'],'human_review')

    def test_restricted_opportunity_and_title_excluded(self):
        result=audit_row(row(ai_opportunity_types=['internship']))
        self.assertEqual(result['candidate_disposition'],'excluded_by_policy')
        r=row();r['job_title']='PhD Data Engineer'
        # Fix title-based saved span metadata to match changed source.
        result=audit_row(r)
        self.assertEqual(result['candidate_disposition'],'excluded_by_policy')

    def test_corrupt_source_evidence_is_technical_failure(self):
        r=row();r['ai_evidence_spans'][0]['text']='invented'
        result=audit_row(r)
        self.assertEqual(result['candidate_disposition'],'technical_failure')
        self.assertFalse(result['automatic_approval'])

    def test_failed_requests_stay_technical_without_retry(self):
        for status in ['invalid_model_response','request_outcome_unknown']:
            r=row();r['ai_status']=status
            with patch('requests.sessions.Session.request',side_effect=AssertionError('No network')):
                result=audit_row(r)
            self.assertEqual(result['candidate_disposition'],'technical_failure')

    def test_quotation_and_confidence_alone_never_suffice(self):
        result=audit_row(row('We are excited about AI. Python experience is preferred.'))
        self.assertEqual(result['candidate_disposition'],'human_review')
        self.assertFalse(result['automatic_approval'])


if __name__=='__main__':unittest.main()
