"""v3.1 proposal: source duties, attribution, negative controls and frozen v3 regression."""
import copy
import unittest
from unittest.mock import patch

from src.duties_audit_v31 import decision,evidence
from tests.test_preview_duties_v3 import row


class DutiesV31Tests(unittest.TestCase):
    def test_english_duties_and_headings(self):
        for heading in ['What You Will Work On','Responsibilities','What you will be doing','Your role']:
            for duty in ['Build and maintain data pipelines.', 'Migrate data pipelines into lakehouse layers.',
                         'Develop ETL/ELT workflows.', 'Orchestrate ETL pipelines with Airflow.',
                         'Develop a data warehouse.', 'Maintain our data platform.',
                         'Build PySpark jobs for data processing.', 'Develop Databricks workflows.',
                         'Implement data ingestion from external sources.',
                         'Transform data for analytics.', 'Operate production data pipelines.']:
                with self.subTest(heading=heading,duty=duty):
                    self.assertTrue(evidence(heading+'\n'+duty))

    def test_swedish_duties(self):
        for duty in ['Bygga och underhålla datapipelines.', 'Migrera datapipelines till en lakehouse.',
                     'Utveckla och orkestrera ETL-flöden.', 'Utveckling av dataplattformar.',
                     'Design och utveckling av datapipelines och dataprodukter.',
                     'Integrera data från källsystem till datalagret.',
                     'Förvalta och övervaka dataflöden i Databricks.',
                     'Bearbeta data för analys.', 'Felsöka och drifta datapipelines.']:
            with self.subTest(duty=duty):self.assertTrue(evidence('Arbetsuppgifter\n'+duty))

    def test_nonstandard_and_concatenated_headings(self):
        for body in ['Vad du kommer arbeta med\nBygga datapipelines.',
                     'Dina huvudsakliga arbetsuppgifter omfattar bland annat:\nUtveckling av datapipelines.',
                     'ArbetsuppgifterDu utvecklar datapipelines.']:
            self.assertTrue(evidence(body),body)

    def test_direct_employee_statements_without_heading(self):
        for body in ['You will migrate data pipelines.',
                     'Du utvecklar och förvaltar dataplattformar.',
                     'Som dataingenjör bygger du datapipelines.']:
            self.assertTrue(evidence(body))

    def test_coordinated_object(self):
        self.assertTrue(evidence('You will design and maintain data models and pipelines.'))

    def test_exact_unicode_offsets_heading_and_duties(self):
        body='Åäö.\nArbetsuppgifter\nDu utvecklar datapipelines.'
        hits=evidence(body);self.assertTrue(hits)
        for h in hits:
            for a,b,key in [('start','end','text'),('duty_start','duty_end','duty_text')]:
                self.assertEqual(body[h[a]:h[b]],h[key])
            if h['heading']:
                heading=h['heading'];self.assertEqual(body[heading['start']:heading['end']],heading['text'])

    def test_tools_and_generic_software_are_insufficient(self):
        for body in ['You will develop SQL backend services on Azure.',
                     'Responsibilities\nBuild ASP.NET applications with SQL Server databases.',
                     'Arbetsuppgifter\nUtveckla API och integrationer mellan SAP och andra system.',
                     'Responsibilities\nMaintain SAP and EDI integration workflows.',
                     'You will develop MySQL-backed PHP and Laravel applications.',
                     'Responsibilities\nBuild CI/CD pipelines and manage Azure databases.',
                     'You will build APIs for a data platform.',
                     'You will create training about data pipelines.',
                     'You will manage a physical warehouse.',
                     'You will build data warehouse dashboards.',
                     'You will use SQL, PySpark and Databricks.']:
            with self.subTest(body=body):self.assertFalse(evidence(body))

    def test_qualification_only_and_optional_negative_no_duties(self):
        for body in ['Experience building data pipelines.',
                     'You have experience in developing data platforms.',
                     'You may build data pipelines.', "You won't build data pipelines.",
                     'Du har erfarenhet av att bygga datapipelines.',
                     'Vill du bygga datapipelines?',
                     'Du kommer inte att bygga datapipelines.']:
            with self.subTest(body=body):self.assertFalse(evidence(body))

    def test_qualifications_benefits_and_company_sections_end_duties(self):
        for heading in ['Requirements','Qualifications','What You Bring','Who you are','Din profil','Krav','Meriterande','About the Company','What we offer']:
            body='Responsibilities\nAttend meetings.\n'+heading+'\nBuild data pipelines.'
            with self.subTest(heading=heading):self.assertFalse(evidence(body))

    def test_third_party_duties_and_branding_are_not_employee_duties(self):
        for body in ['Our company builds data pipelines.',
                     'Responsibilities\nOur clients build data pipelines.',
                     'Responsibilities of our colleagues:\nBuild data pipelines.',
                     'You will work with data engineers who build data pipelines.',
                     'Our services\nWe build data platforms for our clients.']:
            with self.subTest(body=body):self.assertFalse(evidence(body))

    def test_narrow_or_nonoverlapping_model_excerpt_preserves_both_evidence_types(self):
        body='We offer a role. '+'x'*250+'. Responsibilities\nMigrate data pipelines into the lakehouse.'
        r=row(body);before=copy.deepcopy(r)
        d=decision(r)
        self.assertEqual(d['preview_decision'],'would_qualify_v31')
        self.assertTrue(d['uses_full_source_beyond_selected_spans'])
        self.assertEqual(d['original_selected_evidence'],r['ai_evidence_spans'])
        self.assertEqual(r,before);self.assertFalse(d['approved']);self.assertFalse(d['activated'])

    def test_corrupt_saved_evidence_cannot_be_rescued(self):
        r=row();r['ai_evidence_spans'][0]['text']='invented'
        d=decision(r)
        self.assertEqual(d['preview_decision'],'technical_failure')
        self.assertFalse(d['approved'])

    def test_review_flags_and_special_scopes_not_resolved(self):
        for changes in [dict(ai_review_required=True),dict(ai_confidence='medium'),dict(ai_multi_role=True),
                        dict(ai_opportunity_types=['internship']),dict(ai_scope='adjacent_borderline')]:
            d=decision(row(**changes))
            self.assertEqual(d['preview_decision'],'still_abstained')
            self.assertFalse(d['approved'])

    def test_legacy_not_silently_approved(self):
        r=row();r['ai_prompt_version']='step3f-1.0'
        d=decision(r)
        self.assertEqual(d['preview_decision'],'still_abstained')
        self.assertIn('legacy_classifier_not_compatible',d['reasons'])

    def test_previously_valid_v3_route_is_retained_without_id_allowlist(self):
        d=decision(row())
        self.assertEqual(d['preview_decision'],'would_qualify_v31')
        self.assertTrue(d['retains_frozen_v3_route'])

    def test_no_new_categories_or_uncertain_labels(self):
        r=row(ai_primary_category='AI Engineer')
        self.assertEqual(decision(r)['preview_decision'],'outside_DE_proposal')
        r=row(ai_relevance='uncertain',ai_primary_category=None,ai_scope='unresolved')
        self.assertEqual(decision(r)['preview_decision'],'outside_DE_proposal')

    def test_technical_failures_not_retried(self):
        for status in ['invalid_model_response','request_outcome_unknown']:
            r=row();r['ai_status']=status
            with patch('requests.sessions.Session.request',side_effect=AssertionError('No network')):
                d=decision(r)
            self.assertEqual(d['preview_decision'],'technical_failure')

if __name__=='__main__':unittest.main()
