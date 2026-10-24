"""Mock-only network tests. Fake credentials never leave the mocked session."""
import io
import json
import os
import shutil
import unittest
import uuid
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

from src.llm_classifier import (AI_FIELDS, BudgetExceeded, ClassificationError, Config,
    LLMClassifier, atomic_json, strict_json, validate_prediction, cost_usd,
    MAX_EVIDENCE_CHARACTERS, PROMPT_VERSION, SCHEMA)
from src.llm_classifier import evidence_match, VALIDATION_VERSION
from src.run_ai_classification import load_suggestions, run, select_pilot
from src.prepare_ai_retry import prepare_retry
from src.combine_ai_pilot import combine


TITLE = 'Data Engineer'
BODY = 'You will build data pipelines. Full original text remains intact.'
POLICY = 'Use actual duties. Do not infer relevance from incidental AI tools.'


def prediction(**changes):
    value = dict(ai_relevance='relevant', ai_primary_category='Data Engineer',
        ai_secondary_categories=[], ai_reasoning='The role builds data pipelines.',
        ai_evidence='build data pipelines', ai_confidence='high', ai_review_required=False)
    value.update(changes)
    return value


def response(value=None, status=200, usage=None, finish='stop', refusal=None):
    result = Mock()
    result.status_code = status
    result.headers = {}
    body = dict(model='mock-model-snapshot', choices=[dict(finish_reason=finish,
        message=dict(content=json.dumps(prediction() if value is None else value), refusal=refusal))])
    if usage is not False:
        body['usage'] = dict(prompt_tokens=120, completion_tokens=80) if usage is None else usage
    result.json.return_value = body
    return result


class Step3DLLMTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path(__file__).parent/('.test-step3d-'+uuid.uuid4().hex)
        self.temp.mkdir()
        self.addCleanup(lambda:shutil.rmtree(self.temp))
        self.config = Config(model='mock-model',input_usd_per_million=.4,output_usd_per_million=1.6,
                             api_key_env='STEP3D_TEST_KEY',min_interval=0)
        self.environment = patch.dict(os.environ,{'STEP3D_TEST_KEY':'not-a-real-key'})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def client(self, session=None, **changes):
        args = dict(config=self.config,policy_text=POLICY,cache_dir=self.temp/'cache',
                    checkpoint_path=self.temp/'checkpoint.json',budget_usd=1,
                    execute=True,session=session or Mock(),sleep=lambda seconds:None)
        args.update(changes)
        return LLMClassifier(**args)

    def test_valid_structured_response_and_full_input(self):
        session = Mock(); session.post.return_value=response()
        client=self.client(session)
        result=client.classify('001',TITLE,BODY)
        self.assertEqual(result['ai_relevance'],'relevant')
        self.assertEqual(result['ai_model_returned'],'mock-model-snapshot')
        self.assertFalse(result['ai_human_verified'])
        sent=session.post.call_args.kwargs
        payload=sent['json']
        self.assertEqual(json.loads(payload['messages'][1]['content'])['description_text'],BODY)
        self.assertTrue(payload['response_format']['json_schema']['strict'])
        self.assertEqual(payload['max_completion_tokens'],1024)
        self.assertEqual(sent['timeout'],(10,60))
        self.assertFalse(sent['allow_redirects'])
        self.assertAlmostEqual(client.spent,cost_usd(120,80,self.config))

    def test_schema_rejects_incomplete_extra_wrong_type_unknown_category(self):
        cases=[]
        missing=prediction();missing.pop('ai_reasoning');cases.append(missing)
        cases += [prediction(ai_confidence='certain'),prediction(ai_review_required='true'),
                  prediction(ai_primary_category='AI Wizard'),prediction(ai_secondary_categories=['Unknown']),
                  dict(prediction(),human_verified=True),prediction(ai_reasoning=''),
                  prediction(ai_secondary_categories=['Data Engineer'])]
        for value in cases:
            with self.subTest(value=value),self.assertRaises(ClassificationError):
                validate_prediction(value,TITLE,BODY)

    def test_evidence_must_come_from_original_ad(self):
        with self.assertRaises(ClassificationError):
            validate_prediction(prediction(ai_evidence='Invented model training duties'),TITLE,BODY)

    def test_irrelevant_swedish_title_is_sufficient_evidence(self):
        title = 'Specialpedagog till Hagaskolan'
        value = prediction(ai_relevance='irrelevant',ai_primary_category=None,
            ai_evidence=title,ai_reasoning='The advertised occupation is a special education teacher.')
        self.assertEqual(validate_prediction(value,title,'Arbete med elever och pedagogiskt stöd.'),value)

    def test_irrelevant_requires_unrelated_evidence_not_just_missing_keywords(self):
        for evidence in [None,'','   ',[],42,'No Data or AI duties are mentioned']:
            with self.subTest(evidence=evidence),self.assertRaises(ClassificationError):
                validate_prediction(prediction(ai_relevance='irrelevant',ai_primary_category=None,
                    ai_evidence=evidence,ai_reasoning='No Data/AI keywords.'),'Specialpedagog','Arbete med elever.')

    def test_swedish_exact_matching_preserves_line_breaks_and_unicode(self):
        body = 'Arbeta på skolan.\nFörbättra lärmiljön för våra elever.\nVi har 8\u202f000 medarbetare.'
        for excerpt in ['på skolan','Förbättra lärmiljön','8\u202f000 medarbetare',
                        'på skolan.\nFörbättra lärmiljön']:
            self.assertEqual(validate_prediction(prediction(ai_evidence=excerpt),TITLE,body)['ai_evidence'],excerpt)
        for excerpt in ['på skolan. Förbättra lärmiljön','8 000 medarbetare',
                        'p\u00c3\u00a5 skolan','forbattra larmiljon','Förbättra ... elever']:
            with self.subTest(excerpt=excerpt),self.assertRaises(ClassificationError):
                validate_prediction(prediction(ai_evidence=excerpt),TITLE,body)

    def test_short_evidence_limit_matches_prompt_and_schema(self):
        self.assertEqual(MAX_EVIDENCE_CHARACTERS,160)
        self.assertEqual(SCHEMA['properties']['ai_evidence']['maxLength'],160)
        self.assertEqual(PROMPT_VERSION,'step3d-1.3')
        client = self.client()
        self.assertIn('at most 160 characters',client.system_prompt)
        self.assertIn('prefer copying the job title',client.system_prompt)
        validate_prediction(prediction(ai_evidence='a'*160),TITLE,'a'*161)
        with self.assertRaises(ClassificationError):
            validate_prediction(prediction(ai_evidence='a'*161),TITLE,'a'*161)

    def test_utf8_response_overrides_misleading_charset(self):
        title = 'Specialpedagog på Hagaskolan'
        value = prediction(ai_relevance='irrelevant',ai_primary_category=None,ai_evidence=title,
            ai_reasoning='Läraren ska förbättra lärmiljön för elever.')
        reply = requests.Response();reply.status_code = 200
        reply.headers['Content-Type'] = 'application/json; charset=iso-8859-1'
        reply.encoding = 'iso-8859-1'
        body = {'model':'mock-swedish','choices':[{'finish_reason':'stop','message':
                {'content':json.dumps(value,ensure_ascii=False),'refusal':None}}],
                'usage':{'prompt_tokens':120,'completion_tokens':80}}
        reply._content = json.dumps(body,ensure_ascii=False).encode('utf-8')
        session = Mock();session.post.return_value = reply
        client = self.client(session)
        result = client.classify('1',title,'Du ska förbättra skolans lärmiljö.')
        self.assertEqual(result['ai_status'],'completed')
        self.assertEqual(result['ai_evidence'],title)
        saved = (self.temp/'checkpoint.json').read_bytes()
        self.assertIn('förbättra'.encode('utf-8'),saved)
        self.assertNotIn('f\u00c3\u00b6rb\u00c3\u00a4ttra'.encode('utf-8'),saved)
        self.assertEqual(json.loads(saved.decode('utf-8'))['results']['1']['ai_evidence'],title)

    def test_observed_pilot_evidence_mutations_remain_rejected(self):
        cases = [
            ('Data Architects','data Observability stack\nLinux/UNIX competence','data Observability stack Linux/UNIX competence'),
            ('ILS-ingenjör','Din roll som ILS-ingenjör\nDu arbetar med drift','Din roll som ILS-ingenjör ... arbeta med drift'),
            ('VVS-ingenjör','VVS installationer','job_title: VVS-ingenjör'),
            ('DevOps Ingenjör','automatiserade leveransflöden','automatiserade leveransflö-'),
            ('Postdoktor','Som postdoktor kommer du att arbeta med maskininlärning','"Som postdoktor kommer du att arbeta'),
            ('Hej Senior VA-ingenjör!','Det här är en roll för dig','Hej Senior VA-ingenjör! Det här är en roll för dig'),
            ('Backendutvecklare','Arbeta hands-on med C# och .NET\nVara med i arkitekturen','Arbeta hands-on med C# och .NET. Vara med i arkitekturen')]
        for title, text, excerpt in cases:
            with self.subTest(title=title),self.assertRaisesRegex(ClassificationError,'not a verbatim excerpt'):
                validate_prediction(prediction(ai_evidence=excerpt),title,text)

    def test_remaining_retry_paraphrases_rejected_but_exact_spans_allowed(self):
        cases = [
            ('Working knowledge of data modelling,data Warehouse methodologies',
             'Working knowledge in data modelling,data Warehouse methodologies','data Warehouse methodologies'),
            ('Maskininlärningsmetoder kommer att användas för att identifiera samband mellan struktur och egenskaper',
             'maskininlärning för att identifiera samband mellan struktur och egenskaper',
             'identifiera samband mellan struktur och egenskaper')]
        for text,bad,good in cases:
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ClassificationError,'not a verbatim excerpt'):
                    validate_prediction(prediction(ai_evidence=bad),TITLE,text)
                self.assertEqual(validate_prediction(prediction(ai_evidence=good),TITLE,text)['ai_evidence'],good)

    def test_retry_prompt_addresses_observed_formatting_and_role_confusion(self):
        prompt = self.client().system_prompt
        for instruction in ['never add a job_title: label','Never concatenate the title',
                            'choose another shorter source span','ORM/entity modeling',
                            'AI-assisted recruiting tools alone','require review']:
            self.assertIn(instruction,prompt)

    def test_invalid_swedish_evidence_keeps_utf8_diagnostic(self):
        value = prediction(ai_evidence='på skolan! Förbättra lärmiljön')
        session = Mock();session.post.return_value = response(value)
        client = self.client(session,config=replace(self.config,max_retries=0))
        self.assertEqual(client.classify('1',TITLE,'på skolan.\nFörbättra lärmiljön')['ai_status'],
                         'invalid_model_response')
        saved = json.loads((self.temp/'checkpoint.json').read_text(encoding='utf-8'))
        diagnostic = saved['ledger'][0]['model_response_diagnostic']
        self.assertEqual(diagnostic['validation_stage'],'prediction_validation')
        self.assertEqual(json.loads(diagnostic['content'])['ai_evidence'],value['ai_evidence'])
        self.assertIn('not a verbatim excerpt',diagnostic['validation_error'])

    def test_whitespace_only_match_has_exact_auditable_source_span(self):
        text='Before. Data Architects/Designers\r\n\tBachelor or Master Degree in IT. After.'
        quote='Data Architects/Designers Bachelor or Master Degree in IT'
        match=evidence_match(quote,TITLE,text,allow_formatting=True)
        self.assertEqual(match['method'],'whitespace_only')
        self.assertEqual(match['source_field'],'description_text')
        self.assertEqual(text[match['start']:match['end']],match['source_excerpt'])
        self.assertIn('\r\n\t',match['source_excerpt'])
        self.assertEqual(validate_prediction(prediction(ai_evidence=quote),TITLE,text,
                         allow_formatting=True)['ai_evidence'],quote)
        self.assertTrue(validate_prediction(prediction(ai_evidence=quote),TITLE,text,
                         allow_formatting=True)['ai_review_required'])
        with self.assertRaises(ClassificationError):evidence_match(quote,TITLE,text)

    def test_unicode_spaces_are_allowed_but_not_unicode_word_changes(self):
        text='Vi har 8\u202f000 elever på skolan.'
        self.assertEqual(evidence_match('8 000 elever på skolan',TITLE,text,True)['method'],'whitespace_only')
        for quote in ['8000 elever på skolan','8 000 elever pa skolan','8 000 elever p\u00c3\u00a5 skolan']:
            with self.subTest(quote=quote),self.assertRaises(ClassificationError):
                evidence_match(quote,TITLE,text,True)

    def test_formatting_normalization_never_accepts_paraphrases_or_joins_fields(self):
        for title,text,quote in [
            (TITLE,'Working knowledge of data modelling','Working knowledge in data modelling'),
            (TITLE,'Maskininlärningsmetoder kommer att användas för att identifiera samband',
             'maskininlärning för att identifiera samband'),
            ('Data Architects/Designers','Bachelor or Master Degree','Data Architects/Designers Bachelor or Master Degree'),
            (TITLE,'build reliable data pipelines','build data pipelines'),
            (TITLE,'build reliable data pipelines','build ... pipelines'),
            (TITLE,'build reliable data pipelines','Build reliable data pipelines'),
            (TITLE,'build reliable data pipelines','"build reliable data pipelines"')]:
            with self.subTest(quote=quote),self.assertRaises(ClassificationError):
                evidence_match(quote,title,text,True)

    def test_formatted_evidence_metadata_survives_cache_and_checkpoint_without_repay(self):
        quote='build data pipelines';text='You build\n\tdata pipelines.'
        session=Mock();session.post.return_value=response(prediction(ai_evidence=quote))
        client=self.client(session)
        result=client.classify('1',TITLE,text)
        self.assertEqual(result['ai_status'],'completed')
        self.assertEqual(result['ai_evidence'],quote)
        self.assertEqual(result['ai_evidence_method'],'whitespace_only')
        self.assertEqual(result['ai_evidence_source_excerpt'],'build\n\tdata pipelines')
        self.assertEqual(result['ai_validation_version'],VALIDATION_VERSION)
        self.assertTrue(result['ai_review_required'])
        no_api=Mock()
        cached=self.client(no_api,checkpoint_path=self.temp/'new-checkpoint.json').classify('2',TITLE,text)
        self.assertEqual(cached['ai_status'],'cache_hit')
        self.assertEqual(cached['ai_evidence_source_excerpt'],result['ai_evidence_source_excerpt'])
        resumed=self.client(no_api).classify('1',TITLE,text)
        self.assertEqual(resumed['ai_evidence_method'],'whitespace_only')
        no_api.post.assert_not_called()

    def test_validation_policy_participates_in_request_identity(self):
        client=self.client()
        original_key=client.cache_key(TITLE,BODY)
        with patch('src.llm_classifier.VALIDATION_VERSION','other-policy'):
            self.assertNotEqual(client.cache_key(TITLE,BODY),original_key)

    def test_uncertainty_and_special_category_validation(self):
        result=validate_prediction(prediction(ai_relevance='uncertain',ai_primary_category=None,
            ai_evidence='',ai_confidence='low',ai_review_required=False),TITLE,BODY)
        self.assertTrue(result['ai_review_required'])
        for category in ['AI / ML Research','Statistician','Data / AI Management','Data Governance / Data Quality']:
            self.assertEqual(validate_prediction(prediction(ai_primary_category=category),TITLE,BODY)['ai_primary_category'],category)
        with self.assertRaises(ClassificationError):
            validate_prediction(prediction(ai_relevance='irrelevant'),TITLE,BODY)

    def test_json_duplicate_keys_and_nonstandard_numbers_rejected(self):
        for content in ['{"x":1,"x":2}','{"ai_confidence":NaN}','not json']:
            with self.subTest(content=content),self.assertRaises(ClassificationError):strict_json(content)

    def test_dry_run_never_sends_request_or_requires_key(self):
        session=Mock()
        with patch.dict(os.environ,{},clear=True):
            result=self.client(session,execute=False).classify('1',TITLE,BODY)
        self.assertEqual(result['ai_status'],'dry_run')
        session.post.assert_not_called()
        self.assertFalse((self.temp/'checkpoint.json').exists())

    def test_missing_description_skips_network(self):
        session=Mock()
        result=self.client(session).classify('1',TITLE,'')
        self.assertEqual(result['ai_status'],'missing_description')
        self.assertEqual(result['ai_relevance'],'uncertain')
        self.assertTrue(result['ai_review_required'])
        session.post.assert_not_called()

    def test_budget_blocks_before_first_request(self):
        session=Mock()
        client=self.client(session,budget_usd=.000001)
        with self.assertRaises(BudgetExceeded):client.classify('1',TITLE,BODY)
        session.post.assert_not_called()
        self.assertEqual(client.spent,0)

    def test_rate_limit_retry_honors_retry_after_and_accounts_all_attempts(self):
        session=Mock(); limited=response(status=429,usage=False);limited.headers={'Retry-After':'2'}
        session.post.side_effect=[limited,response()]
        sleeps=[]
        client=self.client(session,sleep=sleeps.append)
        result=client.classify('1',TITLE,BODY)
        self.assertEqual(result['ai_status'],'completed')
        self.assertEqual(session.post.call_count,2)
        self.assertIn(2,sleeps)
        self.assertEqual(len(client.state['ledger']),2)
        self.assertGreater(client.spent,cost_usd(120,80,self.config))

    def test_retry_stops_when_next_attempt_exceeds_budget(self):
        session=Mock();session.post.return_value=response(status=429,usage=False)
        probe=self.client(session)
        cap=probe.estimate(TITLE,BODY)['reserved_cost_per_attempt_usd']
        client=self.client(session,budget_usd=cap)
        with self.assertRaises(BudgetExceeded):client.classify('1',TITLE,BODY)
        self.assertEqual(session.post.call_count,1)
        self.assertAlmostEqual(client.spent,cap)

    def test_permanent_failure_is_sanitized_and_not_retried(self):
        session=Mock();session.post.return_value=response(status=401,usage=False)
        client=self.client(session)
        result=client.classify('1',TITLE,BODY)
        self.assertEqual(result['ai_status'],'http_401')
        self.assertIsNone(result['ai_relevance'])
        self.assertEqual(session.post.call_count,1)
        self.assertNotIn('not-a-real-key',(self.temp/'checkpoint.json').read_text())

    def test_invalid_response_can_retry_with_usage_charged(self):
        session=Mock();session.post.side_effect=[response({'ai_relevance':'relevant'}),response()]
        client=self.client(session)
        result=client.classify('1',TITLE,BODY)
        self.assertEqual(result['ai_status'],'completed')
        self.assertEqual(len(client.state['ledger']),2)
        self.assertAlmostEqual(client.spent,2*cost_usd(120,80,self.config))

    def test_auth_error_details_redact_credentials_and_email(self):
        session = Mock()
        reply = response(status=401, usage=False)
        reply.json.return_value = {'error': {'type':'invalid_request_error',
            'code':'invalid_api_key', 'message':'Incorrect key not-a-real-key sk-secret123 for person@example.com Bearer abc123'}}
        reply.headers = {'x-request-id':'req-example', 'Authorization':'not-a-real-key'}
        session.post.return_value = reply
        client = self.client(session)
        client.classify('1', TITLE, BODY)
        saved = (self.temp/'checkpoint.json').read_text()
        for secret in ['not-a-real-key','sk-secret123','person@example.com','abc123','Authorization']:
            self.assertNotIn(secret, saved)
        error = client.state['ledger'][0]['api_error']
        self.assertEqual(error['code'], 'invalid_api_key')
        self.assertEqual(error['request_id'], 'req-example')
        self.assertIn('[REDACTED]', error['message'])
        self.assertEqual(client.cost_breakdown()['provider_reported_usage_estimated_cost_usd'], 0)
        self.assertEqual(client.cost_breakdown()['unreported_usage_reserved_usd'], client.spent)

    def test_reported_usage_separate_from_unreported_reservation(self):
        session = Mock(); session.post.return_value = response()
        client = self.client(session)
        client.classify('1', TITLE, BODY)
        costs = client.cost_breakdown()
        self.assertAlmostEqual(costs['provider_reported_usage_estimated_cost_usd'], cost_usd(120,80,self.config))
        self.assertEqual(costs['unreported_usage_reserved_usd'],0)

    def test_non_json_auth_response_retains_status_without_raw_body(self):
        session = Mock(); reply = response(status=401, usage=False)
        reply.json.side_effect = ValueError('Sensitive raw body')
        session.post.return_value = reply
        client = self.client(session)
        self.assertEqual(client.classify('1',TITLE,BODY)['ai_status'],'http_401')
        self.assertIsNone(client.state['ledger'][0]['api_error']['message'])
        self.assertNotIn('Sensitive raw body', (self.temp/'checkpoint.json').read_text())

    def test_incomplete_or_refused_responses_not_saved_as_valid(self):
        for incomplete in [response(finish='length'),response(refusal='No output')]:
            session=Mock();session.post.return_value=incomplete
            client=self.client(session,config=replace(self.config,max_retries=0))
            result=client.classify(str(uuid.uuid4()),TITLE,BODY)
            self.assertEqual(result['ai_status'],'invalid_model_response')

    def test_invalid_evidence_keeps_exact_error_and_generated_output(self):
        rejected = prediction(ai_evidence='invented evidence')
        session = Mock(); session.post.return_value = response(rejected)
        client = self.client(session,config=replace(self.config,max_retries=0))
        result = client.classify('1',TITLE,BODY)
        entry = client.state['ledger'][0]
        diagnostic = entry['model_response_diagnostic']
        self.assertEqual(entry['http_status'],200)
        self.assertEqual(diagnostic['validation_stage'],'prediction_validation')
        self.assertEqual(diagnostic['validation_error'],'Evidence is not a verbatim excerpt of title/description')
        self.assertEqual(json.loads(diagnostic['content']),rejected)
        self.assertEqual(result['ai_status'],'invalid_model_response')
        self.assertIsNone(result['ai_relevance'])
        self.assertFalse(list((self.temp/'cache').glob('*.json')))
        self.assertEqual(entry['usage_basis'],'provider_reported_tokens')
        self.assertAlmostEqual(client.spent,cost_usd(120,80,self.config))

    def test_malformed_json_keeps_parsing_diagnostic(self):
        session = Mock(); reply = response()
        reply.json.return_value['choices'][0]['message']['content'] = '{broken'
        session.post.return_value = reply
        client = self.client(session,config=replace(self.config,max_retries=0))
        client.classify('1',TITLE,BODY)
        diagnostic = client.state['ledger'][0]['model_response_diagnostic']
        self.assertEqual(diagnostic['validation_stage'],'json_parsing')
        self.assertEqual(diagnostic['content'],'{broken')
        self.assertEqual(diagnostic['validation_error'],'Invalid structured JSON response')

    def test_refusal_and_truncation_have_distinct_diagnostics(self):
        for reply, expected in [(response(finish='length'),'Response did not finish with stop'),
                                (response(refusal='Cannot comply'),'Model refused the request')]:
            with self.subTest(expected=expected):
                session = Mock(); session.post.return_value = reply
                client = self.client(session,config=replace(self.config,max_retries=0))
                client.classify(uuid.uuid4().hex,TITLE,BODY)
                diagnostic = client.state['ledger'][-1]['model_response_diagnostic']
                self.assertEqual(diagnostic['validation_stage'],'completion_status')
                self.assertEqual(diagnostic['validation_error'],expected)

    def test_model_diagnostics_redact_before_truncation(self):
        session = Mock(); reply = response()
        reply.json.return_value['choices'][0]['message']['content'] = 'not-a-real-key sk-secret123 person@example.com ' + 'x'*17000
        reply.headers = {'x-request-id':'req-test', 'Authorization':'secret-header'}
        session.post.return_value = reply
        client = self.client(session,config=replace(self.config,max_retries=0))
        client.classify('1',TITLE,BODY)
        saved = (self.temp/'checkpoint.json').read_text()
        for secret in ['not-a-real-key','sk-secret123','person@example.com','secret-header','Authorization']:
            self.assertNotIn(secret,saved)
        diagnostic = client.state['ledger'][0]['model_response_diagnostic']
        self.assertTrue(diagnostic['content_truncated'])
        self.assertLessEqual(len(diagnostic['content']),16000)
        self.assertEqual(diagnostic['request_id'],'req-test')

    def test_malformed_response_envelope_diagnosed_without_crash(self):
        for body in [[], {'choices':[]}, {'choices':[None]}, {'choices':[{'message':[]}]}]:
            with self.subTest(body=body):
                session = Mock(); reply = response(); reply.json.return_value = body
                session.post.return_value = reply
                client = self.client(session,config=replace(self.config,max_retries=0))
                self.assertEqual(client.classify(uuid.uuid4().hex,TITLE,BODY)['ai_status'],'invalid_model_response')
                self.assertEqual(client.state['ledger'][-1]['model_response_diagnostic']['validation_stage'],'response_envelope')

    def test_failed_diagnostic_checkpoint_resume_never_repays(self):
        session = Mock(); session.post.return_value = response(prediction(ai_evidence='invented'))
        client = self.client(session,config=replace(self.config,max_retries=0))
        client.classify('1',TITLE,BODY)
        before = (self.temp/'checkpoint.json').read_bytes()
        second = Mock(); resumed = self.client(second,config=replace(self.config,max_retries=0))
        self.assertEqual(resumed.classify('1',TITLE,BODY)['ai_status'],'invalid_model_response')
        second.post.assert_not_called()
        self.assertEqual((self.temp/'checkpoint.json').read_bytes(),before)

    def test_timeout_does_not_automatically_retry_or_pay_again_on_resume(self):
        session=Mock();session.post.side_effect=requests.Timeout('ambiguous provider result')
        client=self.client(session)
        result=client.classify('1',TITLE,BODY)
        self.assertEqual(result['ai_status'],'request_outcome_unknown')
        self.assertEqual(session.post.call_count,1)
        second=Mock()
        resumed=self.client(second)
        self.assertEqual(resumed.classify('1',TITLE,BODY)['ai_status'],'request_outcome_unknown')
        second.post.assert_not_called()
        self.assertGreater(resumed.spent,0)

    def test_completed_checkpoint_and_shared_cache_avoid_new_requests(self):
        session=Mock();session.post.return_value=response()
        self.client(session).classify('1',TITLE,BODY)
        second=Mock()
        resumed=self.client(second)
        self.assertEqual(resumed.classify('1',TITLE,BODY)['ai_status'],'completed')
        other=self.client(second,checkpoint_path=self.temp/'other.json')
        self.assertEqual(other.classify('different-id',TITLE,BODY)['ai_status'],'cache_hit')
        second.post.assert_not_called()
        self.assertEqual(other.spent,0)

    def test_crash_after_cache_commit_recovers_reservation(self):
        session=Mock();session.post.return_value=response()
        client=self.client(session)
        client.classify('1',TITLE,BODY)
        client.state['results']={}
        receipt=client.state['ledger'][0]
        receipt['status']='reserved_before_request'
        receipt['accounted_cost_usd']=receipt['reserved_cost_usd']
        client.save()
        resumed=self.client(Mock())
        self.assertEqual(resumed.classify('1',TITLE,BODY)['ai_status'],'cache_hit')
        self.assertAlmostEqual(resumed.spent,cost_usd(120,80,self.config))
        resumed.session.post.assert_not_called()

    def test_unknown_reservation_without_pending_file_blocks_new_requests(self):
        session=Mock();client=self.client(session)
        estimate=client.estimate(TITLE,BODY)
        client.reserve(client.cache_key(TITLE,BODY),'1',estimate)
        result=self.client(session).classify('1',TITLE,BODY)
        self.assertEqual(result['ai_status'],'request_outcome_unknown')
        session.post.assert_not_called()

    def test_cache_identity_changes_with_model_policy_full_text(self):
        client=self.client()
        original=client.cache_key(TITLE,BODY)
        self.assertNotEqual(original,client.cache_key(TITLE,BODY+' More text.'))
        self.assertNotEqual(original,self.client(config=replace(self.config,model='different-model')).cache_key(TITLE,BODY))
        self.assertNotEqual(original,self.client(policy_text=POLICY+' new inclusion policy').cache_key(TITLE,BODY))

    def test_unreported_usage_keeps_full_reservation(self):
        session=Mock();session.post.return_value=response(usage=False)
        client=self.client(session)
        reserve=client.estimate(TITLE,BODY)['reserved_cost_per_attempt_usd']
        result=client.classify('1',TITLE,BODY)
        self.assertAlmostEqual(client.spent,reserve)
        self.assertIsNone(result['ai_input_tokens'])
        self.assertEqual(result['ai_token_usage_basis'],'conservative_reservation')

    def test_invalid_usage_cannot_understate_cost_or_claim_token_counts(self):
        session=Mock();session.post.return_value=response(usage={'prompt_tokens':True,'completion_tokens':-10})
        client=self.client(session)
        result=client.classify('1',TITLE,BODY)
        self.assertIsNone(result['ai_input_tokens'])
        self.assertAlmostEqual(client.spent,client.estimate(TITLE,BODY)['reserved_cost_per_attempt_usd'])

    def test_corrupt_checkpoint_cost_blocks_requests(self):
        client=self.client()
        client.reserve(client.cache_key(TITLE,BODY),'1',client.estimate(TITLE,BODY))
        client.state['ledger'][0]['accounted_cost_usd']=-1
        client.save()
        with self.assertRaises(ClassificationError):self.client()

    def test_server_failure_retries_and_compatible_provider_uses_configured_endpoint(self):
        session=Mock();session.post.side_effect=[response(status=503,usage=False),response()]
        config=replace(self.config,provider='openai-compatible',base_url='http://localhost:1234/v1',completion_token_field='max_tokens')
        result=self.client(session,config=config).classify('1',TITLE,BODY)
        self.assertEqual(result['ai_status'],'completed')
        self.assertEqual(session.post.call_count,2)
        self.assertEqual(session.post.call_args.args[0],'http://localhost:1234/v1/chat/completions')
        self.assertIn('max_tokens',session.post.call_args.kwargs['json'])

    def test_full_description_limit_does_not_truncate_or_send(self):
        session=Mock()
        result=self.client(session,config=replace(self.config,max_input_tokens=5)).classify('1',TITLE,BODY)
        self.assertEqual(result['ai_status'],'input_limit_exceeded')
        session.post.assert_not_called()

    def test_config_rejects_secret_urls_and_bad_prices(self):
        for changes in [dict(base_url='https://name:secret@api.openai.com/v1'),
                        dict(input_usd_per_million=-1),dict(output_usd_per_million=float('nan')),
                        dict(provider='unknown'),dict(base_url='https://other.example/v1')]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):replace(self.config,**changes)


class Step3DPilotTests(unittest.TestCase):
    def setUp(self):
        self.temp=Path(__file__).parent/('.test-step3d-'+uuid.uuid4().hex)
        self.temp.mkdir()
        self.addCleanup(lambda:shutil.rmtree(self.temp))
        self.config=Config(model='mock-model',input_usd_per_million=.4,output_usd_per_million=1.6,
                           api_key_env='STEP3D_TEST_KEY',max_retries=0,min_interval=0)
        self.sample=pd.DataFrame([dict(advertisement_id=str(i),job_title=f'Engineer {i}',description_text=BODY,
            relevance=['relevant','irrelevant','uncertain'][i%3],primary_category='Data Engineer' if i%3==0 else '',
            human_relevance='',human_primary_category='',human_verified='',reviewer='',review_notes='') for i in range(60)])
        self.sample_path=self.temp/'representative.csv'
        self.sample.to_csv(self.sample_path,index=False)
        self.policy_path=self.temp/'policy.md';self.policy_path.write_text(POLICY)

    def run_pilot(self,**changes):
        kwargs=dict(config=self.config,budget_usd=1,sample_path=self.sample_path,
            output=self.temp/'output',cache_dir=self.temp/'cache',policy_path=self.policy_path,
            max_ads=6,sleep=lambda seconds:None)
        kwargs.update(changes)
        with redirect_stdout(io.StringIO()):return run(**kwargs)

    def test_pilot_balanced_deterministic_and_full_corpus_rejected(self):
        a=select_pilot(self.sample)
        pd.testing.assert_frame_equal(a,select_pilot(self.sample))
        self.assertEqual(a.relevance.value_counts().to_dict(),{'irrelevant':10,'relevant':10,'uncertain':10})
        with self.assertRaises(ValueError):select_pilot(self.sample,31)
        with self.assertRaises(ValueError):select_pilot(pd.concat([self.sample]*84,ignore_index=True))

    def retry_parent(self):
        session = Mock()
        session.post.side_effect = [response(),response(prediction(ai_evidence='fabricated')),
                                   response(),response(),response(prediction(ai_evidence='job_title: Engineer')),
                                   response()]
        with patch.dict(os.environ,{'STEP3D_TEST_KEY':'fake'}):
            return self.run_pilot(execute=True,session=session)[0]

    def test_prepare_retry_only_failed_preserves_parent_and_successes(self):
        source = self.retry_parent()
        before = {p:p.read_bytes() for p in source.iterdir() if p.is_file()}
        with patch('requests.sessions.Session.request',side_effect=AssertionError('No network allowed')):
            manifest = prepare_retry(source,self.temp/'retry')
        self.assertEqual(len(manifest['successful_ids']),4)
        self.assertEqual(len(manifest['retry_ids']),2)
        subset = pd.read_csv(self.temp/'retry'/'retry_failed_input.csv',dtype=str,keep_default_na=False)
        self.assertEqual(set(subset.advertisement_id),set(manifest['retry_ids']))
        self.assertFalse(set(AI_FIELDS)&set(subset))
        self.assertEqual(len(select_pilot(subset,max_ads=2)),2)
        raw = (source/'pilot_results.jsonl').read_text(encoding='utf-8').splitlines(keepends=True)
        expected = ''.join(line for line in raw if json.loads(line)['advertisement_id'] in manifest['successful_ids'])
        self.assertEqual((self.temp/'retry'/'preserved_successful_results.jsonl').read_text(encoding='utf-8'),expected)
        for path,data in before.items():self.assertEqual(path.read_bytes(),data)

    def combine_fixture(self):
        parent=self.retry_parent()
        manifest=prepare_retry(parent,self.temp/'prepared')
        session=Mock();session.post.side_effect=[response(),response(prediction(ai_evidence='invented'))]
        with patch.dict(os.environ,{'STEP3D_TEST_KEY':'fake'}):
            child,_=self.run_pilot(sample_path=self.temp/'prepared'/'retry_failed_input.csv',
                output=self.temp/'child',max_ads=2,execute=True,session=session)
        return parent,child,manifest

    def test_combination_preserves_predictions_and_null_unresolved_provenance(self):
        parent,child,manifest=self.combine_fixture()
        before={p:p.read_bytes() for d in [parent,child] for p in d.iterdir() if p.is_file()}
        with patch('requests.sessions.Session.request',side_effect=AssertionError('No network')):
            rows,summary=combine(parent,child,self.temp/'combined',expected_count=6)
        self.assertEqual((summary['unique_ids'],summary['accepted'],summary['unresolved']),(6,5,1))
        original={r['advertisement_id']:r for r in map(json.loads,(parent/'pilot_results.jsonl').read_text().splitlines())}
        for row in rows:
            self.assertEqual(row['job_title'],original[row['advertisement_id']]['job_title'])
            self.assertEqual(row['description_text'],original[row['advertisement_id']]['description_text'])
            self.assertTrue(row['ai_source_run'])
            self.assertTrue(row['ai_schema_sha256'])
            self.assertFalse(row['ai_human_verified'])
            if row['advertisement_id'] in manifest['successful_ids']:
                for field in AI_FIELDS:self.assertEqual(row[field],original[row['advertisement_id']][field])
            if row['ai_validation_status']=='unresolved':
                self.assertIsNone(row['ai_relevance']);self.assertIsNone(row['ai_primary_category'])
                self.assertTrue(row['ai_review_required'])
                self.assertIn('verbatim',row['ai_validation_error'])
        self.assertEqual(sum(summary['relevance_counts'].values()),6)
        self.assertEqual(summary['comparison']['model_classified'],5)
        self.assertEqual(summary['source_unreported_usage_reserved_usd'],0)
        for p,data in before.items():self.assertEqual(p.read_bytes(),data)

    def test_combination_rejects_duplicate_results_and_overwrite(self):
        parent,child,_=self.combine_fixture()
        with self.assertRaises(ValueError):combine(parent,child,parent,expected_count=6)
        path=child/'pilot_results.jsonl'
        text=path.read_text();path.write_text(text+text.splitlines()[0]+'\n')
        with self.assertRaisesRegex(ValueError,'unique frozen IDs'):
            combine(parent,child,self.temp/'combined',expected_count=6)
        self.assertFalse((self.temp/'combined').exists())

    def test_combination_refuses_wrong_retry_ids_and_original_size(self):
        parent,child,_=self.combine_fixture()
        with self.assertRaisesRegex(ValueError,'expected pilot size'):
            combine(parent,child,self.temp/'combined',expected_count=30)
        # A valid independent run is still unsafe if its IDs include parent successes.
        session=Mock();session.post.return_value=response()
        with patch.dict(os.environ,{'STEP3D_TEST_KEY':'fake'}):
            wrong,_=self.run_pilot(output=self.temp/'wrong',max_ads=2,execute=True,session=session)
        with self.assertRaisesRegex(ValueError,'exactly the original failed IDs'):
            combine(parent,wrong,self.temp/'combined',expected_count=6)

    def test_combination_supports_chronological_retry_chain_without_changing_successes(self):
        parent,child,_=self.combine_fixture()
        manifest=prepare_retry(child,self.temp/'prepared-final')
        session=Mock();session.post.return_value=response()
        with patch.dict(os.environ,{'STEP3D_TEST_KEY':'fake'}):
            latest,_=self.run_pilot(sample_path=self.temp/'prepared-final'/'retry_failed_input.csv',
                output=self.temp/'latest',max_ads=1,execute=True,session=session)
        rows,summary=combine(parent,child,self.temp/'combined-final',expected_count=6,additional_retries=[latest])
        self.assertEqual((summary['unique_ids'],summary['accepted'],summary['unresolved']),(6,6,0))
        newest=next(r for r in rows if r['advertisement_id']==manifest['retry_ids'][0])
        self.assertEqual(newest['ai_source_run'],str(latest.resolve()))
        self.assertEqual(len(summary['source_runs']),3)
        with self.assertRaisesRegex(ValueError,'exactly the original failed IDs'):
            combine(parent,latest,self.temp/'bad-order',expected_count=6,additional_retries=[child])

    def test_prepare_retry_marks_changed_prompt_without_relabeling_successes(self):
        source = self.retry_parent()
        with patch('src.prepare_ai_retry.PROMPT_VERSION','future-prompt'):
            manifest = prepare_retry(source,self.temp/'retry')
        self.assertFalse(manifest['same_prompt_schema'])
        rows = [json.loads(line) for line in (self.temp/'retry'/'preserved_successful_results.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(all(row['ai_prompt_version']==PROMPT_VERSION for row in rows))

    def test_prepare_retry_refuses_unknown_outcomes_and_existing_output(self):
        source = self.retry_parent()
        existing = self.temp/'exists';existing.mkdir()
        with self.assertRaises(ValueError):prepare_retry(source,existing)
        path = source/'pilot_results.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        next(r for r in rows if r['ai_status']=='invalid_model_response')['ai_status']='request_outcome_unknown'
        path.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Only known invalid_model_response'):
            prepare_retry(source,self.temp/'retry')
        self.assertFalse((self.temp/'retry').exists())

    def test_swedish_csv_to_model_and_output_round_trip(self):
        title = 'Specialpedagog på Hagaskolan'
        text = 'Du ska förbättra lärmiljön.\nArbeta med 8\u202f000 elever och lärare.'
        self.sample['job_title'] = title
        self.sample['description_text'] = text
        self.sample.to_csv(self.sample_path,index=False,encoding='utf-8-sig')
        before = self.sample_path.read_bytes()
        reply = response(prediction(ai_relevance='irrelevant',ai_primary_category=None,ai_evidence=title))
        session = Mock();session.post.return_value = reply
        with patch.dict(os.environ,{'STEP3D_TEST_KEY':'fake'}):
            out,summary = self.run_pilot(execute=True,max_ads=1,session=session)
        self.assertEqual(summary['comparison']['model_classified'],1)
        sent = json.loads(session.post.call_args.kwargs['json']['messages'][1]['content'])
        self.assertEqual(sent,{'job_title':title,'description_text':text})
        exported = pd.read_csv(out/'pilot_results.csv',encoding='utf-8-sig',keep_default_na=False)
        record = json.loads((out/'pilot_results.jsonl').read_text(encoding='utf-8'))
        self.assertEqual(exported.iloc[0]['description_text'],text)
        self.assertEqual(record['ai_evidence'],title)
        self.assertEqual(self.sample_path.read_bytes(),before)

    def test_auth_failure_stops_remaining_ads_and_resume_without_new_calls(self):
        session = Mock(); session.post.return_value = response(status=401,usage=False)
        before = self.sample_path.read_bytes()
        with patch.dict(os.environ, {'STEP3D_TEST_KEY':'fake-key'}):
            out, summary = self.run_pilot(execute=True, session=session)
            self.assertEqual(summary['status'],'authentication_stopped')
            self.assertEqual(session.post.call_count,1)
            self.assertEqual(summary['ai_status_counts'],
                {'not_attempted_authentication_failure':5,'http_401':1})
            self.assertEqual(summary['comparison']['model_classified'],0)
            self.assertEqual(summary['provider_reported_usage_estimated_cost_usd'],0)
            self.assertGreater(summary['unreported_usage_reserved_usd'],0)
            self.assertEqual(len(pd.read_csv(out/'pilot_results.csv')),6)
            self.assertEqual(self.sample_path.read_bytes(),before)
            self.run_pilot(output=None,resume=out,execute=True,session=session)
            self.assertEqual(session.post.call_count,1)

    def test_dry_run_pilot_preserves_inputs_no_predictions_or_cost(self):
        before=self.sample_path.read_bytes()
        session=Mock()
        out,summary=self.run_pilot(session=session)
        self.assertEqual(summary['status'],'dry_run_only')
        self.assertEqual(summary['accounted_cost_usd'],0)
        self.assertEqual(summary['attempts_reserved_total'],0)
        self.assertEqual(self.sample_path.read_bytes(),before)
        records=[json.loads(line) for line in (out/'pilot_results.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(all(row['ai_relevance'] is None and row['human_verified']=='' for row in records))
        self.assertTrue(all(row['description_text']==BODY for row in records))
        session.post.assert_not_called()

    def test_budget_stop_then_resume_only_unfinished_records(self):
        session=Mock();session.post.return_value=response()
        pilot=select_pilot(self.sample,6)
        first=pilot.iloc[0]
        probe=LLMClassifier(self.config,POLICY,self.temp/'cache',self.temp/'probe.json',1)
        cap=probe.estimate(first.job_title,BODY)['reserved_cost_per_attempt_usd']
        with patch.dict(os.environ,{'STEP3D_TEST_KEY':'not-a-real-key'}):
            out,summary=self.run_pilot(session=session,execute=True,budget_usd=cap)
            self.assertEqual(summary['status'],'budget_stopped')
            self.assertEqual(session.post.call_count,1)
            second=Mock();second.post.return_value=response()
            _,summary=self.run_pilot(output=None,resume=out,execute=True,session=second,budget_usd=1)
        self.assertEqual(summary['status'],'completed')
        self.assertEqual(second.post.call_count,5)
        self.assertEqual(summary['comparison']['model_classified'],6)

    def test_resume_rejects_changed_model_or_source(self):
        out,_=self.run_pilot()
        with self.assertRaises(ValueError):self.run_pilot(output=None,resume=out,config=replace(self.config,model='other'))
        self.sample.loc[0,'description_text']='changed';self.sample.to_csv(self.sample_path,index=False)
        with self.assertRaises(ValueError):self.run_pilot(output=None,resume=out)

    def test_resume_preserves_independent_human_review_file(self):
        out,_=self.run_pilot()
        review_path=out/'human_review_pilot.csv'
        review=pd.read_csv(review_path,dtype=str,keep_default_na=False)
        review.loc[0,'review_notes']='Human work in progress'
        review.to_csv(review_path,index=False,encoding='utf-8-sig')
        before=review_path.read_bytes()
        self.run_pilot(output=None,resume=out)
        self.assertEqual(review_path.read_bytes(),before)

    def test_suggestions_never_import_human_verification(self):
        path=self.temp/'suggestions.csv'
        pd.DataFrame([dict(advertisement_id='1',ai_relevance='relevant',ai_primary_category='AI Engineer',
                           human_verified='true',human_relevance='relevant',reviewer='AI')]).to_csv(path,index=False)
        reference=load_suggestions(path)
        self.assertNotIn('human_verified',reference)
        self.assertNotIn('human_relevance',reference)
        out,summary=self.run_pilot(suggestions_path=path)
        self.assertTrue(summary['suggestions_available'])
        rows=pd.read_csv(out/'pilot_results.csv',dtype=str,keep_default_na=False)
        self.assertTrue(rows.human_verified.eq('').all())
        self.assertNotIn('accuracy',summary['comparison'])

    def test_reference_unknown_schema_requires_explicit_columns(self):
        path=self.temp/'suggestions.csv'
        pd.DataFrame([dict(ad='1',decision='relevant',category='Data Engineer')]).to_csv(path,index=False)
        with self.assertRaises(ValueError):load_suggestions(path)
        ref=load_suggestions(path,id_column='ad',relevance_column='decision',category_column='category')
        self.assertEqual(ref.provisional_ai_reference_relevance.iloc[0],'relevant')

    def test_execute_without_credentials_never_sends(self):
        session=Mock()
        with patch.dict(os.environ,{},clear=True):out,summary=self.run_pilot(execute=True,session=session)
        self.assertEqual(summary['status'],'credential_required')
        self.assertEqual(summary['attempts_reserved_total'],0)
        session.post.assert_not_called()

    def test_new_run_refuses_overwriting_existing_output(self):
        self.run_pilot()
        with self.assertRaises(FileExistsError):self.run_pilot()


if __name__=='__main__':unittest.main()
