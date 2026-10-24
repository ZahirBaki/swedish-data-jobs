"""Mock-only audit of the $3.50 boundary; production prepared identities are unchanged."""
import inspect
import json
import shutil
import unittest
import uuid
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock,patch

import pandas as pd
import requests

from src import corpus_workflow as corpus
from src.llm_classifier import BudgetExceeded,Config
from src.step3f_reliable_classifier import Step3FReliableClassifier,POLICY_PATH
from src.validate_classification import sha256
from tests.test_step3d import response
from tests.test_step3f import TITLE,BODY,prediction


class Budget350Tests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).parent/('.test-budget350-'+uuid.uuid4().hex);self.root.mkdir()
        self.addCleanup(lambda:shutil.rmtree(self.root))
        self.cache=self.root/'cache';self.cache.mkdir()
        self.session=Mock();self.session.post.return_value=response(prediction(),usage=dict(prompt_tokens=1000000,completion_tokens=0))
        self.conf=Config('mock',.4,1.6,max_retries=0,min_interval=0)
        env=patch.dict('os.environ',{'OPENAI_API_KEY':'fake-never-sent'});env.start();self.addCleanup(env.stop)

    def engine(self,output=None,budget=3.5):
        engine=Step3FReliableClassifier(self.conf,POLICY_PATH.read_text(encoding='utf-8'),self.cache,
            (output or self.root/'run')/'checkpoint.json',budget,execute=True,
            blocked_source_hashes=set(),session=self.session)
        # Deliberately scaled mock costs exercise the dollar boundary in a few calls.
        engine.estimate=Mock(return_value=dict(reserved_cost_per_attempt_usd=.4,expected_cost_usd=.4,
            expected_input_tokens=100,expected_output_tokens=50,reserved_input_tokens=100,
            reserved_output_tokens=50,input_limit_exceeded=False))
        return engine

    def seed(self,engine,cost):
        engine.state['ledger'].append(dict(attempt_id='historical',advertisement_id='old',cache_key='old',
            status='completed',accounted_cost_usd=cost,reserved_cost_usd=cost,usage_basis='provider_reported_tokens'))
        engine.save()

    def test_reserve_is_durable_and_checked_before_every_post(self):
        engine=self.engine();self.seed(engine,2.9)
        def post(*args,**kwargs):
            saved=json.loads(engine.checkpoint_path.read_text(encoding='utf-8'))
            self.assertEqual(saved['ledger'][-1]['status'],'reserved_before_request')
            self.assertLessEqual(sum(Decimal(str(x['accounted_cost_usd'])) for x in saved['ledger']),Decimal('3.50'))
            return response(prediction(),usage=dict(prompt_tokens=1000000,completion_tokens=0))
        self.session.post.side_effect=post
        engine.classify('first',TITLE,BODY)
        with self.assertRaises(BudgetExceeded):engine.classify('second',TITLE,BODY+' second')
        self.assertEqual(self.session.post.call_count,1)
        self.assertEqual(engine.spent,3.3)

    def test_exact_boundary_allowed_then_next_request_blocked(self):
        engine=self.engine();self.seed(engine,3.1)
        engine.classify('first',TITLE,BODY)
        self.assertEqual(engine.spent,3.5)
        with self.assertRaises(BudgetExceeded):engine.classify('second',TITLE,BODY+' second')
        self.assertEqual(self.session.post.call_count,1)

    def test_resume_keeps_previous_spend(self):
        engine=self.engine();self.seed(engine,2.9)
        engine.classify('first',TITLE,BODY)
        resumed=self.engine()
        self.assertEqual(resumed.spent,3.3)
        with self.assertRaises(BudgetExceeded):resumed.classify('second',TITLE,BODY+' second')
        self.assertEqual(self.session.post.call_count,1)

    def test_unknown_hold_survives_resume_and_blocks_next_request(self):
        engine=self.engine();self.seed(engine,3.0)
        self.session.post.side_effect=requests.exceptions.Timeout('mock-only')
        result=engine.classify('unknown',TITLE,BODY)
        self.assertEqual(result['ai_status'],'request_outcome_unknown')
        self.assertEqual(engine.spent,3.4)
        resumed=self.engine()
        self.assertEqual(resumed.cost_breakdown()['unreported_usage_reserved_usd'],.4)
        resumed.classify('unknown',TITLE,BODY)
        resumed.classify('alias',TITLE,BODY)
        with self.assertRaises(BudgetExceeded):resumed.classify('next',TITLE,BODY+' second')
        self.assertEqual(self.session.post.call_count,1)

    def test_invalid_and_http_failure_are_not_retried(self):
        for name,reply in [('invalid',response(prediction(),finish='length')),
                           ('http',response(prediction(),status=503))]:
            with self.subTest(name=name):
                self.session.reset_mock();self.session.post.return_value=reply
                engine=self.engine(self.root/name)
                engine.classify(name,TITLE,BODY+name)
                engine.classify(name,TITLE,BODY+name)
                self.assertEqual(self.session.post.call_count,1)
                self.assertEqual(engine.config.max_retries,0)
                self.assertFalse(engine.retry_unknown)

    def test_unexpected_provider_usage_is_detected_after_response(self):
        engine=self.engine()
        self.session.post.return_value=response(prediction(),usage=dict(prompt_tokens=9000000,completion_tokens=0))
        with self.assertRaises(BudgetExceeded):engine.classify('overrun',TITLE,BODY)
        self.assertEqual(engine.spent,3.6)
        with self.assertRaises(BudgetExceeded):engine.classify('next',TITLE,BODY+' second')
        self.assertEqual(self.session.post.call_count,1)

    def runner(self,**kwargs):
        prepared=self.root/'prepared';prepared.mkdir(exist_ok=True)
        (prepared/'manifest.json').write_text('{}')
        output=self.root/'run'
        identity=dict(prepared_sha256=sha256(prepared/'manifest.json'),cache=str(self.cache.resolve()),config=asdict(self.conf))
        (output/'identity.json').write_text(json.dumps(identity))
        source=pd.DataFrame([dict(advertisement_id='a',job_title=TITLE,description_text=BODY),
            dict(advertisement_id='b',job_title=TITLE,description_text=BODY+' second')])
        base=pd.DataFrame(columns=['advertisement_id'])
        with patch.object(corpus,'prepared_inputs',return_value=({'config':asdict(self.conf)},source,(base,pd.DataFrame()))),\
             patch.object(corpus,'historical',return_value=({},set(),set(),set(),{})),\
             patch.object(corpus,'CACHE',self.cache),\
             patch.object(corpus,'client',side_effect=lambda out,budget=60,session=None:self.engine(Path(out),budget)):
            return corpus.run(prepared,output,resume=True,**kwargs)

    def test_corpus_runner_stops_and_resumes_with_cumulative_350_limit(self):
        engine=self.engine();self.seed(engine,3.0)
        report=self.runner(budget=3.5,execute=True,batch_size=1,max_batches=2)
        self.assertEqual(report['stop_reason'],'application_budget_exhausted')
        self.assertEqual(report['accounted_cost_usd'],3.4)
        resumed=self.runner(budget=3.5,execute=True,batch_size=100,max_batches=50)
        self.assertEqual(resumed['stop_reason'],'application_budget_exhausted')
        self.assertEqual(resumed['accounted_cost_usd'],3.4)
        self.assertEqual(self.session.post.call_count,1)

    def test_dry_run_does_not_persist_a_budget_ceiling(self):
        engine=self.engine();self.seed(engine,0)
        self.runner(budget=3.5,execute=False)
        report=self.runner(execute=False)
        self.assertEqual(report['application_budget_usd'],60)
        self.assertEqual(inspect.signature(corpus.run).parameters['budget'].default,60)
        self.session.post.assert_not_called()

if __name__=='__main__':unittest.main()
