"""Regression tests reproduce the seven-attempt stop without any real API calls."""
import copy
import json
import unittest
from unittest.mock import Mock

import requests

from src.resume_step3f_pilot import audit,recover,resume
from src.run_step3f import run
from src.step3f_classifier import validate_span_prediction
from src.llm_classifier import ClassificationError
from src.recover_large_pilot import resilient_atomic_json as atomic_json
from src.validate_classification import sha256
from tests.test_step3d import response
from tests.test_step3f import prediction
from tests import test_step3f_span_pilot as fixture_module


class Step3FResumeTests(unittest.TestCase):
    def setUp(self):
        fixture_module.SpanPilotTests.setUp(self)
        invalid=prediction(ai_evidence_span_ids=['D00000',"D00001','D00002"])
        self.session.post.side_effect=[response(prediction()),response(prediction()),response(prediction()),
            response(invalid),response(prediction()),response(prediction()),requests.exceptions.SSLError('historical SSL detail not saved')]
        run(self.prepared,self.temp/'run',self.config,1,cache_dir=self.temp/'cache',session=self.session,execute=True)
        self.source=self.temp/'run';self.recovered=self.temp/'recovered'
        self.original=json.loads((self.source/'checkpoint.json').read_text(encoding='utf-8'))
        self.source_hash=sha256(self.source/'checkpoint.json')
        self.fresh=Mock();self.fresh.post.return_value=response(prediction())

    def recover(self):return recover(self.prepared,self.source,self.recovered,1)

    def test_exact_observed_combined_span_id_still_rejected_without_repair(self):
        value=prediction(ai_evidence_span_ids=['D00003','D00004',"D00005','D00006','D00008"])
        with self.assertRaisesRegex(ClassificationError,'Unknown evidence span ID'):
            validate_span_prediction(value,'Data Engineer','Source '*1000)

    def test_audit_confirms_all_five_receipts_cost_holds_and_no_hidden_results(self):
        report,_,_,_=audit(self.prepared,self.source,1)
        self.assertEqual((report['accepted'],len(report['failures']),report['safe_remaining']),(5,2,13))
        self.assertEqual(report['accepted_cache_receipts'],5)
        self.assertEqual(report['additional_processed_receipts_without_saved_result'],[])
        self.assertEqual(report['ledger_attempts_without_saved_result'],[])
        self.assertEqual(len(report['pending_markers']),1)
        self.assertTrue(report['unknown_billing_cannot_be_ruled_out'])
        self.assertGreater(report['unreported_usage_reserved_usd'],0)
        self.assertEqual(sha256(self.source/'checkpoint.json'),self.source_hash)

    def test_recovery_is_byte_identical_and_preserves_original_artifacts(self):
        self.recover()
        self.assertEqual(sha256(self.recovered/'checkpoint.json'),self.source_hash)
        self.assertEqual(sha256(self.recovered/'source_artifacts/checkpoint.json'),self.source_hash)
        self.assertEqual(sha256(self.source/'checkpoint.json'),self.source_hash)
        with self.assertRaises(FileExistsError):self.recover()

    def test_original_runner_reproduces_resume_stop_without_new_requests(self):
        run(self.prepared,self.source,self.config,1,cache_dir=self.temp/'cache',session=self.fresh,execute=True,resume=True)
        self.fresh.post.assert_not_called()
        state=json.loads((self.source/'checkpoint.json').read_text(encoding='utf-8'))
        self.assertEqual(len(state['results']),7)

    def test_fixed_resume_calls_only_thirteen_new_jobs_and_preserves_all_seven(self):
        self.recover()
        pending=next((self.temp/'cache').glob('*.pending.json'));pending_hash=sha256(pending)
        report=resume(self.recovered,1,execute=True,session=self.fresh)
        self.assertEqual(self.fresh.post.call_count,13)
        state=json.loads((self.recovered/'checkpoint.json').read_text(encoding='utf-8'))
        self.assertEqual(len(state['results']),20)
        self.assertEqual(len(state['ledger']),20)
        self.assertEqual(state['ledger'][:7],self.original['ledger'])
        self.assertEqual({k:state['results'][k] for k in self.original['results']},self.original['results'])
        self.assertEqual(report['accepted'],18)
        self.assertEqual(report['safe_remaining'],0)
        self.assertEqual(sha256(pending),pending_hash)
        self.assertEqual(sha256(self.source/'checkpoint.json'),self.source_hash)
        # Repeating resume must not retry old failed attempts or newly accepted jobs.
        resume(self.recovered,1,execute=True,session=self.fresh)
        self.assertEqual(self.fresh.post.call_count,13)

    def test_dry_resume_is_unpaid_and_preserves_checkpoint_bytes(self):
        self.recover()
        report=resume(self.recovered,1,execute=False,session=self.fresh)
        self.assertEqual(report['safe_remaining'],13)
        self.fresh.post.assert_not_called()
        self.assertEqual(sha256(self.recovered/'checkpoint.json'),self.source_hash)

    def test_remaining_budget_includes_prior_usage_and_unknown_hold(self):
        self.recover()
        accounted=sum(x['accounted_cost_usd'] for x in self.original['ledger'])
        report=resume(self.recovered,accounted+.0000001,execute=True,session=self.fresh)
        self.fresh.post.assert_not_called()
        self.assertEqual(report['saved_results'],7)
        self.assertAlmostEqual(report['accounted_cost_usd'],accounted)

    def test_future_ssl_stops_once_and_saves_redacted_details_without_retry(self):
        self.recover()
        self.fresh.post.side_effect=requests.exceptions.SSLError(
            'TLS failure https://proxy-user:proxy-pass@example.invalid/path password=hidden fake-never-sent')
        report=resume(self.recovered,1,execute=True,session=self.fresh)
        self.assertEqual(self.fresh.post.call_count,1)
        self.assertEqual(report['saved_results'],8)
        state=json.loads((self.recovered/'checkpoint.json').read_text(encoding='utf-8'))
        diagnostic=state['ledger'][-1]['network_error_diagnostic']
        self.assertEqual(diagnostic['type'],'SSLError')
        self.assertFalse(diagnostic['http_response_received'])
        for secret in ['proxy-user','proxy-pass','hidden','fake-never-sent']:
            self.assertNotIn(secret,json.dumps(diagnostic))

    def test_corrupt_saved_source_span_blocks_resume_before_api(self):
        self.recover()
        path=self.recovered/'checkpoint.json';state=copy.deepcopy(self.original)
        first=next(r for r in state['results'].values() if r['ai_status']=='completed')
        first['ai_evidence_spans'][0]['text']='Invented source'
        atomic_json(path,state)
        with self.assertRaisesRegex(ValueError,'Saved excerpts/offsets'):
            resume(self.recovered,1,execute=True,session=self.fresh)
        self.fresh.post.assert_not_called()

    def test_unrecorded_attempt_is_not_treated_as_a_safe_new_job(self):
        path=self.source/'checkpoint.json';state=copy.deepcopy(self.original)
        unknown=next(k for k,v in state['results'].items() if v['ai_status']=='request_outcome_unknown')
        del state['results'][unknown];atomic_json(path,state)
        report,_,_,_=audit(self.prepared,self.source,1)
        self.assertEqual(report['blocked_remaining'],1)
        self.assertEqual(report['safe_remaining'],13)
        with self.assertRaisesRegex(ValueError,'Untouched jobs have unknown'):
            self.recover()

    def test_external_receipt_attempt_is_not_silently_lost_from_accounting(self):
        receipt_path=next(p for p in (self.temp/'cache').glob('*.json') if not p.name.endswith('.pending.json'))
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        receipt['attempt_id']='outside-this-ledger';atomic_json(receipt_path,receipt)
        with self.assertRaisesRegex(ValueError,'Unaccounted external receipt'):
            audit(self.prepared,self.source,1)


if __name__=='__main__':unittest.main()
