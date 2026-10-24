"""Recovery/resume tests use mock sessions and temporary local data only."""
import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock,patch

import tests.test_step3e as fixtures
from tests.test_step3d import response,prediction
from src.recover_large_pilot import recover,inspect,resilient_atomic_json,writable_probe
from src.resume_large_pilot import resume
from src.llm_classifier import atomic_json


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.Step3ETests('test_new_prompt_identity_dry_run_has_zero_api_calls')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.temp=self.fixture.temp

    def parent(self):
        session=Mock();session.post.side_effect=[response(),response(prediction(ai_evidence='invented')),
            response(),response(),response(),response()]
        source,_=self.fixture.runner(execute=True,session=session)
        # Emulate an interruption after two results; retain only matching receipts.
        cp=json.loads((source/'checkpoint.json').read_text(encoding='utf-8'))
        cp['ledger']=cp['ledger'][:2]
        ids=[e['advertisement_id'] for e in cp['ledger']]
        cp['results']={key:cp['results'][key] for key in ids}
        atomic_json(source/'checkpoint.json',cp)
        keys={e['cache_key'] for e in cp['ledger']}
        for path in (self.temp/'cache').glob('*.json'):
            if path.stem not in keys:path.unlink()
        return source,cp

    def test_recovery_preserves_checkpoint_failed_diagnostics_and_costs(self):
        source,cp=self.parent();before={p:p.read_bytes() for p in source.iterdir() if p.is_file()}
        with patch('requests.sessions.Session.request',side_effect=AssertionError('No network')):
            report=recover(source,self.temp/'recovered',1,self.temp/'cache')
        self.assertEqual((report['accepted'],report['rejected'],report['remaining_unrecorded']),(1,1,4))
        self.assertEqual((self.temp/'recovered'/'checkpoint.json').read_bytes(),before[source/'checkpoint.json'])
        self.assertTrue(report['writable_destination'])
        self.assertEqual(report['provider_reported_usage_estimated_cost_usd'],sum(e['accounted_cost_usd'] for e in cp['ledger']))
        for path,data in before.items():self.assertEqual(path.read_bytes(),data)

    def test_resume_skips_success_and_terminal_failure_pays_only_unfinished(self):
        source,cp=self.parent();recover(source,self.temp/'recovered',1,self.temp/'cache')
        session=Mock();session.post.return_value=response()
        with redirect_stdout(io.StringIO()):
            _,summary=resume(self.temp/'recovered',1,True,self.temp/'cache',session=session,sleep=lambda seconds:None)
        self.assertEqual(session.post.call_count,4)
        recovered=json.loads((self.temp/'recovered'/'checkpoint.json').read_text(encoding='utf-8'))
        self.assertEqual(recovered['ledger'][:2],cp['ledger'])
        rejected=next(key for key,value in cp['results'].items() if value['ai_status']=='invalid_model_response')
        self.assertEqual(recovered['results'][rejected],cp['results'][rejected])
        self.assertEqual(summary['accepted'],5)
        self.assertEqual(summary['new_attempts_reserved'],4)

    def test_dry_resume_never_sends_requests(self):
        source,_=self.parent();recover(source,self.temp/'recovered',1,self.temp/'cache')
        session=Mock()
        with redirect_stdout(io.StringIO()):
            _,summary=resume(self.temp/'recovered',1,False,self.temp/'cache',session=session)
        session.post.assert_not_called()
        self.assertEqual(summary['new_attempts_reserved'],0)

    def test_incompatible_prompt_and_unknown_pending_block_recovery(self):
        source,cp=self.parent()
        with patch('src.recover_large_pilot.PROMPT_VERSION','changed'):
            with self.assertRaisesRegex(ValueError,'incompatible'):
                recover(source,self.temp/'recovered',1,self.temp/'cache')
        marker=self.temp/'cache'/(cp['ledger'][0]['cache_key']+'.pending.json')
        atomic_json(marker,{'attempt_id':'unknown'})
        with self.assertRaisesRegex(ValueError,'Unknown request outcomes'):
            recover(source,self.temp/'recovered',1,self.temp/'cache')
        self.assertFalse((self.temp/'recovered').exists())

    def test_file_replace_transient_permissions_retry_only_filesystem(self):
        path=self.temp/'state.json';atomic_json(path,{'old':True})
        replace=os.replace;sleep=Mock()
        with patch('src.recover_large_pilot.os.replace',side_effect=[PermissionError('sharing violation'),None]) as mock_replace:
            # Use real replacement after the simulated first failure.
            def operation(*args):
                if operation.first:operation.first=False;raise PermissionError('sharing violation')
                return replace(*args)
            operation.first=True;mock_replace.side_effect=operation
            resilient_atomic_json(path,{'new':True},sleep=sleep)
        self.assertEqual(json.loads(path.read_text()),{'new':True})
        self.assertEqual(sleep.call_count,1)
        self.assertTrue(writable_probe(self.temp))

    def test_persistent_permissions_leave_previous_checkpoint_intact(self):
        path=self.temp/'state.json';atomic_json(path,{'old':True});before=path.read_bytes()
        with patch('src.recover_large_pilot.os.replace',side_effect=PermissionError('denied')):
            with self.assertRaises(PermissionError):
                resilient_atomic_json(path,{'new':True},attempts=2,sleep=lambda seconds:None)
        self.assertEqual(path.read_bytes(),before)

    def test_no_overwrite_or_budget_reset(self):
        source,_=self.parent()
        with self.assertRaises(ValueError):recover(source,source,1,self.temp/'cache')
        with self.assertRaisesRegex(ValueError,'Budget below'):
            inspect(source,.000001,self.temp/'cache')


if __name__=='__main__':unittest.main()
