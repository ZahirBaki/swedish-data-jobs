"""Audit/recover/resume only untouched Step 3F jobs, preserving terminal failures.

Frozen model/prompt/schema/transport files are deliberately unchanged. This fixes
the old runner stopping again on a historical unknown outcome during resume.
Unknown outcomes keep their holds and pending markers; they are never retried.
"""
import argparse
import json
import re
import shutil
from collections import Counter
from decimal import Decimal
from pathlib import Path

import requests

from src.llm_classifier import Config, BudgetExceeded, cost_usd, file_lock
from src.recover_large_pilot import resilient_atomic_json as atomic_json, writable_probe
from src.run_step3f import validate_prepared, write_dataset
from src.step3f_classifier import (Step3FClassifier, AI_FIELDS, POLICY_PATH, PROMPT_VERSION,
    POLICY_VERSION, VALIDATION_VERSION, validate_span_prediction, span_match)
from src.validate_classification import sha256


class DiagnosticSession:
    """Do not retry requests; retain a redacted error for future network failures."""
    def __init__(self,session=None):
        self.session=session or requests.Session()
        self.last_error=None

    def post(self,*args,**kwargs):
        self.last_error=None
        try:return self.session.post(*args,**kwargs)
        except requests.RequestException as exc:
            auth=kwargs.get('headers',{}).get('Authorization','')
            secret=auth.removeprefix('Bearer ')
            text=re.sub(r'https?://[^\s\'\"<>]+','[REDACTED_URL]',str(exc))
            text=re.sub(r'(?i)(?:authorization|api[_-]?key|password|token)\s*[:=]\s*[^\s,;]+',
                        '[REDACTED_CREDENTIAL]',text)
            self.last_error=dict(type=type(exc).__name__,
                message=Step3FClassifier.redact_diagnostic(text,secret),
                diagnostic_only=True,http_response_received=False)
            raise


def audit(prepared,run_dir,budget_usd=1):
    prepared,run_dir=Path(prepared),Path(run_dir)
    manifest,frame=validate_prepared(prepared)
    identity=json.loads((run_dir/'run_identity.json').read_text(encoding='utf-8'))
    config=Config(**identity['config'])
    if identity.get('stage')!='step3f' or identity.get('prepared_manifest_sha256')!=sha256(prepared/'manifest.json'):
        raise ValueError('Prepared/run identity mismatch')
    if identity.get('max_ads')!=len(frame):raise ValueError('Resume requires the exact frozen sample')
    client=Step3FClassifier(config,POLICY_PATH.read_text(encoding='utf-8'),Path(identity['cache_dir']),
        run_dir/'checkpoint.json',budget_usd,execute=False)
    state=client.state
    if state.get('identity')!=identity:raise ValueError('Checkpoint identity mismatch')
    sources={r['advertisement_id']:r for r in frame.to_dict('records')}
    if set(state['results'])-set(sources):raise ValueError('Unexpected checkpoint result ID')
    attempts={}; accepted=[];failures=[];receipts=[];markers=[];remaining=[]
    for item in state['ledger']:
        identifier=item['advertisement_id']; source=sources.get(identifier)
        if source is None or item['attempt_id'] in attempts:raise ValueError('Unexpected or duplicate ledger attempt')
        if item['cache_key']!=client.cache_key(source['job_title'],source['description_text']):
            raise ValueError('Ledger request identity mismatch')
        if item['usage_basis']=='provider_reported_tokens':
            tokens=[item.get('prompt_tokens'),item.get('completion_tokens')]
            if any(type(t) is not int or t<0 for t in tokens) or abs(cost_usd(*tokens,config)-item['accounted_cost_usd'])>1e-12:
                raise ValueError('Ledger usage/cost mismatch')
        elif item['accounted_cost_usd']!=item['reserved_cost_usd']:
            raise ValueError('Unreported hold must remain intact')
        attempts[item['attempt_id']]=item
    for identifier,source in sources.items():
        title,description=source['job_title'],source['description_text']
        key=client.cache_key(title,description);result=state['results'].get(identifier)
        receipt=client.cached(title,description)
        pending=client.cache_dir/(key+'.pending.json')
        if pending.exists():
            marker=json.loads(pending.read_text(encoding='utf-8'))
            if marker.get('cache_key')!=key:raise ValueError('Pending marker identity mismatch')
            markers.append(dict(advertisement_id=identifier,attempt_id=marker.get('attempt_id'),
                                known_ledger_attempt=marker.get('attempt_id') in attempts,path=str(pending)))
        if receipt:
            receipts.append(dict(advertisement_id=identifier,attempt_id=receipt.get('attempt_id'),
                                 known_ledger_attempt=receipt.get('attempt_id') in attempts))
        if result:
            if result.get('ai_cache_key')!=key:raise ValueError('Saved result content mismatch')
            if result.get('ai_human_verified') is not False:raise ValueError('AI must not claim human verification')
            if (result.get('ai_prompt_version')!=PROMPT_VERSION or result.get('ai_policy_version')!=POLICY_VERSION
                    or result.get('ai_validation_version')!=VALIDATION_VERSION):raise ValueError('Result version mismatch')
            if result['ai_status'] in {'completed','cache_hit','missing_description'}:
                checked=validate_span_prediction({k:result[k] for k in AI_FIELDS},title,description)
                if result['ai_status']!='missing_description':
                    match=span_match(result['ai_evidence_span_ids'],title,description)
                    if checked['ai_evidence']!=result.get('ai_evidence') or match['spans']!=result.get('ai_evidence_spans',[]):
                        raise ValueError('Saved excerpts/offsets do not match original source')
                    if not receipt:raise ValueError('Accepted result is missing its durable cache receipt')
                    if any(result[k]!=receipt['prediction'][k] for k in AI_FIELDS):
                        raise ValueError('Accepted checkpoint and cache predictions differ')
                accepted.append(identifier)
            else:
                related=[x for x in state['ledger'] if x['advertisement_id']==identifier]
                failures.append(dict(advertisement_id=identifier,job_title=source['job_title'],status=result['ai_status'],
                    diagnostics=[dict(http_status=x.get('http_status'),network_error_type=x.get('network_error_type'),
                        validation_error=x.get('model_response_diagnostic',{}).get('validation_error'),
                        rejected_response=x.get('model_response_diagnostic',{}).get('content'),
                        accounted_cost_usd=x['accounted_cost_usd'],usage_basis=x['usage_basis']) for x in related]))
        else:
            # Never replay an attempt merely because the result export/checkpoint lagged.
            related=[x for x in state['ledger'] if x['cache_key']==key]
            blocked=bool(pending.exists() or related)
            remaining.append(dict(advertisement_id=identifier,cache_key=key,blocked=blocked,
                cached=receipt is not None,**client.estimate(title,description)))
    unknown_extra=[x for x in receipts+markers if not x['known_ledger_attempt']]
    # An external receipt can have external costs: audit, never silently discard accounting.
    if unknown_extra:raise ValueError('Unaccounted external receipt/marker requires separate inspection')
    reserved_without_result=[x['advertisement_id'] for x in state['ledger'] if x['advertisement_id'] not in state['results']]
    remaining_budget=Decimal(str(budget_usd))-Decimal(str(client.spent))
    if remaining_budget<0:raise BudgetExceeded('Budget is below existing usage/holds')
    safe=[x for x in remaining if not x['blocked']]
    report=dict(total=len(frame),saved_results=len(state['results']),ledger_entries=len(state['ledger']),
        accepted_ids=accepted,accepted=len(accepted),failures=failures,
        not_executed=len(remaining),safe_remaining=len(safe),blocked_remaining=len(remaining)-len(safe),
        accepted_cache_receipts=len(receipts),cache_receipts=receipts,pending_markers=markers,
        ledger_attempts_without_saved_result=reserved_without_result,
        additional_processed_receipts_without_saved_result=[x['advertisement_id'] for x in receipts
            if x['advertisement_id'] not in state['results']],
        accounted_cost_usd=client.spent,**client.cost_breakdown(),
        total_budget_usd=budget_usd,remaining_budget_usd=float(remaining_budget),
        remaining_expected_cost_usd=sum(x['expected_cost_usd'] for x in safe if not x['cached']),
        remaining_conservative_cost_usd=sum(x['reserved_cost_per_attempt_usd'] for x in safe if not x['cached']),
        checkpoint_sha256=sha256(run_dir/'checkpoint.json'),compatibility_verified=True,
        unknown_billing_cannot_be_ruled_out=any(x['usage_basis']!='provider_reported_tokens' for x in state['ledger']),
        paid_requests_made_by_audit=0)
    return report,frame,client,remaining


def recover(prepared,source,output,budget_usd=1):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Recovery requires a new output directory')
    report,frame,client,remaining=audit(prepared,source,budget_usd)
    if report['blocked_remaining']:raise ValueError('Untouched jobs have unknown/unrecorded attempts; inspect first')
    output.mkdir(parents=True)
    archive=output/'source_artifacts';archive.mkdir()
    for path in source.iterdir():
        if path.is_file() and path.name!='run.lock':shutil.copyfile(path,archive/path.name)
    shutil.copyfile(source/'checkpoint.json',output/'checkpoint.json')
    shutil.copyfile(source/'run_identity.json',output/'run_identity.json')
    if sha256(output/'checkpoint.json')!=report['checkpoint_sha256']:raise RuntimeError('Checkpoint copy changed')
    writable_probe(output);writable_probe(client.cache_dir)
    report.update(source_run=str(source),recovered_run=str(output),prepared=str(Path(prepared).resolve()),
        runner_sha256=sha256(Path(__file__)),recovery_status='recovered_not_executed')
    atomic_json(output/'recovery_manifest.json',report)
    export(output,frame,client,'recovered_not_executed')
    return report


def export(output,frame,client,stop_reason):
    rows=[dict(row,**client.state['results'].get(row['advertisement_id'],dict(ai_status='not_executed',
        classification_status=None,ai_relevance=None,ai_primary_category=None,ai_human_verified=False)))
        for row in frame.to_dict('records')]
    write_dataset(output,rows)
    atomic_json(output/'resume_summary.json',dict(total=len(rows),
        ai_statuses=dict(Counter(r['ai_status'] for r in rows)),stop_reason=stop_reason,
        accounted_cost_usd=client.spent,**client.cost_breakdown(),
        human_verified=False,quality_metrics_calculated=False))


def resume(run_dir,budget_usd=1,execute=False,session=None):
    run_dir=Path(run_dir)
    recovery=json.loads((run_dir/'recovery_manifest.json').read_text(encoding='utf-8'))
    if recovery.get('runner_sha256')!=sha256(Path(__file__)):raise ValueError('Recovery runner changed; audit a new recovery')
    with file_lock(run_dir/'run.lock'):
        report,frame,client,remaining=audit(recovery['prepared'],run_dir,budget_usd)
        writable_probe(run_dir);writable_probe(client.cache_dir)
        if report['blocked_remaining']:raise ValueError('Unsaved attempts block safe continuation')
        if not execute:return report
        # All old results, including terminal unknowns, are deliberately skipped.
        diagnostics=DiagnosticSession(session)
        client.execute=True;client.session=diagnostics
        client.save()  # Verify actual checkpoint replacement before external requests.
        stop_reason=None
        try:
            untouched={x['advertisement_id'] for x in remaining}
            for source in frame.to_dict('records'):
                if source['advertisement_id'] not in untouched:continue
                result=client.classify(source['advertisement_id'],source['job_title'],source['description_text'])
                if diagnostics.last_error:
                    client.state['ledger'][-1]['network_error_diagnostic']=diagnostics.last_error
                    client.save()
                if result['ai_status'] in {'http_401','http_402','http_403','http_429','request_outcome_unknown'}:
                    stop_reason=result['ai_status'];break
        except BudgetExceeded as exc:stop_reason=str(exc)
        finally:export(run_dir,frame,client,stop_reason)
        return audit(recovery['prepared'],run_dir,budget_usd)[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--recover-to',type=Path)
    parser.add_argument('--budget-usd',type=float,default=1)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.recover_to:
        if args.execute or not args.prepared:parser.error('Recovery is offline and requires --prepared')
        result=recover(args.prepared,args.run,args.recover_to,args.budget_usd)
    elif args.prepared:
        if args.execute:parser.error('Live continuation requires a recovered directory')
        result=audit(args.prepared,args.run,args.budget_usd)[0]
    else:result=resume(args.run,args.budget_usd,args.execute)
    print(json.dumps(result,indent=2,ensure_ascii=True))


if __name__=='__main__':main()
