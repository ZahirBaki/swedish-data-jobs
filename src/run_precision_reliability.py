"""Explicitly budgeted Step 3F 1.1 test, hard-limited to three known-outcome cases."""
import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.llm_classifier import Config,BudgetExceeded,file_lock
from src.recover_large_pilot import resilient_atomic_json as atomic_json,writable_probe
from src.resume_step3f_pilot import DiagnosticSession
from src.run_step3f import write_dataset
from src.step3f_reliable_classifier import Step3FReliableClassifier,source_hash,POLICY_PATH,PROMPT_PATH
from src.validate_classification import ROOT,read_csv,sha256

ALLOWED_IDS={'31276346','31557998','31395973'}
CODE_NAMES=['run_precision_reliability.py','precision_inclusion.py','step3f_reliable_classifier.py',
    'step3f_classifier.py','llm_classifier.py','recover_large_pilot.py','resume_step3f_pilot.py']


def code_hashes():return {name:sha256(ROOT/'src'/name) for name in CODE_NAMES}


def validate_prepared(prepared):
    prepared=Path(prepared);m=json.loads((prepared/'pilot_manifest.json').read_text(encoding='utf-8'))
    if (m.get('stage')!='precision-reliability-three' or m['input_sha256']!=sha256(prepared/'reliability_cases.csv')
        or m['guard_sha256']!=sha256(prepared/'unknown_outcome_guard.json') or m['code_hashes']!=code_hashes()
        or m['policy_sha256']!=sha256(POLICY_PATH) or m['prompt_sha256']!=sha256(PROMPT_PATH)
        or m['history_checkpoint_sha256']!=sha256(m['history_checkpoint'])):
        raise ValueError('Pilot source/history/code/prompt/policy compatibility mismatch')
    if (m['precision_policy_sha256']!=sha256(ROOT/'docs/precision_first_inclusion_policy_v1.md')
        or sha256(prepared/'precision_first_policy.md')!=m['precision_policy_sha256']
        or sha256(prepared/'model_policy.md')!=m['policy_sha256'] or sha256(prepared/'model_prompt.txt')!=m['prompt_sha256']):
        raise ValueError('Prepared policy/prompt snapshot mismatch')
    f=read_csv(prepared/'reliability_cases.csv')
    if len(f)!=3 or set(f.advertisement_id)!=ALLOWED_IDS:raise ValueError('Only the three approved known-outcome cases are permitted')
    guard=json.loads((prepared/'unknown_outcome_guard.json').read_text(encoding='utf-8'))
    blocked={x['source_sha256'] for x in guard['blocked_sources']}
    if {source_hash(r.job_title,r.description_text) for r in f.itertuples()}&blocked:raise ValueError('Unknown outcome is blocked')
    return m,f,blocked


def run(prepared,output,budget_usd=1,max_ads=3,execute=False,resume=False,session=None,cache_dir=None):
    prepared,output=Path(prepared),Path(output)
    manifest,frame,blocked=validate_prepared(prepared)
    if type(max_ads) is not int or not 1<=max_ads<=3:raise ValueError('Maximum advertisements must be 1..3')
    frame=frame.head(max_ads)
    history_cost=manifest['history_accounted_cost_usd']
    if budget_usd<=history_cost:raise BudgetExceeded('Budget must cover preserved historical usage and unknown hold')
    config=Config(**manifest['config'])
    cache_dir=Path(cache_dir or ROOT/'data/processed/.llm_cache')
    identity=dict(manifest_sha256=sha256(prepared/'pilot_manifest.json'),config=asdict(config),
        max_ads=max_ads,cache_dir=str(cache_dir.resolve()))
    if output.exists() and not resume:raise FileExistsError('Use a new output directory, or --resume with unchanged settings')
    if resume and json.loads((output/'run_identity.json').read_text(encoding='utf-8'))!=identity:raise ValueError('Resume identity mismatch')
    output.mkdir(parents=True,exist_ok=True);cache_dir.mkdir(parents=True,exist_ok=True)
    writable_probe(output);writable_probe(cache_dir)
    with file_lock(output/'run.lock'):
        if not resume:atomic_json(output/'run_identity.json',identity)
        client=Step3FReliableClassifier(config,POLICY_PATH.read_text(encoding='utf-8'),cache_dir,output/'checkpoint.json',
            budget_usd-history_cost,execute=False,blocked_source_hashes=blocked)
        if (output/'checkpoint.json').exists() and client.state.get('identity')!=identity:raise ValueError('Checkpoint identity mismatch')
        client.state['identity']=identity
        if client.spent+history_cost>budget_usd:raise BudgetExceeded('Saved accounting exceeds total application budget')
        if set(client.state['results'])-set(frame.advertisement_id):raise ValueError('Unexpected checkpoint IDs')
        plan=[]
        for r in frame.itertuples():
            check=client.classify(r.advertisement_id,r.job_title,r.description_text) # execute=False throughout preflight
            prior=client.state['results'].get(r.advertisement_id)
            if prior and prior['ai_cache_key']!=client.cache_key(r.job_title,r.description_text):raise ValueError('Prior request identity changed')
            estimate=client.estimate(r.job_title,r.description_text)
            marker=cache_dir/(client.cache_key(r.job_title,r.description_text)+'.pending.json')
            unfinished=any(x['cache_key']==client.cache_key(r.job_title,r.description_text) and x['status'] in {'reserved_before_request','request_outcome_unknown'} for x in client.state['ledger'])
            action='preserve_saved_result' if prior else 'cache_reuse' if check['ai_status']=='cache_hit' else 'blocked_unknown_outcome' if marker.exists() or unfinished else 'new_request'
            if estimate['input_limit_exceeded']:raise ValueError('Input limit exceeded; no source truncation allowed')
            plan.append(dict(advertisement_id=r.advertisement_id,action=action,**estimate))
        if any(p['action']=='blocked_unknown_outcome' for p in plan):raise ValueError('Unrecorded pending outcome blocks this test')
        forecast=dict(selected=len(frame),actions=dict(Counter(x['action'] for x in plan)),
            expected_cost_usd=sum(x['expected_cost_usd'] for x in plan if x['action']=='new_request'),
            conservative_cost_usd=sum(x['reserved_cost_per_attempt_usd'] for x in plan if x['action']=='new_request'),
            preserved_history_accounted_cost_usd=history_cost,new_accounted_cost_usd=client.spent,
            total_accounted_cost_usd=history_cost+client.spent,remaining_budget_usd=budget_usd-history_cost-client.spent,
            application_budget_usd=budget_usd,automatic_retries=0,ssl_unknown_excluded=True,
            prompt_version='step3f-1.1',policy_version='step3f-v1',inclusion_policy='precision-first-v1',
            estimates_not_reported_usage=True,paid_requests_this_invocation=0 if not execute else None)
        atomic_json(output/'dry_run_report.json',forecast)
        pd.DataFrame(plan).to_csv(output/'request_plan.csv',index=False,encoding='utf-8-sig')
        if not execute:return forecast
        client.execute=True;diagnostics=DiagnosticSession(session);client.session=diagnostics
        client.save();stop=None
        try:
            for r in frame.itertuples():
                if r.advertisement_id in client.state['results']:continue  # No automatic retries of failures.
                result=client.classify(r.advertisement_id,r.job_title,r.description_text)
                if diagnostics.last_error:
                    client.state['ledger'][-1]['network_error_diagnostic']=diagnostics.last_error;client.save()
                if result['ai_status'] in {'request_outcome_unknown','http_401','http_402','http_403','http_429'}:
                    stop=result['ai_status'];break
        except BudgetExceeded as exc:stop=str(exc)
        finally:
            rows=[dict(r,**client.state['results'].get(r['advertisement_id'],dict(ai_status='not_executed',ai_relevance=None,ai_primary_category=None))) for r in frame.to_dict('records')]
            write_dataset(output,rows)
            atomic_json(output/'run_summary.json',dict(statuses=dict(Counter(r['ai_status'] for r in rows)),
                new_accounted_cost_usd=client.spent,**client.cost_breakdown(),
                preserved_history_accounted_cost_usd=history_cost,total_accounted_cost_usd=history_cost+client.spent,
                remaining_budget_usd=budget_usd-history_cost-client.spent,stop_reason=stop))
        return forecast


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--budget-usd',type=float,default=1);p.add_argument('--max-ads',type=int,default=3)
    p.add_argument('--resume',action='store_true')
    mode=p.add_mutually_exclusive_group();mode.add_argument('--execute',action='store_true');mode.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    print(json.dumps(run(args.prepared,args.output,args.budget_usd,args.max_ads,args.execute,args.resume),indent=2,ensure_ascii=True))


if __name__=='__main__':main()
