"""Step 3E: plan/explicitly execute only a prepared, capped 200-ad pilot."""
import argparse
import json
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path

import pandas as pd

from src.classification_policy import policy_flags
from src.llm_classifier import (AI_FIELDS,SCHEMA,VALIDATION_VERSION,Config,CredentialMissing,
                               BudgetExceeded,atomic_json,file_lock,stable_hash)
from src.run_ai_classification import compare_rows,write_results
from src.step3e_classifier import Step3EClassifier,PROMPT_VERSION,POLICY_PATH,PROMPT_PATH
from src.validate_classification import ROOT,read_csv,sha256

ACCEPTED={'completed','cache_hit'}


def validate_prepared(prepared):
    prepared=Path(prepared)
    manifest=json.loads((prepared/'sampling_manifest.json').read_text(encoding='utf-8'))
    sample=read_csv(prepared/'representative_200.csv')
    if manifest.get('stage')!='step3e' or not 1<=manifest['sample_size']<=200:
        raise ValueError('Only a prepared Step 3E pilot of at most 200 is supported')
    if sha256(prepared/'representative_200.csv')!=manifest['sample_sha256']:
        raise ValueError('Prepared sample hash changed')
    if len(sample)!=manifest['sample_size'] or sample.advertisement_id.duplicated().any() or sample.advertisement_id.eq('').any():
        raise ValueError('Prepared sample size/IDs invalid')
    if sample.advertisement_id.tolist()!=manifest['sample_ids'] or set(sample.advertisement_id)&set(manifest['excluded_ids']):
        raise ValueError('Sample identities or development-pilot exclusions changed')
    if set(AI_FIELDS)&set(sample) or any(c.startswith('assistant_') or c.startswith('human_') for c in sample):
        raise ValueError('Model input must not contain AI suggestions or human/reference labels')
    if not {'job_title','description_text','relevance','primary_category','sampling_group'}<=set(sample):
        raise ValueError('Missing original sample fields')
    if set(sample.relevance)-{'relevant','irrelevant','uncertain'} or not sample.sampling_group.equals(sample.relevance):
        raise ValueError('Invalid rule-based sampling strata')
    return sample,manifest


def operational_summary(rows,execute):
    accepted=[r for r in rows if r['ai_status'] in ACCEPTED]
    by_group=[]
    for group in sorted({r['sampling_group'] for r in rows}):
        subset=[r for r in rows if r['sampling_group']==group]
        done=[r for r in subset if r['ai_status'] in ACCEPTED]
        by_group.append(dict(group=group,selected=len(subset),accepted=len(done),unclassified=len(subset)-len(done),
            technical_success_rate=len(done)/len(subset) if execute else None,
            relevance_counts=pd.Series([r['ai_relevance'] for r in done],dtype='object').value_counts().to_dict()))
    return dict(selected=len(rows),accepted=len(accepted),unclassified=len(rows)-len(accepted),
        technical_success_rate=len(accepted)/len(rows) if execute else None,
        comparison=compare_rows(rows),by_sampling_group=by_group,
        low_confidence=sum(r.get('ai_confidence')=='low' for r in accepted),
        review_required=sum(bool(r.get('ai_review_required')) for r in rows),
        status_counts=pd.Series([r['ai_status'] for r in rows]).value_counts().to_dict(),
        human_metrics='Not calculated; rule/AI/assistant opinions are not ground truth')


def run(prepared,config,budget_usd,execute=False,output=None,resume=None,max_ads=200,
        cache_dir=None,session=None,sleep=None):
    if output and resume:raise ValueError('Choose new output or resume')
    sample,manifest=validate_prepared(prepared)
    if not 1<=max_ads<=200 or max_ads>len(sample):raise ValueError('Only 1..200 prepared advertisements are allowed')
    sample=sample.iloc[:max_ads].copy()
    policy=POLICY_PATH.read_text(encoding='utf-8');prompt=PROMPT_PATH.read_text(encoding='utf-8')
    identity=dict(stage='step3e',config=asdict(config),sample_sha256=manifest['sample_sha256'],
        sampling_manifest_sha256=sha256(Path(prepared)/'sampling_manifest.json'),ids=sample.advertisement_id.tolist(),
        policy_sha256=sha256(POLICY_PATH),prompt_sha256=sha256(PROMPT_PATH),prompt_version=PROMPT_VERSION,
        schema_sha256=stable_hash(SCHEMA),validation_version=VALIDATION_VERSION,max_ads=max_ads,
        code_hashes={n:sha256(ROOT/'src'/n) for n in ['run_large_pilot.py','step3e_classifier.py','llm_classifier.py']})
    output=Path(resume) if resume else Path(output) if output else ROOT/'data/processed'/(
        'step3e-run-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if resume:
        plan=json.loads((output/'plan.json').read_text(encoding='utf-8'))
        if plan['identity']!=identity:raise ValueError('Resume identity changed; use a fresh run')
        if sha256(output/'pilot_input.csv')!=plan['pilot_input_sha256']:raise ValueError('Frozen run input changed')
    else:
        output.mkdir(parents=True,exist_ok=False)
        sample.to_csv(output/'pilot_input.csv',index=False,encoding='utf-8-sig')
        atomic_json(output/'plan.json',dict(identity=identity,prepared=str(Path(prepared).resolve()),
            initial_budget_usd=budget_usd,pilot_input_sha256=sha256(output/'pilot_input.csv')))
        for name,path in [('policy_snapshot.md',POLICY_PATH),('prompt_snapshot.txt',PROMPT_PATH),
                          *[(n,ROOT/'src'/n) for n in ['run_large_pilot.py','step3e_classifier.py','llm_classifier.py']]]:
            (output/name).write_bytes(path.read_bytes())
        atomic_json(output/'output_schema.json',SCHEMA)
    with file_lock(output/'run.lock'):
        kwargs={}
        if session is not None:kwargs['session']=session
        if sleep is not None:kwargs['sleep']=sleep
        client=Step3EClassifier(config,policy,cache_dir or ROOT/'data/processed/.llm_cache',
            output/'checkpoint.json',budget_usd,execute=execute,prompt_text=prompt,**kwargs)
        if not resume:(output/'system_prompt.txt').write_text(client.system_prompt,encoding='utf-8')
        estimates=[]
        for row in sample.itertuples():
            estimate=client.estimate(row.job_title,row.description_text)
            cached=client.cached(row.job_title,row.description_text)
            prior=client.state['results'].get(row.advertisement_id)
            estimates.append(dict(advertisement_id=row.advertisement_id,**estimate,
                cache_available=bool(cached),checkpoint_status=prior.get('ai_status') if prior else None,
                request_needed=bool(row.description_text.strip()) and not cached and not prior and not estimate['input_limit_exceeded']))
        pd.DataFrame(estimates).to_csv(output/'token_cost_plan.csv',index=False,encoding='utf-8-sig')
        needed=[e for e in estimates if e['request_needed']]
        forecast=dict(selected=len(sample),cached=sum(e['cache_available'] for e in estimates),uncached_requests_needed=len(needed),
            checkpoint_records=sum(e['checkpoint_status'] is not None for e in estimates),
            missing_descriptions=int(sample.description_text.str.strip().eq('').sum()),
            input_limit_exceeded=sum(e['input_limit_exceeded'] for e in estimates),
            expected_input_tokens=sum(e['expected_input_tokens'] for e in needed),
            expected_output_tokens=sum(e['expected_output_tokens'] for e in needed),
            conservative_input_tokens=sum(e['reserved_input_tokens'] for e in needed),
            conservative_output_tokens=sum(e['reserved_output_tokens'] for e in needed),
            expected_first_pass_usd=sum(e['expected_cost_usd'] for e in needed),
            conservative_first_pass_usd=sum(e['reserved_cost_per_attempt_usd'] for e in needed),
            conservative_all_attempts_usd=sum(e['reserved_cost_per_attempt_usd'] for e in needed)*(config.max_retries+1),
            estimation='Expected chars/3.5+256; conservative UTF-8 payload bytes+2048 plus capped output. Forecast, not provider token counting.')
        atomic_json(output/'cost_forecast.json',forecast)
        print(json.dumps(dict(mode='execute' if execute else 'dry_run',budget_usd=budget_usd,forecast=forecast),indent=2))
        start=len(client.state['ledger']);rows=[];stop=None
        for source in sample.to_dict('records'):
            if stop:
                result=dict(ai_status=stop,ai_review_required=True,ai_human_verified=False)
            else:
                try:result=client.classify(source['advertisement_id'],source['job_title'],source['description_text'])
                except CredentialMissing:
                    stop='credential_required';result=dict(ai_status=stop,ai_review_required=True,ai_human_verified=False)
                except BudgetExceeded:
                    stop='budget_stopped';result=dict(ai_status=stop,ai_review_required=True,ai_human_verified=False)
                if result['ai_status'] in {'http_401','http_402','http_403','http_429'}:
                    stop='not_attempted_api_access_failure'
            result={**source,**dict.fromkeys(AI_FIELDS),**result}
            result.update(ai_prompt_version=PROMPT_VERSION,ai_validation_version=VALIDATION_VERSION,
                ai_model=config.model,ai_provider=config.provider,ai_policy_sha256=client.policy_hash,
                ai_schema_sha256=stable_hash(SCHEMA),ai_source_run=str(output.resolve()))
            if result.get('ai_review_required') is None:result['ai_review_required']=True
            if result.get('ai_relevance') in {'relevant','irrelevant','uncertain'}:
                result.update({'ai_'+k:v for k,v in policy_flags(source['job_title'],result['ai_relevance'],result['ai_primary_category']).items()})
            rows.append(result)
            if execute:write_results(output,rows)
        write_results(output,rows)
        operational=operational_summary(rows,execute)
        status=('dry_run_only' if not execute else 'completed' if operational['accepted']==len(rows) else
                stop or 'partial_requires_inspection')
        summary=dict(stage='step3e',mode='execute' if execute else 'dry_run',status=status,
            prompt_version=PROMPT_VERSION,validation_version=VALIDATION_VERSION,model=config.model,
            sampling_scope=manifest['sample_scope'],forecast=forecast,**operational,
            budget_usd=budget_usd,accounted_cost_usd=client.spent,**client.cost_breakdown(),
            attempts_reserved=len(client.state['ledger']),new_attempts_reserved=len(client.state['ledger'])-start,
            provider_reported_input_tokens=sum(e['prompt_tokens'] or 0 for e in client.state['ledger']),
            provider_reported_output_tokens=sum(e['completion_tokens'] or 0 for e in client.state['ledger']),
            human_verified_assigned=False,
            resume_command=['python','-m','src.run_large_pilot','--prepared',str(Path(prepared).resolve()),
                '--resume',str(output.resolve()),'--max-ads',str(max_ads),'--budget-usd',str(budget_usd),
                '--model',config.model,'--max-retries',str(config.max_retries),
                '--input-usd-per-million',str(config.input_usd_per_million),'--output-usd-per-million',str(config.output_usd_per_million),
                '--timeout',str(config.timeout),'--max-output-tokens',str(config.max_output_tokens),'--execute'])
        atomic_json(output/'run_summary.json',summary)
        return output,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',required=True,type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--resume',type=Path)
    parser.add_argument('--max-ads',type=int,default=200)
    parser.add_argument('--budget-usd',type=float,required=True)
    parser.add_argument('--model',default='gpt-4.1-mini')
    parser.add_argument('--input-usd-per-million',type=float,default=.4)
    parser.add_argument('--output-usd-per-million',type=float,default=1.6)
    parser.add_argument('--max-retries',type=int,default=0)
    parser.add_argument('--timeout',type=float,default=60)
    parser.add_argument('--max-output-tokens',type=int,default=1024)
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--execute',action='store_true');modes.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    config=Config(model=args.model,input_usd_per_million=args.input_usd_per_million,
        output_usd_per_million=args.output_usd_per_million,max_retries=args.max_retries,
        timeout=args.timeout,max_output_tokens=args.max_output_tokens)
    output,summary=run(args.prepared,config,args.budget_usd,execute=args.execute,output=args.output,resume=args.resume,max_ads=args.max_ads)
    print(json.dumps(dict(output=str(output),status=summary['status'],accounted_cost_usd=summary['accounted_cost_usd']),indent=2))
    return 0 if summary['status'] in {'completed','dry_run_only'} else 2


if __name__=='__main__':raise SystemExit(main())
