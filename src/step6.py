"""Step 6: unpaid preparation, explicitly enabled classification, and offline publication."""
import argparse
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.dashboard_data import load_dataset, summary, validate
from src.extract_requirements import extract_job
from src.llm_classifier import Config, BudgetExceeded, file_lock
from src.precision_inclusion import InclusionConfig, inclusion_decision
from src.recover_large_pilot import resilient_atomic_json as atomic_json, writable_probe
from src.run_step4 import write_csv, json_rows
from src.step3f_reliable_classifier import Step3FReliableClassifier, source_hash, POLICY_PATH, PROMPT_PATH
from src.step4_jobs import SOURCE, load_existing, reporting_scope
from src.validate_classification import ROOT, read_csv, sha256

BASE = ROOT/'data/processed/step5-available108-20261023'
CACHE = ROOT/'data/processed/.llm_cache'
POLICY = ROOT/'docs/precision_first_inclusion_policy_v1.md'


def identities():
    paths = [POLICY, POLICY_PATH, PROMPT_PATH, SOURCE]
    paths += [ROOT/'src'/n for n in ['step6.py','step3f_reliable_classifier.py','step3f_classifier.py',
              'llm_classifier.py','precision_inclusion.py']]
    return {str(p): sha256(p) for p in paths}


def history_guard():
    """Conservatively exclude every attempted ID, including archived failed attempts."""
    attempted=set(); unknown=set(); keys=set(); snapshots={}
    for path in sorted((ROOT/'data/processed').rglob('checkpoint.json')):
        state=json.loads(path.read_text(encoding='utf-8'))
        snapshots[str(path)]=sha256(path)
        attempted.update(str(i) for i in state.get('results',{}))
        for item in state.get('ledger',[]):
            identifier=str(item.get('advertisement_id',''))
            attempted.add(identifier)
            if item.get('cache_key'):keys.add(item['cache_key'])
            if item.get('status') in {'request_outcome_unknown','reserved_before_request'}:unknown.add(identifier)
    return attempted,unknown,keys,snapshots


def client_for(output, config, budget, execute=False):
    return Step3FReliableClassifier(config,POLICY_PATH.read_text(encoding='utf-8'),CACHE,
        Path(output)/'checkpoint.json',budget,execute=execute,blocked_source_hashes=set())


def prepare(output, size=500, seed=20261024, budget=10):
    output=Path(output)
    if output.exists():raise FileExistsError('Preparation requires a new directory')
    if type(size) is not int or size<1:raise ValueError('Size must be positive')
    if not math.isfinite(budget) or budget<=0:raise ValueError('Budget must be finite and positive')
    registry,_,_=load_existing()
    attempted,unknown,keys,snapshots=history_guard()
    attempted.update(registry.loc[registry.ai_status.ne('not_classified'),'advertisement_id'])
    source=read_csv(SOURCE)
    blocked_hashes={source_hash(r.job_title,r.description_text) for r in source.itertuples() if r.advertisement_id in attempted}
    reserve=set()
    for path in (ROOT/'data/processed').rglob('evaluation_reserve_manifest.json'):
        reserve.update(json.loads(path.read_text(encoding='utf-8')).get('reserved_ids',[]))
    output.mkdir(parents=True); writable_probe(output)
    config=Config('gpt-4.1-mini-2025-04-14',.4,1.6,max_retries=0)
    client=client_for(output,config,budget)
    rows=[]; counts=Counter()
    for r in source.sort_values('advertisement_id').to_dict('records'):
        identifier=r['advertisement_id']; digest=source_hash(r['job_title'],r['description_text'])
        if identifier in unknown:reason='unknown_request_outcome'
        elif identifier in attempted:reason='previously_processed'
        elif digest in blocked_hashes:reason='previously_processed_content'
        elif identifier in reserve:reason='evaluation_reserve'
        elif not r['description_text'].strip():reason='missing_description'
        else:
            try:
                key=client.cache_key(r['job_title'],r['description_text'])
                if (CACHE/(key+'.pending.json')).exists():reason='unknown_request_outcome'
                elif key in keys or (CACHE/(key+'.json')).exists():reason='known_completed_or_attempted_request'
                else:
                    estimate=client.estimate(r['job_title'],r['description_text'])
                    reason='input_limit' if estimate['input_limit_exceeded'] else 'eligible'
            except ValueError:reason='unsupported_source'
        counts[reason]+=1
        if reason=='eligible':rows.append(r)
    eligible=pd.DataFrame(rows)
    # Remove equal source content before sampling: one request per unique title/body.
    eligible['_source_hash']=[source_hash(r.job_title,r.description_text) for r in eligible.itertuples()]
    eligible=eligible.drop_duplicates('_source_hash').drop(columns='_source_hash')
    if len(eligible)<size:raise ValueError('Insufficient untouched advertisements')
    selected=eligible.sample(n=size,random_state=seed).sort_values('advertisement_id')
    write_csv(selected,output/'selected_jobs.csv')
    plans=[dict(advertisement_id=r.advertisement_id,**client.estimate(r.job_title,r.description_text)) for r in selected.itertuples()]
    write_csv(pd.DataFrame(plans),output/'request_plan.csv')
    report=dict(total_collected=len(source),selected=len(selected),selection_seed=seed,
        sampling_method='Uniform sample without replacement from ID-sorted untouched unique title/body sources; evaluation reserve excluded',
        population_breakdown=dict(counts),eligible_unique_sources=len(eligible),
        selected_statuses={'unprocessed':len(selected)},selected_categories=dict(Counter(selected.primary_category)) if 'primary_category' in selected else {},
        expected_cost_usd=sum(p['expected_cost_usd'] for p in plans),
        conservative_cost_usd=sum(p['reserved_cost_per_attempt_usd'] for p in plans),
        configured_input_usd_per_million=.4,configured_output_usd_per_million=1.6,
        pricing_note='Inherited planning rates, not a current provider quote; estimates are not billed usage.',
        application_budget_usd=budget,paid_requests=0,automatic_retries=0,destination_writable=True)
    atomic_json(output/'dry_run_report.json',report)
    atomic_json(output/'manifest.json',dict(identities=identities(),input_sha256=sha256(output/'selected_jobs.csv'),
        config=asdict(config),history_snapshots=snapshots,report=report))
    return report


def run(prepared, output, budget=10, execute=False, resume=False, session=None):
    prepared,output=Path(prepared),Path(output)
    m=json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    if m['identities']!=identities() or m['input_sha256']!=sha256(prepared/'selected_jobs.csv'):
        raise ValueError('Prepared source/code/policy changed; prepare a new run')
    if not math.isfinite(budget) or budget<=0:raise ValueError('Invalid budget')
    frame=read_csv(prepared/'selected_jobs.csv')
    identity=dict(preparation=sha256(prepared/'manifest.json'),cache=str(CACHE.resolve()))
    if output.exists() and not resume:raise FileExistsError('Use a new output or --resume')
    if resume and json.loads((output/'identity.json').read_text(encoding='utf-8'))!=identity:raise ValueError('Resume identity mismatch')
    output.mkdir(parents=True,exist_ok=True); writable_probe(output)
    if execute:CACHE.mkdir(parents=True,exist_ok=True);writable_probe(CACHE)
    with file_lock(output/'run.lock'):
        atomic_json(output/'identity.json',identity)
        client=client_for(output,Config(**m['config']),budget)
        if session is not None:client.session=session
        # History may have advanced since preparation. Own checkpoint is safe to resume.
        attempted,unknown,keys,_=history_guard()
        own=set(client.state['results'])|{str(x['advertisement_id']) for x in client.state['ledger']}
        historical_sources=read_csv(SOURCE)
        forbidden_content={source_hash(r.job_title,r.description_text) for r in historical_sources.itertuples() if r.advertisement_id in attempted-own}
        plan=[]
        for r in frame.itertuples():
            key=client.cache_key(r.job_title,r.description_text)
            if r.advertisement_id in client.state['results']:action='preserve_saved_result'
            elif r.advertisement_id in unknown or (CACHE/(key+'.pending.json')).exists():action='unknown_request_outcome'
            elif r.advertisement_id in attempted-own or source_hash(r.job_title,r.description_text) in forbidden_content:action='previously_processed'
            elif client.cached(r.job_title,r.description_text):action='cache_reuse'
            elif key in keys and r.advertisement_id not in own:action='previously_processed'
            else:action='new_request'
            plan.append(dict(advertisement_id=r.advertisement_id,action=action,**client.estimate(r.job_title,r.description_text)))
        report=dict(actions=dict(Counter(p['action'] for p in plan)),paid_execution_enabled=execute,
            expected_cost_usd=sum(p['expected_cost_usd'] for p in plan if p['action']=='new_request'),
            conservative_cost_usd=sum(p['reserved_cost_per_attempt_usd'] for p in plan if p['action']=='new_request'),
            application_budget_usd=budget,accounted_cost_usd=client.spent,automatic_retries=0)
        atomic_json(output/'preflight.json',report)
        if not execute:return report
        client.execute=True;client.save()
        try:
            for r,p in zip(frame.itertuples(),plan):
                if p['action'] not in {'new_request','cache_reuse'}:continue
                client.classify(r.advertisement_id,r.job_title,r.description_text)
        except BudgetExceeded:
            report['stop_reason']='application_budget_exhausted'
        finally:
            report['accounted_cost_usd']=client.spent
            atomic_json(output/'run_summary.json',report)
        return report


def merge_frames(base_jobs,base_facts,new_jobs,new_facts):
    validate(base_jobs,base_facts)
    if new_jobs.empty:return base_jobs.copy(),base_facts.copy()
    validate(new_jobs,new_facts)
    fresh=new_jobs[~new_jobs.advertisement_id.isin(base_jobs.advertisement_id)]
    jobs=pd.concat([base_jobs,fresh],ignore_index=True)
    facts=pd.concat([base_facts,new_facts[new_facts.advertisement_id.isin(fresh.advertisement_id)]],ignore_index=True)
    validate(jobs,facts)
    return jobs,facts


def publish(prepared,run_dir,output,audit=None):
    output=Path(output)
    if output.exists():raise FileExistsError('Publish requires a new versioned directory')
    m=json.loads((Path(prepared)/'manifest.json').read_text(encoding='utf-8'))
    if m['input_sha256']!=sha256(Path(prepared)/'selected_jobs.csv'):raise ValueError('Selection changed')
    frame=read_csv(Path(prepared)/'selected_jobs.csv')
    if run_dir and (Path(run_dir)/'checkpoint.json').exists():
        expected=dict(preparation=sha256(Path(prepared)/'manifest.json'),cache=str(CACHE.resolve()))
        if json.loads((Path(run_dir)/'identity.json').read_text(encoding='utf-8'))!=expected:raise ValueError('Run does not match preparation')
    state=json.loads((Path(run_dir)/'checkpoint.json').read_text(encoding='utf-8')) if run_dir and (Path(run_dir)/'checkpoint.json').exists() else {'results':{}}
    if set(state['results'])-set(frame.advertisement_id):raise ValueError('Unexpected checkpoint IDs')
    audits=read_csv(audit).set_index('advertisement_id').to_dict('index') if audit else {}
    jobs=[];facts=[];decisions=[];queue=[]
    config=InclusionConfig(scopes=('core_industry','data_ai_governance','academic_ai_ml_research','adjacent_borderline'))
    for original in frame.to_dict('records'):
        identifier=original['advertisement_id'];r=dict(original,**state['results'].get(identifier,{'ai_status':'not_executed'}))
        # Audit files cannot replace source fields or model predictions.
        for k in ['duties_assessment','category_assessment','duties_assessor','inclusion_audit_notes','inclusion_review_resolved']:
            r[k]=audits.get(identifier,{}).get(k,'')
        decision=inclusion_decision(r,config);r.update(decision)
        status=decision['inclusion_status']
        disposition=('accepted' if status=='included' else 'irrelevant' if status=='excluded' else
            'uncertain' if status in {'abstained','out_of_scope'} else 'invalid_response' if r['ai_status']=='invalid_model_response' else
            'unknown_request_outcome' if 'unknown' in r['ai_status'] else status)
        decisions.append(dict(advertisement_id=identifier,status=disposition,**decision))
        if r.get('ai_relevance')=='relevant':queue.append(r)
        if status=='included':
            r.update(classification_source_valid=True,classification_human_verified=False,
                classification_selected_run=str(Path(run_dir).resolve()),step4_disposition='eligible_relevant',
                reporting_scope=reporting_scope(r['ai_primary_category']))
            job,evidence=extract_job(r);jobs.append(job);facts.extend(evidence)
    base_jobs,base_facts=load_dataset(BASE)
    new_jobs=pd.DataFrame(jobs);new_facts=pd.DataFrame(facts,columns=base_facts.columns if not facts else None)
    combined,evidence=merge_frames(base_jobs,base_facts,new_jobs,new_facts)
    output.mkdir(parents=True);writable_probe(output)
    write_csv(combined,output/'jobs_requirements.csv')
    with (output/'jobs_requirements.jsonl').open('w',encoding='utf-8') as f:
        for r in json_rows(combined):f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write_csv(evidence,output/'requirements_evidence.csv')
    write_csv(pd.DataFrame(decisions),output/'classification_decisions.csv')
    write_csv(pd.DataFrame(queue),output/'inclusion_audit_queue.csv')
    audit_template=frame[frame.advertisement_id.isin([r['advertisement_id'] for r in queue])][['advertisement_id','job_title','description_text']].copy()
    for column in ['duties_assessment','category_assessment','duties_assessor','inclusion_audit_notes','inclusion_review_resolved']:audit_template[column]=''
    write_csv(audit_template,output/'inclusion_audit_template.csv')
    registry,_,_=load_existing()
    write_csv(registry,output/'historical_classification_registry.csv')
    atomic_json(output/'classification_provenance.json',dict(prepared_manifest=m,checkpoint=state,audit_path=str(audit) if audit else None))
    counts=Counter(registry.step4_disposition)
    report=summary(combined,evidence)
    report.update(classification_requests=len(state.get('ledger',[])),classification_accounted_cost_usd=sum(x.get('accounted_cost_usd',0) for x in state.get('ledger',[])),total_collected=len(registry),total_successfully_classified=int(registry.classification_source_valid.eq(True).sum())+
        sum(r.get('ai_status') in {'completed','cache_hit'} for r in state['results'].values()),
        total_relevant_analyzed=len(combined),excluded_irrelevant=counts['excluded_irrelevant']+sum(d['status']=='irrelevant' for d in decisions),
        uncertain=counts['set_aside_uncertain']+sum(d['status']=='uncertain' for d in decisions),
        technical_failures=counts['set_aside_technical_failure']+sum(d['status'] in {'invalid_response','technical_failure','unknown_request_outcome'} for d in decisions),
        expansion_statuses=dict(Counter(d['status'] for d in decisions)),baseline_analyzed=len(base_jobs),new_analyzed=len(combined)-len(base_jobs))
    atomic_json(output/'step6_summary.json',report)
    atomic_json(output/'manifest.json',dict(stage='step6-combined',preparation_sha256=sha256(Path(prepared)/'manifest.json'),
        baseline=str(BASE),baseline_sha256=sha256(BASE/'jobs_requirements.jsonl'),
        jobs_sha256=sha256(output/'jobs_requirements.jsonl'),evidence_sha256=sha256(output/'requirements_evidence.csv'),
        audit_sha256=sha256(audit) if audit else None))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','publish'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--prepared',type=Path)
    p.add_argument('--run-dir',type=Path);p.add_argument('--audit',type=Path)
    p.add_argument('--size',type=int,default=500);p.add_argument('--seed',type=int,default=20261024)
    p.add_argument('--budget-usd',type=float,default=10);p.add_argument('--execute',action='store_true')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.execute and a.dry_run:p.error('--execute and --dry-run are mutually exclusive')
    if a.action=='prepare':result=prepare(a.output,a.size,a.seed,a.budget_usd)
    elif a.action=='run':result=run(a.prepared,a.output,a.budget_usd,a.execute,a.resume)
    else:result=publish(a.prepared,a.run_dir,a.output,a.audit)
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__=='__main__':main()
