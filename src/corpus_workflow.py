"""Full collected-corpus workflow: unpaid prepare, opt-in budgeted batches, offline publish."""
import argparse
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src import preview_duties_v3 as duties
from src.activate_duties_v3 import DEFAULT_OUTPUT as V3_BASE, ACTIVE_POLICY
from src.dashboard_data import latest_dataset, load_dataset, summary, validate
from src.extract_requirements import extract_job
from src.llm_classifier import Config, BudgetExceeded, ClassificationError, file_lock, stable_hash
from src.recover_large_pilot import resilient_atomic_json as atomic_json, writable_probe
from src.run_step4 import write_csv, json_rows
from src.step3f_reliable_classifier import (Step3FReliableClassifier,source_hash,POLICY_PATH,PROMPT_PATH)
from src.step4_jobs import SOURCE, load_existing, validate_classification
from src.step6 import merge_frames
from src.validate_classification import ROOT,read_csv,sha256

CACHE=ROOT/'data/processed/.llm_cache'
AUTHORIZATION='User requested applying the approved precision-first v3 duties rules to the remaining collected corpus; no per-ad preview allowlist. Paid classification still requires --execute.'
SCOPE_POLICY=ROOT/'docs/precision_first_v3_corpus_scope.md'


def config():return Config('gpt-4.1-mini-2025-04-14',.4,1.6,max_retries=0)


def code_identity():
    names=['corpus_workflow.py','preview_duties_v3.py','step3f_reliable_classifier.py',
        'step3f_classifier.py','llm_classifier.py','extract_requirements.py','step4_jobs.py']
    paths=[ROOT/'src'/name for name in names]+[SOURCE,ACTIVE_POLICY,SCOPE_POLICY,POLICY_PATH,PROMPT_PATH]
    return {str(p):sha256(p) for p in paths}


def client(output,budget=60,cache=None,session=None):
    cache=Path(cache or CACHE)
    return Step3FReliableClassifier(config(),POLICY_PATH.read_text(encoding='utf-8'),cache,
        Path(output)/'checkpoint.json',budget,execute=False,blocked_source_hashes=set(),session=session)


def historical(source):
    by_id=source.set_index('advertisement_id').to_dict('index')
    choices={};attempted=set();unknown=set();keys=set();snapshots={}
    # Canonical legacy decisions plus every checkpoint (archive copies are not new attempts).
    _,history,_=load_existing()
    candidates=[(h['advertisement_id'],h['prediction'],h['source_run']) for h in history]
    for path in sorted((ROOT/'data/processed').rglob('checkpoint.json'),key=lambda p:(p.stat().st_mtime_ns,str(p))):
        state=json.loads(path.read_text(encoding='utf-8'));snapshots[str(path)]=sha256(path)
        for identifier,prediction in state.get('results',{}).items():
            if identifier not in by_id:continue
            candidates.append((identifier,prediction,str(path)))
            if prediction.get('ai_status') not in {'dry_run','not_executed','not_classified'}:attempted.add(identifier)
        for entry in state.get('ledger',[]):
            identifier=str(entry.get('advertisement_id',''))
            if identifier in by_id:
                attempted.add(identifier)
                if entry.get('status') in {'reserved_before_request','request_outcome_unknown'}:unknown.add(identifier)
            if entry.get('cache_key'):keys.add(entry['cache_key'])
    for identifier,prediction,origin in candidates:
        if identifier not in by_id:continue
        valid,_=validate_classification(prediction,by_id[identifier])
        if not valid:continue
        rank=prediction.get('ai_prompt_version')=='step3f-1.1'
        if identifier not in choices or rank>=choices[identifier][0]:choices[identifier]=(rank,prediction,origin)
    return choices,attempted,unknown,keys,snapshots


def reconcile(source,engine,base_jobs):
    """One coverage row per collected ID; request unit is unique original source text."""
    if source.advertisement_id.duplicated().any():raise ValueError('Duplicate collected IDs')
    choices,attempted,unknown,keys,snapshots=historical(source)
    content=lambda r:source_hash(r['job_title'],r['description_text'])
    by_id=source.set_index('advertisement_id',drop=False).to_dict('index')
    attempted_hashes={content(by_id[i]) for i in attempted}
    unknown_hashes={content(by_id[i]) for i in unknown}
    source_choices={}
    for identifier,(rank,prediction,origin) in choices.items():
        digest=content(by_id[identifier])
        if digest not in source_choices or rank>source_choices[digest][0]:source_choices[digest]=(rank,prediction,origin,identifier)
    predictions={};rows=[];memo={};representatives={};base_ids=set(base_jobs.advertisement_id)
    base_predictions={r['advertisement_id']:{k:v for k,v in r.items() if k.startswith('ai_')} for r in json_rows(base_jobs)}
    for r in source.sort_values('advertisement_id').to_dict('records'):
        identifier=r['advertisement_id'];digest=content(r)
        row=dict(advertisement_id=identifier,source_sha256=digest,
            unknown_request_outcome=digest in unknown_hashes,baseline_analyzed=identifier in base_ids,
            historical_attempted=identifier in attempted,direct_historical_success=identifier in choices,successfully_classified=False,
            compatible_classification=False,compatible_cache=False,classification_relevance=None,
            classification_origin=None,reused_from_advertisement_id=None,request_key=None,
            representative_id=identifier,request_action=None)
        if digest not in memo:
            try:
                key=engine.cache_key(r['job_title'],r['description_text'])
                cached=engine.cached(r['job_title'],r['description_text'])
                estimate=engine.estimate(r['job_title'],r['description_text'])
                memo[digest]=(key,cached,estimate,None)
            except (ClassificationError,KeyError,TypeError,json.JSONDecodeError) as exc:
                memo[digest]=(None,None,{},type(exc).__name__)
        key,cached,estimate,error=memo[digest];row['request_key']=key
        prediction=None;origin=None;reused=None
        if identifier in choices:
            _,prediction,origin=choices[identifier]
        elif digest in source_choices:
            _,prediction,origin,reused=source_choices[digest]
        if identifier in base_ids and prediction is None:
            prediction=base_predictions[identifier];origin='preserved_analyzed_baseline'
        if cached and (prediction is None or prediction.get('ai_prompt_version')!='step3f-1.1' or prediction.get('ai_cache_key')!=key):
            prediction=engine.metadata(key,cached,cache_hit=True);origin=str(engine.cache_dir/(key+'.json'));reused=None
        compatible=bool(prediction and prediction.get('ai_prompt_version')=='step3f-1.1' and prediction.get('ai_cache_key')==key)
        if prediction:
            prediction=dict(prediction)
            if reused:prediction['ai_reused_from_advertisement_id']=reused
            predictions[identifier]=prediction
            row.update(successfully_classified=True,compatible_classification=compatible,
                compatible_cache=bool(cached),classification_relevance=prediction.get('ai_relevance'),
                classification_origin=origin,reused_from_advertisement_id=reused,
                request_action='reuse_compatible' if compatible else 'preserve_legacy_completed')
        elif cached:
            raise RuntimeError('Unreachable compatible receipt')
        elif digest in unknown_hashes or (key and (engine.cache_dir/(key+'.pending.json')).exists()):
            row.update(request_action='blocked_unknown_outcome',unknown_request_outcome=True)
        elif error:row['request_action']='blocked_invalid_cache_or_source'
        elif digest in attempted_hashes or key in keys:row['request_action']='skip_previous_failure_or_attempt'
        elif not r['description_text'].strip():row['request_action']='skip_missing_description'
        elif estimate.get('input_limit_exceeded'):row['request_action']='skip_input_limit'
        elif digest in representatives:
            row.update(request_action='shared_source',representative_id=representatives[digest])
        else:
            row.update(request_action='new_request',**estimate);representatives[digest]=identifier
        rows.append(row)
    return pd.DataFrame(rows),predictions,snapshots


def prepare(output,budget=60):
    output=Path(output)
    if output.exists():raise FileExistsError('Use a new preparation directory')
    if not math.isfinite(budget) or budget<=0:raise ValueError('Budget must be finite and positive')
    frozen=json.loads((V3_BASE/'policy_activation.json').read_text(encoding='utf-8'))
    if frozen['rules_sha256']!=sha256(Path(duties.__file__)):raise ValueError('Approved v3 rules changed')
    source=read_csv(SOURCE)
    if len(source)!=5023:raise ValueError('Unexpected original collection size')
    baseline=latest_dataset(ROOT/'data/processed');base_jobs,_=load_dataset(baseline)
    engine=client(output,budget)
    coverage,predictions,snapshots=reconcile(source,engine,base_jobs)
    output.mkdir(parents=True);writable_probe(output)
    # Probe the shared cache without modifying any existing receipt or pending marker.
    CACHE.mkdir(parents=True,exist_ok=True);writable_probe(CACHE)
    write_csv(source,output/'collected_jobs.csv')
    write_csv(coverage,output/'initial_coverage.csv')
    workload=coverage[coverage.request_action.eq('new_request')]
    write_csv(workload,output/'request_plan.csv')
    atomic_json(output/'reusable_predictions.json',predictions)
    report=dict(total_collected=len(source),existing_analyzed=len(base_jobs),
        unique_successfully_classified=int(coverage.successfully_classified.sum()),
        direct_historical_successful_ids=int(coverage.direct_historical_success.sum()),
        never_successfully_classified=len(source)-int(coverage.direct_historical_success.sum()),
        without_successful_result_after_reuse=int((~coverage.successfully_classified).sum()),
        compatible_classification_ads=int(coverage.compatible_classification.sum()),
        compatible_cache_ads=int(coverage.compatible_cache.sum()),
        compatible_cache_unique_requests=coverage.loc[coverage.compatible_cache,'request_key'].nunique(),
        actions=dict(Counter(coverage.request_action)),new_paid_requests_planned=len(workload),
        new_ads_served_by_planned_requests=int(coverage.request_action.isin(['new_request','shared_source']).sum()),
        expected_cost_usd=float(workload.expected_cost_usd.sum()) if len(workload) else 0,
        conservative_cost_usd=float(workload.reserved_cost_per_attempt_usd.sum()) if len(workload) else 0,
        application_budget_usd=budget,model=config().model,input_usd_per_million=.4,output_usd_per_million=1.6,
        pricing_note='Inherited configured planning rates, not a current provider price quote or invoice.',
        order='All remaining unique title/body sources in advertisement-ID order; no sample. Identical source aliases share a single receipt.',
        automatic_retries=0,unknown_outcome_retry=False,paid_requests_made=0,
        destination_writable=True,full_classification_not_required=True)
    atomic_json(output/'dry_run_report.json',report)
    atomic_json(output/'manifest.json',dict(stage='corpus-prepared',code_identity=code_identity(),config=asdict(config()),
        source_sha256=sha256(SOURCE),input_sha256=sha256(output/'collected_jobs.csv'),
        predictions_sha256=sha256(output/'reusable_predictions.json'),coverage_sha256=sha256(output/'initial_coverage.csv'),
        baseline=str(baseline),baseline_jobs_sha256=sha256(baseline/'jobs_requirements.jsonl'),
        baseline_facts_sha256=sha256(baseline/'requirements_evidence.csv'),history_snapshots=snapshots,
        scope_authorization=AUTHORIZATION,report=report))
    return report


def prepared_inputs(prepared):
    prepared=Path(prepared);m=json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    if m['code_identity']!=code_identity():raise ValueError('Code/source/policy changed; prepare a fresh workload')
    for name,field in [('collected_jobs.csv','input_sha256'),('reusable_predictions.json','predictions_sha256'),('initial_coverage.csv','coverage_sha256')]:
        if sha256(prepared/name)!=m[field]:raise ValueError('Prepared input identity mismatch')
    baseline=Path(m['baseline'])
    if sha256(baseline/'jobs_requirements.jsonl')!=m['baseline_jobs_sha256'] or sha256(baseline/'requirements_evidence.csv')!=m['baseline_facts_sha256']:
        raise ValueError('Baseline changed')
    return m,read_csv(prepared/'collected_jobs.csv'),load_dataset(baseline)


def run(prepared,output,budget=60,batch_size=100,max_batches=1,execute=False,resume=False,session=None):
    output=Path(output);prepared=Path(prepared)
    if type(batch_size) is not int or batch_size<1 or type(max_batches) is not int or max_batches<1:raise ValueError('Positive batch limits required')
    if not math.isfinite(budget) or budget<=0:raise ValueError('Budget must be finite and positive')
    m,source,(base_jobs,_)=prepared_inputs(prepared)
    identity=dict(prepared_sha256=sha256(prepared/'manifest.json'),cache=str(CACHE.resolve()),config=m['config'])
    if output.exists() and not resume:raise FileExistsError('Use a new run directory or --resume')
    if resume and json.loads((output/'identity.json').read_text(encoding='utf-8'))!=identity:raise ValueError('Resume identity mismatch')
    output.mkdir(parents=True,exist_ok=True);writable_probe(output);writable_probe(CACHE)
    with file_lock(output/'run.lock'):
        atomic_json(output/'identity.json',identity)
        engine=client(output,budget,session=session)
        if engine.spent>budget:raise BudgetExceeded('Saved charges/unknown holds exceed the application budget')
        if engine.state.get('identity',identity)!=identity:raise ValueError('Checkpoint identity mismatch')
        engine.state['identity']=identity
        coverage,predictions,_=reconcile(source,engine,base_jobs)
        pending=coverage[coverage.request_action.eq('new_request')].head(batch_size*max_batches)
        report=dict(execute=execute,remaining_new_requests=int(coverage.request_action.eq('new_request').sum()),
            invocation_request_cap=batch_size*max_batches,planned_this_invocation=len(pending),
            expected_cost_usd=float(pending.expected_cost_usd.sum()) if len(pending) else 0,
            conservative_cost_usd=float(pending.reserved_cost_per_attempt_usd.sum()) if len(pending) else 0,
            application_budget_usd=budget,accounted_cost_usd=engine.spent,automatic_retries=0,
            actions=dict(Counter(coverage.request_action)),paid_requests_this_invocation=0)
        atomic_json(output/'preflight.json',report)
        if not execute:return report
        by_id=source.set_index('advertisement_id').to_dict('index')
        engine.execute=True;engine.save();start=len(engine.state['ledger']);unknown_streak=0;stop=None
        # Serialize paid execution across corpus runners; per-key classifier locks also remain active.
        with file_lock(CACHE/'corpus-execution.lock'):
            try:
                for index,item in enumerate(pending.to_dict('records')):
                    identifier=item['advertisement_id'];r=by_id[identifier]
                    # Check historical changes after preflight; never pay again after another completed run.
                    if index%batch_size==0:
                        choices,attempted,unknown,_,_=historical(source)
                        blocked_sources={source_hash(by_id[i]['job_title'],by_id[i]['description_text']) for i in attempted}
                    if identifier in choices or source_hash(r['job_title'],r['description_text']) in blocked_sources:continue
                    result=engine.classify(identifier,r['job_title'],r['description_text'])
                    unknown_streak=unknown_streak+1 if result['ai_status']=='request_outcome_unknown' else 0
                    if (index+1)%batch_size==0:
                        atomic_json(output/('batch_'+str((index+1)//batch_size).zfill(4)+'_'+str(len(engine.state['ledger'])).zfill(5)+'.json'),
                            dict(completed_results=len(engine.state['results']),accounted_cost_usd=engine.spent))
                    if result['ai_status'] in {'http_401','http_402','http_403','http_429'} or unknown_streak>=3:
                        stop=result['ai_status'];break
            except BudgetExceeded:stop='application_budget_exhausted'
            finally:
                report.update(paid_requests_this_invocation=len(engine.state['ledger'])-start,
                    accounted_cost_usd=engine.spent,stop_reason=stop,completed_results=len(engine.state['results']),
                    cost_breakdown=engine.cost_breakdown())
                final_coverage,_,_=reconcile(source,engine,base_jobs)
                report['remaining_new_requests']=int(final_coverage.request_action.eq('new_request').sum())
                atomic_json(output/'run_summary.json',report)
        return report


def assess(row,prediction):
    original=dict(row,**prediction);audit=duties.decision(original)
    passing=audit['preview_decision']=='would_qualify_v3'
    status=('included' if passing else audit['reporting_lane'] if audit['preview_decision']=='separate_lane' else
        'abstention' if audit['preview_decision']=='abstain' else
        'technical_failure' if audit['preview_decision']=='technical_failure' else audit['saved_relevance'])
    receipt=dict(audit,activated=True,approved=passing,policy_version='precision-first-v3',
        inclusion_policy_version='precision-first-v3',inclusion_status=status,
        inclusion_method='automated_deterministic' if passing else 'v3_abstention_or_exclusion',
        authorization=AUTHORIZATION,human_reviewed=False)
    receipt['audit_receipt_sha256']=stable_hash(receipt)
    return original,receipt


def publish(prepared,run_dir,output,finalize=False):
    output=Path(output);prepared=Path(prepared);run_dir=Path(run_dir)
    if output.exists():raise FileExistsError('Use a new publication directory')
    protected={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'data').rglob('*') if p.is_file() and not p.is_relative_to(run_dir.resolve())}
    m,source,(base_jobs,base_facts)=prepared_inputs(prepared)
    expected=dict(prepared_sha256=sha256(prepared/'manifest.json'),cache=str(CACHE.resolve()),config=m['config'])
    if json.loads((run_dir/'identity.json').read_text(encoding='utf-8'))!=expected:raise ValueError('Run identity mismatch')
    engine=client(run_dir)
    coverage,predictions,_=reconcile(source,engine,base_jobs)
    originals=source.set_index('advertisement_id',drop=False).to_dict('index')
    jobs=[];facts=[];receipts=[];base_ids=set(base_jobs.advertisement_id)
    for index,row in coverage.iterrows():
        identifier=row.advertisement_id;prediction=predictions.get(identifier)
        if identifier in base_ids:status='preserved_analyzed'
        elif prediction and row.compatible_classification:
            original,receipt=assess(originals[identifier],prediction);receipts.append(receipt)
            status=receipt['inclusion_status']
            if status=='included':
                enriched=dict(original,classification_source_valid=True,classification_human_verified=False,
                    classification_selected_run=row.classification_origin,step4_disposition='eligible_relevant',
                    reporting_scope='core_industry',inclusion_method='automated_deterministic',
                    inclusion_policy_version='precision-first-v3',inclusion_human_reviewed=False,
                    inclusion_audit_receipt_sha256=receipt['audit_receipt_sha256'],
                    inclusion_source_sha256=receipt['source_sha256'],inclusion_duties_evidence=receipt['duty_evidence'],
                    inclusion_audit_engine_version=duties.VERSION)
                job,evidence=extract_job(enriched);jobs.append(job);facts.extend(evidence)
        elif prediction:
            status='legacy_'+prediction['ai_relevance']
        else:
            status={'new_request':'unprocessed_set_aside' if finalize else 'unprocessed',
                'shared_source':'unprocessed_set_aside' if finalize else 'unprocessed_shared_source','skip_previous_failure_or_attempt':'technical_failure' if row.historical_attempted else 'skipped_prior_source_attempt',
                'blocked_unknown_outcome':'unknown_request_outcome'}.get(row.request_action,row.request_action)
        coverage.loc[index,'final_processing_status']=status
    new_jobs=pd.DataFrame(jobs);new_facts=pd.DataFrame(facts) if facts else base_facts.iloc[:0].copy()
    merged,evidence=merge_frames(base_jobs,base_facts,new_jobs,new_facts);validate(merged,evidence)
    output.mkdir(parents=True);writable_probe(output)
    write_csv(merged,output/'jobs_requirements.csv')
    with (output/'jobs_requirements.jsonl').open('w',encoding='utf-8') as f:
        for row in json_rows(merged):f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    write_csv(evidence,output/'requirements_evidence.csv')
    write_csv(coverage,output/'corpus_coverage.csv')
    write_csv(merged[['advertisement_id','inclusion_method','inclusion_policy_version','inclusion_human_reviewed']],output/'job_inclusion_provenance.csv')
    with (output/'inclusion_audit_receipts.jsonl').open('w',encoding='utf-8') as f:
        for receipt in receipts:f.write(json.dumps(receipt,ensure_ascii=False,allow_nan=False)+'\n')
    atomic_json(output/'classification_provenance.json',predictions)
    snapshot=output/'run_snapshot';snapshot.mkdir()
    for name in ['identity.json','checkpoint.json','preflight.json','run_summary.json']:
        if (run_dir/name).exists():(snapshot/name).write_bytes((run_dir/name).read_bytes())
    for group,frame in coverage.groupby('final_processing_status'):
        write_csv(frame,output/('group_'+group+'.csv'))
    statuses=Counter(coverage.final_processing_status)
    report=summary(merged,evidence)
    report.update(total_collected=len(source),total_successfully_classified=int(coverage.successfully_classified.sum()),
        total_relevant_analyzed=len(merged),excluded_irrelevant=int(coverage.classification_relevance.eq('irrelevant').sum()),
        uncertain=int(coverage.classification_relevance.eq('uncertain').sum())+statuses['abstention'],
        technical_failures=statuses['technical_failure']+statuses['unknown_request_outcome']+statuses['blocked_invalid_cache_or_source'],
        active_inclusion_policy='precision-first-v3',coverage_scope='all_collected_ids',expansion_statuses=dict(statuses),
        baseline_analyzed=len(base_jobs),new_analyzed=len(merged)-len(base_jobs),
        new_usable_requirements_jobs=sum(bool(j['requirements_data_usable']) for j in jobs),
        pending_new_requests=int(coverage.request_action.eq('new_request').sum()),finalized=finalize,
        selection_caveat='Analyzed accepted subset of the collected search-based corpus, not the entire Swedish market. Unclassified and excluded IDs are explicitly retained in coverage.',
        publication_api_requests=0,accounted_classification_cost_this_run_usd=engine.spent)
    atomic_json(output/'step6_summary.json',report)
    write_csv(pd.DataFrame(report['coverage']),output/'requirement_coverage.csv')
    atomic_json(output/'policy_activation.json',dict(policy_version='precision-first-v3',active=True,
        scope='full_collected_corpus',authorization=AUTHORIZATION,new_unique_jobs=report['new_analyzed'],
        policy_sha256=sha256(ACTIVE_POLICY),scope_sha256=sha256(SCOPE_POLICY),rules_sha256=sha256(Path(duties.__file__))))
    text='# Collected-corpus job market analysis\n\n'
    text+=f"{len(source)} collected IDs; {report['total_successfully_classified']} successfully classified; {len(merged)} accepted analyzed jobs; {report['usable_requirements_jobs']} with usable requirements evidence.\n\n"
    text+='Processing statuses (unique advertisement IDs):\n\n'+pd.DataFrame(sorted(statuses.items()),columns=['status','ads']).to_string(index=False)+'\n\n'
    text+='Requirement coverage (denominator = accepted analyzed jobs):\n\n'+pd.DataFrame(report['coverage']).to_string(index=False)+'\n\n'
    text+='Requested skills and technologies:\n\n'+pd.DataFrame(report['requested_skills']).to_string(index=False)+'\n\n'
    text+=report['limitations']+' '+report['selection_caveat']+'\n'
    (output/'report.md').write_text(text,encoding='utf-8')
    saved_jobs,saved_facts=load_dataset(output)
    old=saved_jobs[saved_jobs.advertisement_id.isin(base_ids)][base_jobs.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(old,base_jobs,check_dtype=False)
    pd.testing.assert_frame_equal(saved_facts[saved_facts.advertisement_id.isin(base_ids)].reset_index(drop=True),base_facts)
    if set(coverage.advertisement_id)!=set(source.advertisement_id) or coverage.advertisement_id.duplicated().any():raise ValueError('Incomplete corpus coverage')
    if any(not (ROOT/p).is_file() or sha256(ROOT/p)!=h for p,h in protected.items()):raise RuntimeError('Historical data changed')
    atomic_json(output/'preservation_manifest.json',dict(files=protected,unchanged_files_verified=len(protected),mutable_run_excluded=str(run_dir.resolve()),run_snapshotted=True))
    atomic_json(output/'manifest.json',dict(stage='step6-combined',active_inclusion_policy='precision-first-v3',
        coverage_scope='all_collected_ids',jobs_sha256=sha256(output/'jobs_requirements.jsonl'),
        evidence_sha256=sha256(output/'requirements_evidence.csv'),coverage_sha256=sha256(output/'corpus_coverage.csv')))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','publish'])
    p.add_argument('--prepared',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-dir',type=Path)
    p.add_argument('--budget-usd',type=float,default=60);p.add_argument('--batch-size',type=int,default=100);p.add_argument('--max-batches',type=int,default=1)
    modes=p.add_mutually_exclusive_group();modes.add_argument('--execute',action='store_true');modes.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--finalize',action='store_true')
    a=p.parse_args()
    if a.action=='prepare':result=prepare(a.output,a.budget_usd)
    elif a.action=='run':result=run(a.prepared,a.output,a.budget_usd,a.batch_size,a.max_batches,a.execute,a.resume)
    else:result=publish(a.prepared,a.run_dir,a.output,a.finalize)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
