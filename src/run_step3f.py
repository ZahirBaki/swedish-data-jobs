"""Prepare and run the versioned full-dataset classifier; dry-run is the default."""
import argparse
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.llm_classifier import BudgetExceeded, Config, file_lock, stable_hash
from src.recover_large_pilot import resilient_atomic_json as atomic_json, writable_probe
from src.step3f_classifier import (Step3FClassifier, POLICY_PATH, PROMPT_PATH,
    POLICY_VERSION, PROMPT_VERSION, VALIDATION_VERSION, SCHEMA, AI_FIELDS,
    validate_span_prediction)
from src.validate_classification import ROOT, read_csv, sha256, latest_step3b

DEFAULT_PILOT=ROOT/'data/processed/step3e-analysis-20261022/dataset/pilot_results.csv'
DEFAULT_REVIEW=ROOT/'data/reviews/step3e_200_ai_assisted_review.csv'
CODE_PATHS=[ROOT/'src'/name for name in ['run_step3f.py','step3f_classifier.py',
    'llm_classifier.py','recover_large_pilot.py']]


def unique_ids(frame):
    required={'advertisement_id','job_title','description_text','relevance','primary_category'}
    if not required.issubset(frame.columns):
        raise ValueError('Missing required source columns: '+str(sorted(required-set(frame.columns))))
    if frame.advertisement_id.eq('').any() or frame.advertisement_id.duplicated().any():
        raise ValueError('Advertisement IDs must be nonempty and unique')


def review_analysis(pilot,review,output):
    unique_ids(pilot); unique_ids(review)
    if set(pilot.advertisement_id)!=set(review.advertisement_id):
        raise ValueError('Review and pilot IDs must match')
    reference=pilot.set_index('advertisement_id')
    for row in review.to_dict('records'):
        original=reference.loc[row['advertisement_id']]
        if row['job_title']!=original.job_title or row['description_text']!=original.description_text:
            raise ValueError('Review source content differs from pilot')
    accepted=review.ai_status.isin(['completed','cache_hit'])
    review=review.copy()
    review['accepted_prediction']=accepted
    review['provisional_relevance_disagreement']=accepted & review.ai_relevance.ne(review.assistant_suggested_relevance)
    review['provisional_category_disagreement']=accepted & review.ai_primary_category.ne(review.assistant_suggested_primary_category)
    review['review_priority']=((~accepted).astype(int)*4
        +review.provisional_relevance_disagreement.astype(int)*3
        +review.provisional_category_disagreement.astype(int)*2
        +review.assistant_suggested_relevance.eq('uncertain').astype(int))
    review.sort_values(['review_priority','advertisement_id'],ascending=[False,True]).to_csv(
        output/'policy_review_200.csv',index=False,encoding='utf-8-sig')
    summary=dict(total=len(review),accepted_original_predictions=int(accepted.sum()),
        original_technical_failures=int((~accepted).sum()),
        provisional_suggestion_counts=dict(Counter(review.assistant_suggested_relevance)),
        relevance_disagreements=int(review.provisional_relevance_disagreement.sum()),
        category_disagreements=int(review.provisional_category_disagreement.sum()),
        unique_disagreement_ads=int((review.provisional_relevance_disagreement|review.provisional_category_disagreement).sum()),
        human_verified_labels_used=0,labels_are_ground_truth=False)
    atomic_json(output/'review_analysis.json',summary)
    return summary


def prepare(source,output,pilot=DEFAULT_PILOT,review=DEFAULT_REVIEW):
    source,pilot,review,output=map(Path,[source,pilot,review,output])
    if output.exists():raise FileExistsError('Use a new preparation directory')
    frame=read_csv(source); unique_ids(frame)
    previous=read_csv(pilot); suggestions=read_csv(review)
    # Inventory existing data before creating new artifacts. Nothing inventoried is written.
    protected={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'data').rglob('*') if p.is_file()}
    output.mkdir(parents=True)
    analysis=review_analysis(previous,suggestions,output)
    histories=[('previous_step3e_',previous,pilot)]
    combined=ROOT/'data/processed/step3d-combined-v2-20261022/dataset/pilot_results.csv'
    if combined.exists():histories.append(('previous_step3d_',read_csv(combined),combined))
    for prefix,history,path in histories:
        unique_ids(history)
        intersect=frame.merge(history[['advertisement_id','job_title','description_text']],on='advertisement_id',suffixes=('','_prior'))
        if not (intersect.job_title.eq(intersect.job_title_prior)&intersect.description_text.eq(intersect.description_text_prior)).all():
            raise ValueError('Historical source content mismatch: '+str(path))
        fields=[c for c in history if c.startswith('ai_') or c.startswith('rejected_')]
        keep=history[['advertisement_id']+fields].rename(columns={c:prefix+c for c in fields})
        keep[prefix+'dataset_path']=str(path.resolve())
        frame=frame.merge(keep,on='advertisement_id',how='left',validate='one_to_one').fillna('')
    # Preserve duplicate-review annotations without deleting or merging any ads.
    duplicate_paths=sorted((ROOT/'data/processed').glob('step3c-*/jobs_duplicate_review.csv'))
    duplicate_path=duplicate_paths[-1] if duplicate_paths else None
    if duplicate_path:
        duplicates=read_csv(duplicate_path)
        fields=[c for c in duplicates if 'duplicate' in c and c!='advertisement_id']
        frame=frame.merge(duplicates[['advertisement_id']+fields],on='advertisement_id',how='left',validate='one_to_one').fillna('')
    frame['step3f_source_content_sha256']=[stable_hash(dict(job_title=r.job_title,description_text=r.description_text))
                                          for r in frame.itertuples()]
    frame.to_csv(output/'prepared_jobs.csv',index=False,encoding='utf-8-sig')
    (output/'policy.md').write_bytes(POLICY_PATH.read_bytes())
    (output/'prompt.txt').write_bytes(PROMPT_PATH.read_bytes())
    atomic_json(output/'schema.json',SCHEMA)
    manifest=dict(stage='step3f',total_advertisements=len(frame),source_path=str(source.resolve()),
        source_sha256=sha256(source),prepared_sha256=sha256(output/'prepared_jobs.csv'),
        pilot_path=str(pilot.resolve()),pilot_sha256=sha256(pilot),review_path=str(review.resolve()),
        review_sha256=sha256(review),policy_version=POLICY_VERSION,prompt_version=PROMPT_VERSION,
        validation_version=VALIDATION_VERSION,policy_sha256=sha256(POLICY_PATH),
        prompt_sha256=sha256(PROMPT_PATH),schema_sha256=stable_hash(SCHEMA),
        code_sha256={str(p.relative_to(ROOT)):sha256(p) for p in CODE_PATHS},
        duplicate_review_path=str(duplicate_path) if duplicate_path else None,
        old_predictions_preserved_not_reused=True,review_summary=analysis)
    atomic_json(output/'manifest.json',manifest)
    changed=[p for p,digest in protected.items() if not (ROOT/p).is_file() or sha256(ROOT/p)!=digest]
    if changed:raise RuntimeError('Preservation check failed: '+str(changed))
    atomic_json(output/'preservation_manifest.json',dict(files=protected,unchanged_files_verified=len(protected)))
    return manifest


def validate_prepared(prepared):
    prepared=Path(prepared)
    manifest=json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    checks={'stage':'step3f','policy_version':POLICY_VERSION,'prompt_version':PROMPT_VERSION,
        'validation_version':VALIDATION_VERSION,'policy_sha256':sha256(POLICY_PATH),
        'prompt_sha256':sha256(PROMPT_PATH),'schema_sha256':stable_hash(SCHEMA),
        'prepared_sha256':sha256(prepared/'prepared_jobs.csv'),
        'code_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in CODE_PATHS}}
    if any(manifest.get(k)!=v for k,v in checks.items()):
        raise ValueError('Prepared source/policy/prompt/schema/code compatibility check failed; prepare a new version')
    if sha256(prepared/'policy.md')!=manifest['policy_sha256'] or sha256(prepared/'prompt.txt')!=manifest['prompt_sha256']:
        raise ValueError('Prepared snapshot was modified')
    if stable_hash(json.loads((prepared/'schema.json').read_text(encoding='utf-8')))!=manifest['schema_sha256']:
        raise ValueError('Prepared schema was modified')
    frame=read_csv(prepared/'prepared_jobs.csv'); unique_ids(frame)
    if len(frame)!=manifest['total_advertisements']:raise ValueError('Prepared count mismatch')
    return manifest,frame


def write_dataset(output,rows):
    # Checkpoint remains authoritative if Windows prevents this derived CSV update.
    serial=pd.DataFrame(rows)
    for col in serial:
        if any(isinstance(v,(list,dict)) for v in serial[col]):
            serial[col]=serial[col].map(lambda v:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v)
    temporary=output/'classification_results.csv.tmp'
    serial.to_csv(temporary,index=False,encoding='utf-8-sig')
    temporary.replace(output/'classification_results.csv')


def run(prepared,output,config,budget_usd,max_ads=None,execute=False,resume=False,cache_dir=None,session=None):
    prepared,output=Path(prepared),Path(output)
    manifest,frame=validate_prepared(prepared)
    if max_ads is None:max_ads=len(frame)
    if type(max_ads) is not int or not 1<=max_ads<=len(frame):raise ValueError('Invalid maximum advertisement count')
    frame=frame.head(max_ads)
    cache_dir=Path(cache_dir or ROOT/'data/processed/.llm_cache')
    identity=dict(stage='step3f',prepared_manifest_sha256=sha256(prepared/'manifest.json'),
        config=asdict(config),max_ads=max_ads,cache_dir=str(cache_dir.resolve()))
    if output.exists() and not resume:raise FileExistsError('Existing output requires --resume; use a new live output')
    if resume:
        if not (output/'run_identity.json').exists():raise ValueError('Resume identity is missing')
        if json.loads((output/'run_identity.json').read_text(encoding='utf-8'))!=identity:
            raise ValueError('Run configuration/source compatibility mismatch')
    output.mkdir(parents=True,exist_ok=True); cache_dir.mkdir(parents=True,exist_ok=True)
    writable_probe(output); writable_probe(cache_dir)
    with file_lock(output/'run.lock'):
        if not resume:atomic_json(output/'run_identity.json',identity)
        client=Step3FClassifier(config,POLICY_PATH.read_text(encoding='utf-8'),cache_dir,
            output/'checkpoint.json',budget_usd,execute=execute,session=session)
        if (output/'checkpoint.json').exists() and client.state.get('identity')!=identity:
            raise ValueError('Checkpoint compatibility mismatch')
        client.state['identity']=identity
        if client.spent>budget_usd:raise BudgetExceeded('Existing accounted costs exceed application budget')
        sources={r['advertisement_id']:r for r in frame.to_dict('records')}
        if set(client.state['results'])-set(sources):raise ValueError('Checkpoint has unexpected IDs')
        plans=[]; unique_uncached={}
        for identifier,row in sources.items():
            title,description=row['job_title'],row['description_text']
            key=client.cache_key(title,description); prior=client.state['results'].get(identifier)
            if prior and prior.get('ai_cache_key')!=key:raise ValueError('Checkpoint source identity mismatch')
            if prior and prior.get('ai_status') in {'completed','cache_hit','missing_description'}:
                validate_span_prediction({k:prior[k] for k in AI_FIELDS},title,description)
                if prior.get('ai_human_verified') is not False:raise ValueError('AI output claims human verification')
            cached=client.cached(title,description)
            pending=(cache_dir/(key+'.pending.json')).exists()
            unknown=any(x['cache_key']==key and x['status'] in {'reserved_before_request','request_outcome_unknown'}
                        for x in client.state['ledger'])
            estimate=client.estimate(title,description)
            action=('checkpoint_preserved' if prior else 'cache_reuse' if cached else 'local_uncertain'
                if not description.strip() else 'blocked_unknown_outcome' if pending or unknown
                else 'input_limit_exceeded' if estimate['input_limit_exceeded'] else 'new_request')
            plans.append(dict(advertisement_id=identifier,cache_key=key,action=action,**estimate))
            if action=='new_request':unique_uncached[key]=estimate
        plan=pd.DataFrame(plans); plan.to_csv(output/'request_plan.csv',index=False,encoding='utf-8-sig')
        forecast=dict(total_advertisements=manifest['total_advertisements'],selected_advertisements=len(frame),
            action_counts=dict(Counter(plan.action)),cached_advertisements=int(plan.action.eq('cache_reuse').sum()),
            uncached_advertisements=int(plan.action.eq('new_request').sum()),
            unique_uncached_requests=len(unique_uncached),
            expected_input_tokens=sum(e['expected_input_tokens'] for e in unique_uncached.values()),
            expected_output_tokens=sum(e['expected_output_tokens'] for e in unique_uncached.values()),
            expected_cost_usd=sum(e['expected_cost_usd'] for e in unique_uncached.values()),
            conservative_cost_usd=sum(p['reserved_cost_per_attempt_usd'] for p in plans if p['action']=='new_request'),
            policy_version=POLICY_VERSION,prompt_version=PROMPT_VERSION,validation_version=VALIDATION_VERSION,
            model=config.model,input_usd_per_million=config.input_usd_per_million,
            output_usd_per_million=config.output_usd_per_million,budget_usd=budget_usd,
            accounted_cost_usd=client.spent,remaining_budget_usd=budget_usd-client.spent,
            pricing_source='https://developers.openai.com/api/docs/models/gpt-4.1-mini',
            estimates_not_actual_usage=True,retries=0,
            risks=['New span-selection prompt has not been live-tested; source grounding is not classification accuracy.',
                'Heuristic token estimates, no paid token-count calls; conservative reservations are not invoices.',
                'Unknown request outcomes block repayment and require explicit inspection.',
                'Old prompt/policy/schema results remain historical, not compatible cache hits.',
                'Human quality evaluation and policy approval remain necessary before publishing market statistics.'])
        atomic_json(output/'dry_run_report.json',forecast)
        if not execute:return forecast
        # All identities/caches/checkpoint and filesystem checks completed before this point.
        client.save(); rows=[]; stop_reason=None
        try:
            for row in sources.values():
                prediction=client.classify(row['advertisement_id'],row['job_title'],row['description_text'])
                rows.append(dict(row,**prediction))
                if len(rows)%25==0:write_dataset(output,rows)
                if prediction['ai_status'] in {'http_401','http_402','http_403','http_429','request_outcome_unknown'}:
                    stop_reason=prediction['ai_status'];break
        except BudgetExceeded as exc:
            stop_reason=str(exc)
        finally:
            # Include every prepared row; never mistake technical or not-yet-run for irrelevant.
            rows=[dict(row,**client.state['results'].get(identifier,dict(ai_status='not_executed',
                classification_status=None,ai_relevance=None,ai_primary_category=None,ai_human_verified=False)))
                for identifier,row in sources.items()]
            write_dataset(output,rows)
            atomic_json(output/'run_summary.json',dict(selected=len(rows),
                statuses=dict(Counter(r.get('classification_status') or 'Not executed' for r in rows)),
                ai_statuses=dict(Counter(r['ai_status'] for r in rows)),
                accounted_cost_usd=client.spent,**client.cost_breakdown(),stop_reason=stop_reason,
                human_verified=False,quality_metrics_calculated=False))
        return forecast


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--source',type=Path)
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--dry-run',action='store_true')
    parser.add_argument('--budget-usd',type=float,default=50)
    parser.add_argument('--max-ads',type=int)
    parser.add_argument('--model',default='gpt-4.1-mini-2025-04-14')
    parser.add_argument('--input-usd-per-million',type=float,default=.4)
    parser.add_argument('--output-usd-per-million',type=float,default=1.6)
    parser.add_argument('--provider',default='openai')
    parser.add_argument('--base-url',default='https://api.openai.com/v1')
    parser.add_argument('--api-key-env',default='OPENAI_API_KEY')
    parser.add_argument('--max-output-tokens',type=int,default=1024)
    parser.add_argument('--timeout',type=float,default=60)
    parser.add_argument('--cache-dir',type=Path)
    args=parser.parse_args()
    if args.prepare:
        if args.execute or args.resume:parser.error('Preparation never executes or resumes requests')
        result=prepare(args.source or latest_step3b()/'jobs_classified_v2.csv',args.output)
    else:
        if not args.prepared:parser.error('--prepared is required')
        config=Config(args.model,args.input_usd_per_million,args.output_usd_per_million,
            provider=args.provider,base_url=args.base_url,api_key_env=args.api_key_env,
            max_output_tokens=args.max_output_tokens,max_retries=0,timeout=args.timeout)
        result=run(args.prepared,args.output,config,args.budget_usd,args.max_ads,
            execute=args.execute,resume=args.resume,cache_dir=args.cache_dir)
    print(json.dumps(result,indent=2,ensure_ascii=True))


if __name__=='__main__':main()
