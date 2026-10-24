"""Combine accepted pilot/retry predictions without API requests or parent writes."""
import argparse
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from src.llm_classifier import AI_FIELDS, SCHEMA, VALIDATION_VERSION, atomic_json, stable_hash, validate_prediction
from src.run_ai_classification import compare_rows, write_results
from src.validate_classification import read_csv, sha256

ACCEPTED = {'completed','cache_hit'}


def load_run(path):
    path = Path(path).resolve()
    plan = json.loads((path/'plan.json').read_text(encoding='utf-8'))
    checkpoint = json.loads((path/'checkpoint.json').read_text(encoding='utf-8'))
    if sha256(path/'pilot_input.csv') != plan['pilot_input_sha256']:
        raise ValueError('Frozen input changed')
    if plan['identity']['schema_sha256'] != stable_hash(SCHEMA):
        raise ValueError('Schema migration requires explicit review')
    inputs = read_csv(path/'pilot_input.csv').set_index('advertisement_id')
    rows = [json.loads(line) for line in (path/'pilot_results.jsonl').read_text(encoding='utf-8').splitlines()]
    ids = [r['advertisement_id'] for r in rows]
    if not inputs.index.is_unique or len(set(ids)) != len(ids) or set(ids) != set(inputs.index):
        raise ValueError('Results must cover unique frozen IDs exactly once')
    for row in rows:
        identifier = row['advertisement_id']
        for name in ['job_title','description_text','relevance','primary_category']:
            if row[name] != inputs.loc[identifier,name]:
                raise ValueError('Original fields changed')
        saved = checkpoint['results'].get(identifier,{})
        if any(saved.get(k) != row.get(k) for k in AI_FIELDS+[
                'ai_status','ai_cache_key','ai_prompt_version','ai_model','ai_provider','ai_policy_sha256']):
            raise ValueError('Checkpoint/results mismatch')
        if row.get('ai_human_verified') is not False:
            raise ValueError('AI output must not claim human verification')
        if row.get('ai_prompt_version') != plan['identity']['prompt_version']:
            raise ValueError('Row prompt provenance differs from run')
        if row.get('ai_model') != plan['identity']['config']['model']:
            raise ValueError('Row model provenance differs from run')
        if row['ai_status'] in ACCEPTED:
            validate_prediction({k:row[k] for k in AI_FIELDS},row['job_title'],row['description_text'],
                allow_formatting=plan['identity'].get('validation_version')==VALIDATION_VERSION)
    return path, plan, checkpoint, rows


def combine(original, retry, output, expected_count=30, additional_retries=None):
    output = Path(output)
    if output.exists():
        raise ValueError('Combined output must be a new folder')
    loaded=[load_run(path) for path in [original,retry,*(additional_retries or [])]]
    parent,pplan,pcheckpoint,prows=loaded[0]
    if len({item[0] for item in loaded})!=len(loaded) or len(prows) != expected_count:
        raise ValueError('Distinct runs and expected pilot size required')
    selections={r['advertisement_id']:(r,parent,pplan,pcheckpoint) for r in prows}
    for child,cplan,ccheckpoint,crows in loaded[1:]:
        if pplan['identity']['policy_sha256'] != cplan['identity']['policy_sha256']:
            raise ValueError('Policy differs; review before combining')
        failures={identifier for identifier,(r,*_) in selections.items() if r['ai_status'] not in ACCEPTED}
        retry_by_id={r['advertisement_id']:r for r in crows}
        if set(retry_by_id)!=failures:
            raise ValueError('Retry must contain exactly the original failed IDs')
        for identifier,row in retry_by_id.items():
            previous=selections[identifier][0]
            for name in ['job_title','description_text','relevance','primary_category']:
                if previous[name]!=row[name]:
                    raise ValueError('Retry changes original advertisement/rule fields')
            selections[identifier]=(row,child,cplan,ccheckpoint)
    combined = []
    for base in prows:
        selected,run,plan,checkpoint=selections[base['advertisement_id']]
        for name in ['job_title','description_text','relevance','primary_category']:
            if base[name] != selected[name]:
                raise ValueError('Retry changes original advertisement/rule fields')
        row = {**base,**{k:v for k,v in selected.items() if k.startswith('ai_')}}
        accepted = row['ai_status'] in ACCEPTED
        attempts = [e for e in checkpoint['ledger'] if e['advertisement_id']==row['advertisement_id']]
        diagnostic = attempts[-1].get('model_response_diagnostic',{}) if attempts else {}
        row.update(ai_source_run=str(run),ai_schema_sha256=plan['identity']['schema_sha256'],
                   ai_validation_version=plan['identity'].get('validation_version','exact-legacy'),
                   ai_validation_status='accepted' if accepted else 'unresolved',
                   ai_failure_type=None if accepted else row['ai_status'],
                   ai_validation_error=None if accepted else diagnostic.get('validation_error'),
                   ai_validation_stage=None if accepted else diagnostic.get('validation_stage'))
        if not accepted:
            for field in AI_FIELDS:row[field]=None
            row.update(ai_review_required=True,ai_human_verified=False,ai_role_scope='not_classified',
                       ai_model_returned=diagnostic.get('model'))
        combined.append(row)
    output.mkdir(parents=True,exist_ok=False)
    write_results(output,combined)
    # Retain explicit nulls in JSONL; CSV missing values are blank, never irrelevant.
    summary = dict(total_advertisements=len(combined),unique_ids=len({r['advertisement_id'] for r in combined}),
        accepted=sum(r['ai_validation_status']=='accepted' for r in combined),
        unresolved=sum(r['ai_validation_status']=='unresolved' for r in combined),
        relevance_counts=dict(Counter(r['ai_relevance'] if r['ai_validation_status']=='accepted' else 'unresolved' for r in combined)),
        category_counts=dict(Counter(r['ai_primary_category'] or '(null)' for r in combined if r['ai_validation_status']=='accepted')),
        accepted_prompt_versions=dict(Counter(r['ai_prompt_version'] for r in combined if r['ai_validation_status']=='accepted')),
        comparison=compare_rows(combined),source_runs=[str(item[0]) for item in loaded],
        mixed_prompt_versions=len({r['ai_prompt_version'] for r in combined})>1,new_paid_requests=0,
        source_usage_estimated_cost_usd=float(sum((Decimal(str(e['accounted_cost_usd'])) for _,_,cp,_ in loaded for e in cp['ledger'] if e['usage_basis']=='provider_reported_tokens'),Decimal(0))),
        source_unreported_usage_reserved_usd=float(sum((Decimal(str(e['accounted_cost_usd'])) for _,_,cp,_ in loaded for e in cp['ledger'] if e['usage_basis']!='provider_reported_tokens'),Decimal(0))),
        provenance={str(run):{name:sha256(run/name) for name in ['plan.json','checkpoint.json','pilot_input.csv','pilot_results.jsonl']}
                    for run,_,_,_ in loaded},
        evaluation_scope='Mixed-version pilot; no human accuracy claimed. Failed rows are operationally unresolved, not predicted irrelevant.')
    atomic_json(output/'combined_summary.json',summary)
    return combined,summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',required=True,type=Path)
    parser.add_argument('--retry',required=True,type=Path)
    parser.add_argument('--additional-retry',action='append',type=Path,default=[],help='Additional failed-only runs, in chronological order')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    _,summary=combine(args.original,args.retry,args.output,additional_retries=args.additional_retry)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
