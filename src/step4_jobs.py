"""Read existing AI pilot decisions; no collection, reclassification or API calls."""
import json
from pathlib import Path

import pandas as pd

from src.llm_classifier import AI_FIELDS as LEGACY_FIELDS,validate_prediction
from src.step3f_classifier import AI_FIELDS,validate_span_prediction,span_match
from src.step3f_reliable_classifier import validate_reliable_prediction
from src.validate_classification import ROOT,read_csv,sha256

SOURCE=ROOT/'data/processed/step3b-20261022T083354891374Z/jobs_classified_v2.csv'


def validate_classification(prediction,source):
    if prediction.get('ai_status') not in {'completed','cache_hit'}:return False,'not_accepted'
    try:
        version=prediction.get('ai_prompt_version','')
        if version in {'step3f-1.0','step3f-1.1'}:
            validator=validate_reliable_prediction if version=='step3f-1.1' else validate_span_prediction
            validator({k:prediction[k] for k in AI_FIELDS},source['job_title'],source['description_text'])
            spans=span_match(prediction['ai_evidence_span_ids'],source['job_title'],source['description_text'])['spans']
            if prediction.get('ai_evidence')!=[s['text'] for s in spans] or prediction.get('ai_evidence_spans',[])!=spans:
                raise ValueError('Saved source offsets/excerpts differ')
        elif version.startswith('step3d-') or version=='step3e-1.0':
            validate_prediction({k:prediction[k] for k in LEGACY_FIELDS},source['job_title'],source['description_text'],allow_formatting=True)
        else:raise ValueError('Unknown classification prompt version')
        return True,'source_validated'
    except (ValueError,KeyError,TypeError) as error:return False,str(error)


def reporting_scope(category):
    if category=='AI / ML Research':return 'research'
    if category=='Data Governance / Data Quality':return 'governance'
    if category in {'Statistician','Data / AI Management','Other relevant Data & AI roles'}:return 'adjacent'
    return 'core_industry' if category else 'unresolved'


def load_existing(source_path=SOURCE):
    source=read_csv(source_path)
    if source.advertisement_id.duplicated().any():raise ValueError('Source IDs must be unique')
    by_id=source.set_index('advertisement_id',drop=False).to_dict('index')
    sources=[];records=[]
    # Canonical completed pilot outputs, oldest to newest. Avoid replaying archived
    # checkpoints/dry runs as independent model results.
    combined=ROOT/'data/processed/step3d-combined-v2-20261022/dataset/pilot_results.jsonl'
    if combined.exists():
        rows=[json.loads(line) for line in combined.read_text(encoding='utf-8').splitlines()]
        sources.append(dict(path=str(combined),sha256=sha256(combined)))
        records.extend((combined.parent,row,{k:v for k,v in row.items() if k.startswith('ai_')}) for row in rows)
    definitions=[
        ('step3e-live-recovered-20261022','pilot_input.csv'),
        ('step3f-span-pilot20-recovered-20261022',None),
        ('precision-reliability-live-20261023',None),
    ]
    for name,input_name in definitions:
        run=ROOT/'data/processed'/name;checkpoint=run/'checkpoint.json'
        if not checkpoint.exists():continue
        state=json.loads(checkpoint.read_text(encoding='utf-8'))
        sources.append(dict(path=str(checkpoint),sha256=sha256(checkpoint)))
        if input_name:
            inputs=read_csv(run/input_name).set_index('advertisement_id',drop=False).to_dict('index')
        elif name.startswith('step3f'):
            inputs=read_csv(ROOT/'data/processed/step3f-span-pilot20-20261022/prepared_jobs.csv').set_index('advertisement_id',drop=False).to_dict('index')
        else:
            inputs=read_csv(ROOT/'data/processed/precision-first-prepared-20261023/reliability_cases.csv').set_index('advertisement_id',drop=False).to_dict('index')
        for identifier,prediction in state['results'].items():
            records.append((run,inputs[identifier],prediction))
    history=[];latest={};latest_accepted={}
    for order,(run,original,prediction) in enumerate(records):
        identifier=str(original['advertisement_id'])
        if identifier not in by_id:raise ValueError('Pilot ID missing from collected source')
        canonical=by_id[identifier]
        if any(original[k]!=canonical[k] for k in ['job_title','description_text']):
            raise ValueError('Original title/description changed across pilot/source')
        valid,error=validate_classification(prediction,canonical)
        item=dict(advertisement_id=identifier,source_run=str(run),history_order=order,
            classification_source_valid=valid,validation_result=error,prediction=prediction)
        history.append(item);latest[identifier]=item
        # A failed paid retry cannot erase an earlier valid semantic judgment.
        # A later accepted uncertain/irrelevant judgment does supersede relevance.
        if valid:latest_accepted[identifier]=item
    registry=[]
    for original in source.to_dict('records'):
        identifier=original['advertisement_id'];chosen=latest_accepted.get(identifier) or latest.get(identifier)
        row=dict(original)
        if chosen:
            row.update(chosen['prediction'])
            row.update(classification_selected_run=chosen['source_run'],
                classification_source_valid=chosen['classification_source_valid'])
            if chosen['classification_source_valid']:
                disposition={'relevant':'eligible_relevant','irrelevant':'excluded_irrelevant','uncertain':'set_aside_uncertain'}[row['ai_relevance']]
            else:disposition='set_aside_technical_failure'
        else:
            row.update(ai_status='not_classified',classification_source_valid=False)
            disposition='not_classified'
        row['step4_disposition']=disposition
        row['reporting_scope']=reporting_scope(row.get('ai_primary_category'))
        row['classification_human_verified']=False
        registry.append(row)
    result=pd.DataFrame(registry).astype(object)
    result=result.where(result.notna(),None)
    return result,history,sources


def select_pilot(registry,size=50,seed=20261023):
    eligible=registry[registry.step4_disposition.eq('eligible_relevant')].copy()
    eligible=eligible.sort_values('advertisement_id')
    n=min(size,len(eligible))
    if not n:return eligible
    # Preserve representation of available categories. This is a development
    # sample of previously selected pilot ads, never a market probability sample.
    groups=list(eligible.groupby('ai_primary_category',sort=True,dropna=False))
    if n<len(groups):return eligible.sample(n=n,random_state=seed).sort_values('advertisement_id')
    quotas={str(k):1 for k,_ in groups}
    while sum(quotas.values())<n:
        key,group=max(((str(k),g) for k,g in groups if quotas[str(k)]<len(g)),
            key=lambda pair:(n*len(pair[1])/len(eligible)-quotas[pair[0]],pair[0]))
        quotas[key]+=1
    parts=[group.sample(n=quotas[str(key)],random_state=seed+i) for i,(key,group) in enumerate(groups)]
    return pd.concat(parts).sort_values('advertisement_id').reset_index(drop=True)
