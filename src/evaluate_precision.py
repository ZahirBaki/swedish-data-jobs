"""Offline independent-label precision evaluation; no AI suggestions as truth."""
import argparse
import json
from collections import Counter
from dataclasses import replace,asdict
from pathlib import Path

import pandas as pd

from src.classify_jobs_v2 import CATEGORIES
from src.precision_inclusion import InclusionConfig,inclusion_decision
from src.step3f_reliable_classifier import source_hash
from src.validate_classification import read_csv,sha256
from src.recover_large_pilot import resilient_atomic_json as atomic_json


def validate_human_labels(labels,reserve):
    if labels.advertisement_id.duplicated().any() or set(labels.advertisement_id)!=set(reserve['reserved_ids']):
        raise ValueError('Human sheet must match the independent reserve IDs exactly')
    if {source_hash(r.job_title,r.description_text) for r in labels.itertuples()}!=set(reserve['content_keys']):
        raise ValueError('Human sheet source text differs from independent reserve')
    if 'source_by_id' not in reserve:raise ValueError('Use the prepared evaluation manifest with per-ID source hashes')
    if any(source_hash(r.job_title,r.description_text)!=reserve['source_by_id'].get(r.advertisement_id) for r in labels.itertuples()):
        raise ValueError('Human sheet source/ID association changed')
    if any(c.startswith(('assistant_','ai_','suggested_','provisional_')) for c in labels):
        raise ValueError('AI-assisted review files are not independent human ground truth')
    if set(labels.human_verified.str.strip().str.lower())-{'','false','true'}:raise ValueError('Invalid human verification flag')
    verified=labels[labels.human_verified.str.strip().str.lower().eq('true')]
    if verified.reviewer.str.strip().eq('').any():raise ValueError('Verified labels require a named human reviewer')
    if set(verified.human_relevance)-{'relevant','irrelevant','uncertain'}:raise ValueError('Invalid verified relevance')
    if set(verified.human_primary_category)-set(CATEGORIES)-{''}:raise ValueError('Invalid verified human category')
    if (verified.human_relevance.ne('relevant')&verified.human_primary_category.ne('')).any():raise ValueError('Non-relevant human categories must be blank')
    return verified


def evaluate(predictions,labels,reserve,config=InclusionConfig()):
    verified=validate_human_labels(labels,reserve)
    if len(predictions) and (predictions.advertisement_id.duplicated().any() or set(predictions.advertisement_id)-set(labels.advertisement_id)):
        raise ValueError('Unexpected or duplicate prediction IDs')
    if any(c.startswith('human_') for c in predictions):raise ValueError('Freeze predictions separately from human evaluation labels')
    by_id={r['advertisement_id']:r for r in predictions.to_dict('records')}
    truth={r['advertisement_id']:r for r in verified.to_dict('records')}
    rows=[]
    for original in labels.to_dict('records'):
        identifier=original['advertisement_id'];p=by_id.get(identifier,{})
        if p and (p.get('job_title')!=original['job_title'] or p.get('description_text')!=original['description_text']):
            raise ValueError('Prediction source differs from independent human source')
        decision=inclusion_decision(p,config)
        t=truth.get(identifier)
        if t and decision['inclusion_status']=='included' and p.get('duties_assessor','').strip().casefold()==t['reviewer'].strip().casefold():
            raise ValueError('Inclusion auditor and independent ground-truth reviewer must differ')
        resolved=bool(t and t['human_relevance'] in {'relevant','irrelevant'})
        included=decision['inclusion_status']=='included'
        relevance_ok=(t['human_relevance']=='relevant') if included and resolved else None
        cat_evaluable=bool(included and resolved and (t['human_relevance']=='irrelevant' or t['human_primary_category']))
        category_ok=(t['human_relevance']=='relevant' and p.get('ai_primary_category')==t['human_primary_category']) if cat_evaluable else None
        false_positive=included and resolved and not relevance_ok
        wrong_category=cat_evaluable and not category_ok
        rows.append(dict(advertisement_id=identifier,job_title=original['job_title'],description_text=original['description_text'],
            **decision,ai_relevance=p.get('ai_relevance'),ai_primary_category=p.get('ai_primary_category'),
            ai_status=p.get('ai_status') or 'not_executed',
            ai_scope=p.get('ai_scope') or 'unavailable',ai_reasoning=p.get('ai_reasoning'),ai_evidence=p.get('ai_evidence'),
            ai_confidence=p.get('ai_confidence'),duties_assessment=p.get('duties_assessment'),
            human_relevance=t.get('human_relevance') if t else None,human_primary_category=t.get('human_primary_category') if t else None,
            human_role_scope=t.get('human_role_scope') if t else None,
            relevance_evaluable=included and resolved,relevance_correct=relevance_ok,
            category_evaluable=cat_evaluable,category_correct=category_ok,
            relevance_false_positive=false_positive,category_false_positive=wrong_category,
            human_false_positive_cause=t.get('human_false_positive_cause','') if t else '',
            observed_error='included_unrelated_occupation' if false_positive else 'wrong_primary_category' if wrong_category else '',
            cause_status='human_supplied' if t and t.get('human_false_positive_cause') else 'requires_human_cause_review' if false_positive or wrong_category else ''))
    def measures(subset):
        rel=[r for r in subset if r['relevance_evaluable']];cat=[r for r in subset if r['category_evaluable']]
        return dict(total=len(subset),included=sum(r['inclusion_status']=='included' for r in subset),
            relevance_precision=sum(r['relevance_correct'] for r in rel)/len(rel) if rel else None,relevance_precision_denominator=len(rel),
            category_precision=sum(r['category_correct'] for r in cat)/len(cat) if cat else None,category_precision_denominator=len(cat),
            inclusion_coverage=sum(r['inclusion_status']=='included' for r in subset)/len(subset) if subset else None,
            abstention_rate=sum(r['inclusion_status']=='abstained' for r in subset)/len(subset) if subset else None,
            technical_failure_rate=sum(r['inclusion_status']=='technical_failure' for r in subset)/len(subset) if subset else None,
            not_executed_rate=sum(r['inclusion_status']=='not_executed' for r in subset)/len(subset) if subset else None,
            out_of_scope_rate=sum(r['inclusion_status']=='out_of_scope' for r in subset)/len(subset) if subset else None)
    summary=measures(rows)
    summary.update(verified_human_labels=len(verified),unreviewed=len(labels)-len(verified),
        unresolved_human_relevance=int(verified.human_relevance.eq('uncertain').sum()),
        unresolved_human_category=int((verified.human_relevance.eq('relevant')&verified.human_primary_category.eq('')).sum()),
        included_without_resolved_truth=sum(r['inclusion_status']=='included' and not r['relevance_evaluable'] for r in rows),
        statuses=dict(Counter(r['inclusion_status'] for r in rows)),configuration=asdict(config),
        status='verified labels available' if len(verified) else 'awaiting independently verified human labels',
        confidence_is_calibrated=False,scope='Reserved distinct-content evaluation frame only; no full-market inference')
    errors=[r for r in rows if r['relevance_false_positive'] or r['category_false_positive']]
    summary['false_positive_observations']=dict(Counter(r['observed_error'] for r in errors))
    summary['false_positive_cause_counts']=dict(Counter(r['human_false_positive_cause'] or 'requires_human_cause_review' for r in errors))
    scopes={scope:measures([r for r in rows if r['ai_scope']==scope]) for scope in ['core_industry','data_ai_governance','academic_ai_ml_research','adjacent_borderline','unrelated','unresolved','unavailable']}
    categories={category:measures([r for r in rows if r['ai_primary_category']==category]) for category in CATEGORIES}
    return dict(summary=summary,by_predicted_scope=scopes,by_predicted_category=categories,rows=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--reserve-manifest',type=Path,required=True)
    parser.add_argument('--predictions',type=Path)
    parser.add_argument('--inclusion-audit',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--confidence-levels',default='high')
    parser.add_argument('--scopes',default='core_industry,data_ai_governance')
    parser.add_argument('--allow-review',action='store_true')
    parser.add_argument('--allow-unaudited',action='store_true')
    parser.add_argument('--allow-opportunities',action='store_true')
    parser.add_argument('--allow-other-category',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a new evaluation output directory')
    if any(x in args.labels.name.lower() for x in ['ai_assisted','assistant','suggestions']):raise ValueError('AI-assisted reference is not ground truth')
    labels=read_csv(args.labels);reserve=json.loads(args.reserve_manifest.read_text(encoding='utf-8'))
    predictions=read_csv(args.predictions) if args.predictions else pd.DataFrame()
    if args.inclusion_audit:
        audit=read_csv(args.inclusion_audit)
        if audit.advertisement_id.duplicated().any() or set(audit.advertisement_id)-set(labels.advertisement_id):raise ValueError('Unexpected inclusion audit IDs')
        fields=['duties_assessment','category_assessment','duties_assessor','inclusion_audit_notes','inclusion_review_resolved']
        if any(c in predictions for c in fields):raise ValueError('Ambiguous duplicate inclusion audit fields')
        if len(predictions):predictions=predictions.merge(audit[['advertisement_id']+fields],on='advertisement_id',how='left',validate='one_to_one').fillna('')
    config=InclusionConfig(tuple(args.confidence_levels.split(',')),tuple(args.scopes.split(',')),
        not args.allow_review,not args.allow_unaudited,args.allow_other_category,args.allow_opportunities)
    result=evaluate(predictions,labels,reserve,config)
    thresholds={}
    for name,levels in [('high',['high']),('high_medium',['high','medium']),('all',['high','medium','low'])]:
        thresholds[name]=evaluate(predictions,labels,reserve,replace(config,confidence_levels=tuple(levels)))['summary']
    args.output.mkdir(parents=True)
    atomic_json(args.output/'evaluation_summary.json',{k:v for k,v in result.items() if k!='rows'})
    atomic_json(args.output/'threshold_comparison.json',dict(results=thresholds,
        caution='Diagnostic comparison only. Choosing thresholds here consumes this holdout; use a separate calibration set for tuning.'))
    rows=pd.DataFrame(result['rows'])
    for c in rows:
        rows[c]=rows[c].map(lambda x:json.dumps(x,ensure_ascii=False) if isinstance(x,(dict,list)) else x)
    rows.to_csv(args.output/'inclusion_decisions.csv',index=False,encoding='utf-8-sig')
    rows.loc[rows.relevance_false_positive|rows.category_false_positive].to_csv(args.output/'false_positives.csv',index=False,encoding='utf-8-sig')
    atomic_json(args.output/'evaluation_provenance.json',dict(labels_sha256=sha256(args.labels),
        predictions_sha256=sha256(args.predictions) if args.predictions else None,
        inclusion_audit_sha256=sha256(args.inclusion_audit) if args.inclusion_audit else None,
        reserve_manifest_sha256=sha256(args.reserve_manifest),human_labels_auto_populated=0))
    print(json.dumps(result['summary'],ensure_ascii=True,indent=2))


if __name__=='__main__':main()
