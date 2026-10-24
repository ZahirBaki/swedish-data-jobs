"""Offline, read-only technical analysis of the targeted Step 3F pilot."""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.recover_large_pilot import resilient_atomic_json as atomic_json
from src.run_step3f import validate_prepared
from src.step3f_classifier import AI_FIELDS, validate_span_prediction, span_match, Step3FClassifier, POLICY_PATH
from src.llm_classifier import Config
from src.validate_classification import sha256


def analyze(prepared,run_dir,output):
    prepared,run_dir,output=map(Path,[prepared,run_dir,output])
    if output.exists():raise FileExistsError('Use a new analysis output directory')
    manifest,frame=validate_prepared(prepared)
    if manifest.get('purpose')!='targeted_source_span_technical_pilot_not_evaluation':
        raise ValueError('Expected targeted pilot preparation, not a full dataset/evaluation set')
    identity=json.loads((run_dir/'run_identity.json').read_text(encoding='utf-8'))
    if identity.get('prepared_manifest_sha256')!=sha256(prepared/'manifest.json') or identity.get('max_ads')!=len(frame):
        raise ValueError('Pilot run does not match the full targeted sample')
    path=run_dir/'checkpoint.json'
    if not path.exists():raise ValueError('No executed checkpoint; a dry run has no model acceptance results')
    state=json.loads(path.read_text(encoding='utf-8'))
    if state.get('identity')!=identity:raise ValueError('Checkpoint/run identity mismatch')
    if set(state['results'])-set(frame.advertisement_id):raise ValueError('Checkpoint contains unexpected advertisements')
    client=Step3FClassifier(Config(**identity['config']),POLICY_PATH.read_text(encoding='utf-8'),
        Path(identity['cache_dir']),path,max(1,sum(x['accounted_cost_usd'] for x in state['ledger'])+1),execute=False)
    rows=[]
    for source in frame.to_dict('records'):
        identifier=source['advertisement_id'];prediction=state['results'].get(identifier,{})
        accepted=prediction.get('ai_status') in {'completed','cache_hit'}
        valid=False;error=''
        if prediction and prediction.get('ai_cache_key')!=client.cache_key(source['job_title'],source['description_text']):
            raise ValueError('Checkpoint content/request identity mismatch: '+identifier)
        if accepted:
            try:
                checked=validate_span_prediction({k:prediction[k] for k in AI_FIELDS},source['job_title'],source['description_text'])
                match=span_match(prediction['ai_evidence_span_ids'],source['job_title'],source['description_text'])
                if checked['ai_evidence']!=prediction.get('ai_evidence'):
                    raise ValueError('Saved evidence differs from selected source spans')
                if match['spans']!=prediction.get('ai_evidence_spans',[]):
                    raise ValueError('Saved evidence offsets/excerpts differ from original source')
                if prediction.get('ai_human_verified') is not False:
                    raise ValueError('AI output must not claim human verification')
                valid=True
            except (ValueError,TypeError,KeyError) as exc:error=str(exc)
        ledger=[x for x in state['ledger'] if x['advertisement_id']==identifier]
        diagnostic=[x.get('model_response_diagnostic',{}).get('validation_error','') for x in ledger]
        relevance=prediction.get('ai_relevance') if valid else None
        category=prediction.get('ai_primary_category') if valid else None
        relevance_disagreement=valid and relevance!=source['relevance']
        category_disagreement=valid and (category or '')!=source['primary_category']
        review_required=not valid or bool(prediction.get('ai_review_required'))
        priority=(4*int(not valid)+3*int(relevance_disagreement)+2*int(category_disagreement)
                  +int(prediction.get('ai_confidence') in {'low','medium'})+int(review_required))
        rows.append(dict(source,**prediction,technical_accepted=accepted,source_span_valid=valid,
            independent_source_span_error=error,validation_errors=json.dumps([x for x in diagnostic if x]),
            checked_ai_relevance=relevance,checked_ai_primary_category=category,
            rule_ai_relevance_disagreement=relevance_disagreement,rule_ai_category_disagreement=category_disagreement,
            human_inspection_required=review_required,review_priority=priority,
            human_relevance='',human_primary_category='',human_review_notes='',human_verified=''))
    valid=[r for r in rows if r['source_span_valid']]
    usage=sum(x['accounted_cost_usd'] for x in state['ledger'] if x['usage_basis']=='provider_reported_tokens')
    holds=sum(x['accounted_cost_usd'] for x in state['ledger'] if x['usage_basis']!='provider_reported_tokens')
    summary=dict(total=len(rows),saved_result_entries=len(state['results']),api_attempts=len(state['ledger']),
        technically_accepted_responses=sum(r['technical_accepted'] for r in rows),
        accepted_and_source_span_valid=len(valid),technical_coverage_fraction=len(valid)/len(rows),
        technical_acceptance_per_attempt=(sum(x['status'].startswith('completed') for x in state['ledger'])/len(state['ledger'])
                                          if state['ledger'] else None),
        not_executed=sum(not state['results'].get(r['advertisement_id']) for r in rows),
        failed_or_invalid_saved_results=sum(bool(state['results'].get(r['advertisement_id'])) and not r['source_span_valid'] for r in rows),
        relevance_counts=dict(Counter(r['checked_ai_relevance'] for r in valid)),
        category_counts=dict(Counter(r['checked_ai_primary_category'] or '(unassigned)' for r in valid)),
        rule_ai_relevance_disagreements=sum(r['rule_ai_relevance_disagreement'] for r in rows),
        rule_ai_category_disagreements=sum(r['rule_ai_category_disagreement'] for r in rows),
        review_flags=sum(r['human_inspection_required'] for r in rows),
        low_or_medium_confidence=sum(r.get('ai_confidence') in {'low','medium'} for r in valid),
        previously_rule_uncertain_resolved=sum(r['relevance']=='uncertain' and r['checked_ai_relevance'] in {'relevant','irrelevant'} for r in valid),
        provider_reported_usage_estimated_cost_usd=usage,unreported_usage_reserved_usd=holds,
        accounted_cost_usd=usage+holds,checkpoint_sha256=sha256(path),
        classification_accuracy_calculated=False,human_verified_labels_used=0,
        purpose='targeted technical pilot; not an independent quality evaluation',
        evidence_support_warning='Exact source spans do not establish logical support; human inspection required.')
    output.mkdir(parents=True)
    export=pd.DataFrame(rows).sort_values(['review_priority','advertisement_id'],ascending=[False,True])
    for c in export:
        export[c]=export[c].map(lambda v:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v)
    export.to_csv(output/'pilot_technical_review.csv',index=False,encoding='utf-8-sig')
    atomic_json(output/'technical_summary.json',summary)
    (output/'report.md').write_text('# Step 3F technical pilot analysis\n\n'
        'This targeted sample is not an independent classification-quality evaluation.\n\n'
        '```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n\n'
        'Inspect full sources, evidence support, special scopes/opportunity flags and disagreements '
        'in `pilot_technical_review.csv`. Do not treat rules or assistant labels as truth.\n',encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(analyze(args.prepared,args.run,args.output),indent=2,ensure_ascii=True))


if __name__=='__main__':main()
