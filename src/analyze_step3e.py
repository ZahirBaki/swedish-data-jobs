"""Analyze saved Step 3E results only. Never requests new model classifications."""
import argparse
import json
import re
import unicodedata
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pandas as pd

from src.llm_classifier import AI_FIELDS,ClassificationError,evidence_match
from src.recover_large_pilot import inspect
from src.run_ai_classification import compare_rows,write_results
from src.validate_classification import sha256

ACCEPTED={'completed','cache_hit'}
# Individually inspected evidence fragments skip/join separated source passages.
INSPECTED_DISCONTIGUOUS={'31454174','31430335','31554105','31159123','30944938'}


def normalize_space(text):
    return re.sub(r'\s+',' ',text)


def evidence_root_cause(identifier,quote,title,description):
    """Inspection aid only: case/punctuation comparisons NEVER accept a response."""
    try:
        return evidence_match(quote,title,description,True)['method']
    except ClassificationError:pass
    if any(normalize_space(quote).casefold() in normalize_space(s).casefold() for s in [title,description]):
        return 'capitalization_only'
    def punctuation_view(s):
        return normalize_space(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in s)).strip()
    if any(punctuation_view(quote) in punctuation_view(s) for s in [title,description]):
        return 'punctuation_only'
    if '...' in quote or '…' in quote or identifier in INSPECTED_DISCONTIGUOUS:
        return 'omitted_or_joined_passages'
    return 'wording_spelling_or_grammar'


def analyze(run_dir,output,expected_count=200,cache_dir=None):
    run_dir,output=Path(run_dir).resolve(),Path(output)
    if output.exists():raise ValueError('Use a new analysis output directory')
    saved_summary=json.loads((run_dir/'run_summary.json').read_text(encoding='utf-8'))
    audit,frame,checkpoint,plan=inspect(run_dir,saved_summary.get('budget_usd',2),cache_dir)
    if len(frame)!=expected_count or len(checkpoint['results'])!=expected_count:
        raise ValueError('Expected complete result coverage, including failures')
    if set(checkpoint['results'])!=set(frame.advertisement_id):raise ValueError('Result IDs differ from sample')
    exported=[json.loads(line) for line in (run_dir/'pilot_results.jsonl').read_text(encoding='utf-8').splitlines()]
    export_by_id={row['advertisement_id']:row for row in exported}
    if len(export_by_id)!=len(exported) or set(export_by_id)!=set(frame.advertisement_id):
        raise ValueError('Export must cover unique original IDs')
    attempts_by_id={identifier:[e for e in checkpoint['ledger'] if e['advertisement_id']==identifier]
                    for identifier in frame.advertisement_id}
    combined=[];failures=[]
    for original in frame.to_dict('records'):
        identifier=original['advertisement_id'];prediction=checkpoint['results'][identifier]
        export=export_by_id[identifier]
        if any(export.get(k)!=prediction.get(k) for k in AI_FIELDS+['ai_status']):
            raise ValueError('Checkpoint and exported prediction disagree')
        if any(export[k]!=original[k] for k in ['job_title','description_text','relevance','primary_category']):
            raise ValueError('Original fields changed in export')
        row={**export,**prediction}
        row.update(ai_source_run=str(run_dir),ai_schema_sha256=plan['identity']['schema_sha256'],
            ai_checkpoint_sha256=sha256(run_dir/'checkpoint.json'),
            ai_validation_status='accepted' if prediction['ai_status'] in ACCEPTED else 'unresolved',
            api_attempts_for_id=len(attempts_by_id[identifier]),
            usage_estimated_cost_for_id=float(sum((Decimal(str(e['accounted_cost_usd']))
                for e in attempts_by_id[identifier]),Decimal(0))))
        if prediction['ai_status'] not in ACCEPTED:
            entries=attempts_by_id[identifier]
            if not entries:raise ValueError('Rejected response must have a corresponding attempt')
            entry=entries[-1];diagnostic=entry.get('model_response_diagnostic',{})
            proposed=json.loads(diagnostic['content'])
            try:independent_evidence=evidence_match(proposed['ai_evidence'],row['job_title'],row['description_text'],True)['method']
            except ClassificationError:independent_evidence='unmatched'
            error=diagnostic.get('validation_error')
            if error=='Evidence is not a verbatim excerpt of title/description':
                root=evidence_root_cause(identifier,proposed['ai_evidence'],row['job_title'],row['description_text'])
            elif error=='Non-relevant decisions must abstain from categories':
                root='uncertain_with_secondary_categories' if proposed['ai_relevance']=='uncertain' else 'nonrelevant_with_categories'
            else:root='other_validation_error'
            row.update(ai_failure_type=prediction['ai_status'],ai_validation_error=error,
                ai_validation_stage=diagnostic.get('validation_stage'),failure_root_cause=root,
                independent_evidence_check=independent_evidence)
            for field in AI_FIELDS:
                row['rejected_'+field]=proposed.get(field)
                row[field]=None
            row['ai_review_required']=True
            failures.append(dict(advertisement_id=identifier,job_title=row['job_title'],
                http_status=entry.get('http_status'),validation_error=error,
                validation_error_type=diagnostic.get('validation_error_type'),validation_stage=diagnostic.get('validation_stage'),
                root_cause=root,independent_evidence_check=independent_evidence,
                rejected_proposal_only=True,**{field:proposed[field] for field in AI_FIELDS},
                recorded_attempt_cost_usd=entry['accounted_cost_usd']))
        row['ai_human_verified']=False
        combined.append(row)
    accepted=[r for r in combined if r['ai_validation_status']=='accepted']
    review=[]
    for row in combined:
        valid=row['ai_validation_status']=='accepted'
        relevance_diff=valid and row['relevance']!=row['ai_relevance']
        category_diff=valid and (row['primary_category'] or '')!=(row['ai_primary_category'] or '')
        low=valid and row['ai_confidence']=='low'
        review_flag=bool(row.get('ai_review_required'))
        priority=1 if not valid or low or (valid and row['ai_relevance']=='uncertain') else 2 if relevance_diff or category_diff or review_flag else 3
        reasons=[label for label,flag in [('rejected_response',not valid),('low_confidence',low),
            ('model_uncertain',valid and row['ai_relevance']=='uncertain'),('relevance_disagreement',relevance_diff),
            ('category_disagreement',category_diff),('model_review_required',review_flag)] if flag]
        reviewed={**row,'review_priority':priority,'quality_review_reasons':'; '.join(reasons) or 'agreement_control',
                  'relevance_disagreement':relevance_diff,'category_disagreement':category_diff}
        for name in ['human_relevance','human_primary_category','human_review_notes','human_verified','reviewer']:
            reviewed[name]=''
        review.append(reviewed)
    output.mkdir(parents=True,exist_ok=False)
    write_results(output,combined)
    pd.DataFrame(failures).to_csv(output/'rejected_responses.csv',index=False,encoding='utf-8-sig')
    review_frame=pd.DataFrame(review).sort_values(['review_priority','relevance_disagreement','category_disagreement','advertisement_id'],
        ascending=[True,False,False,True],kind='stable')
    review_frame.insert(0,'review_order',range(1,len(review_frame)+1))
    review_frame.to_csv(output/'prioritized_human_review.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(checkpoint['ledger']).to_csv(output/'api_usage_records.csv',index=False,encoding='utf-8-sig')
    strata=[]
    for group in sorted(frame.sampling_group.unique()):
        selected=[r for r in combined if r['sampling_group']==group]
        valid=[r for r in selected if r['ai_validation_status']=='accepted']
        strata.append(dict(group=group,selected=len(selected),accepted=len(valid),unresolved=len(selected)-len(valid),
            technical_acceptance_rate=len(valid)/len(selected),relevance_counts=dict(Counter(r['ai_relevance'] for r in valid))))
    summary=dict(total_ids=len(combined),unique_ids=len({r['advertisement_id'] for r in combined}),
        accepted=len(accepted),unresolved=len(failures),api_attempts=len(checkpoint['ledger']),
        statuses=dict(Counter(r['ai_status'] for r in combined)),
        technical_acceptance_rate=len(accepted)/len(combined),
        comparison=compare_rows(combined),by_sampling_group=strata,
        exact_errors=dict(Counter(r['validation_error'] for r in failures)),
        root_causes=dict(Counter(r['root_cause'] for r in failures)),
        additional_evidence_failures_among_category_rejections=sum(r['root_cause']=='uncertain_with_secondary_categories' and
            r['independent_evidence_check']=='unmatched' for r in failures),
        review_priority_counts={str(k):int(v) for k,v in review_frame.review_priority.value_counts().items()},
        accepted_low_confidence=sum(r['ai_confidence']=='low' for r in accepted),
        accepted_review_required=sum(bool(r['ai_review_required']) for r in accepted),
        cache_hit_ids=[r['advertisement_id'] for r in combined if r['ai_status']=='cache_hit'],
        no_direct_attempt_ids=[r['advertisement_id'] for r in combined if r['api_attempts_for_id']==0],
        recorded_usage_estimated_cost_usd=audit['provider_reported_usage_estimated_cost_usd'],
        reserved_unreported_usage_usd=audit['unreported_usage_reserved_usd'],
        rejected_response_usage_usd=float(sum((Decimal(str(r['recorded_attempt_cost_usd'])) for r in failures),Decimal(0))),
        provider_reported_input_tokens=sum(e['prompt_tokens'] or 0 for e in checkpoint['ledger']),
        provider_reported_output_tokens=sum(e['completion_tokens'] or 0 for e in checkpoint['ledger']),
        http_status_counts={str(k):v for k,v in Counter(e.get('http_status') for e in checkpoint['ledger']).items()},
        cache_receipts_explained=all(r['known_ledger_attempt'] for r in audit['cache_receipts']),
        source_run=str(run_dir),source_checkpoint_sha256=sha256(run_dir/'checkpoint.json'),new_paid_requests=0,
        accuracy_claimed=False)
    (output/'analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return combined,failures,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();_,_,summary=analyze(args.run,args.output)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
