"""Offline reliability investigation and blind human-review preparation."""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from src.recover_large_pilot import resilient_atomic_json as atomic_json
from src.resume_step3f_pilot import audit
from src.step3f_reliable_classifier import (Step3FReliableClassifier,PROMPT_PATH,
    PROMPT_VERSION,VALIDATION_VERSION,source_hash,SCHEMA)
from src.step3f_classifier import POLICY_PATH,POLICY_VERSION
from src.llm_classifier import Config,stable_hash
from src.validate_classification import ROOT,read_csv,sha256

REVIEW_NOTES={
    '31454174':'ETL/SAS/dbt duties support a warehouse/ETL primary; BI remains a plausible secondary. Human should decide dominant duties rather than trust title or model.',
    '31427098':'Highest priority: GIS/geodata stewardship may fit governance or adjacent work, but does not automatically establish core industry. Review dedicated duties and Other-category/core-scope choice.',
    '31365156':'Explicit ownership of AI-governance frameworks supports possible relevance under governance policy; distinguish accountable governance from generic legal/compliance work.',
    '31338012':'Explicit ordinary C#/.NET development duties support possible exclusion. Base the judgment on those duties, not merely absent AI keywords.',
    '31574285':'Explicit production LLM/RAG/agent integration supports possible AI engineering. Check whether ML Engineer secondary is independently supported beyond integrating models.',
    '31518660':'Highest priority: Toyota industry thesis has substantive DL/model-evaluation duties, but thesis status alone does not establish academic scope. Review industry versus academic research and ML Engineer secondary; keep thesis separate.',
    '31460068':'Review Analytics Engineer and Data / AI Management secondaries: technical support/strategy need not establish team/programme leadership. Primary scope depends on primary duties.',
    '31215842':'Review AI-method research versus physical/data-driven control modeling; employer/postdoc status alone cannot establish AI research.',
    '31454872':'Test-data/QA versus analytical data-platform responsibility remains a genuine policy boundary; do not force relevance.',
    '31575787':'Clinical doctoral collection/qualitative analysis does not automatically establish a Data/AI occupation. Uncertainty is permissible; doctoral attribute stays independent.',
}


def prepare_human_reserve(source,manifest_path,pilot,output):
    source,manifest_path,output=map(Path,[source,manifest_path,output])
    frame=read_csv(source);manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if frame.advertisement_id.duplicated().any() or set(frame.advertisement_id)!=set(manifest['reserved_ids']):
        raise ValueError('Independent reserve ID mismatch')
    keys={source_hash(r.job_title,r.description_text) for r in frame.itertuples()}
    if keys!=set(manifest['content_keys']):raise ValueError('Independent reserve content mismatch')
    pilot_keys={source_hash(r.job_title,r.description_text) for r in pilot.itertuples()}
    if keys&pilot_keys or set(frame.advertisement_id)&set(pilot.advertisement_id):raise ValueError('Pilot/evaluation leakage')
    if any(c in frame for c in ['ai_relevance','relevance','primary_category','assistant_suggested_relevance']):
        raise ValueError('Blind human sheet must not contain prediction labels')
    # Preserve any manually supplied fields; add only empty annotation columns.
    for col in ['human_relevance','human_primary_category','human_role_scope','human_opportunity_types',
                'human_review_notes','reviewer','human_verified']:
        if col not in frame:frame[col]=''
    frame.to_csv(output/'human_labeling_200.csv',index=False,encoding='utf-8-sig')
    options=dict(relevance=['relevant','irrelevant','uncertain'],
        primary_categories=SCHEMA['properties']['ai_primary_category']['enum'],
        scopes=SCHEMA['properties']['ai_scope']['enum'],
        opportunity_types=SCHEMA['properties']['ai_opportunity_types']['items']['enum'],
        verification='Enter true only after a named human reviewer reads full title/description; unresolved labels remain separate.')
    atomic_json(output/'human_label_options.json',options)
    atomic_json(output/'evaluation_manifest.json',dict(original_reserve_manifest=manifest,
        source_sha256=sha256(source),source_manifest_sha256=sha256(manifest_path),
        labeling_csv_sha256=sha256(output/'human_labeling_200.csv'),predictions_in_sheet=False,
        automatically_populated_labels=0,policy_version=POLICY_VERSION,policy_sha256=sha256(POLICY_PATH)))
    (output/'human_labeling_instructions.md').write_text(
        '# Independent human review\n\nRead the full title and description before labeling. '
        'Use `policy.md` and `human_label_options.json`; no rule/model labels are shown. '
        'Do not consult the technical pilot predictions while deciding these labels.\n\n'
        'Fill human_relevance (relevant/irrelevant/uncertain); set a canonical primary category '
        'only when relevant and defensible, otherwise leave blank. Optional role scope and '
        'opportunity types should be tracked separately. Explain ambiguous/hybrid decisions in '
        'human_review_notes and identify the reviewer. Set human_verified=true only after '
        'actual human inspection; never copy AI suggestions as verification. A verified but '
        'unresolved judgment remains excluded from resolved-label quality metrics.\n\n'
        'Keep this evaluation reserve out of prompt tuning. Evaluate only verified, resolved '
        'human labels after freezing predictions. The sampling unit is distinct source content '
        'and the eligible frame excludes policy-development inputs; results are not automatically '
        'unweighted accuracy across all 5,023 advertisements. Keep targeted diagnostic samples separate.\n',encoding='utf-8')
    return frame


def prepare(prepared,run_dir,reserve_source,reserve_manifest,output):
    prepared,run_dir,output=map(Path,[prepared,run_dir,output])
    if output.exists():raise FileExistsError('Use a new investigation directory')
    report,frame,old_client,_=audit(prepared,run_dir,budget_usd=1)
    protected={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'data').rglob('*') if p.is_file()}
    output.mkdir(parents=True)
    shutil.copyfile(run_dir/'checkpoint.json',output/'checkpoint_preserved.json')
    shutil.copyfile(run_dir/'classification_results.csv',output/'pilot_results_preserved.csv')
    shutil.copyfile(POLICY_PATH,output/'policy.md');shutil.copyfile(PROMPT_PATH,output/'prompt-v1.1.txt')
    state=old_client.state
    accepted={k:v for k,v in state['results'].items() if v['ai_status'] in {'completed','cache_hit'}}
    atomic_json(output/'accepted_historical_predictions.json',accepted)
    rows=[];failures=[];blocked=[];retry_ids=[]
    for source in frame.to_dict('records'):
        identifier=source['advertisement_id'];result=state['results'][identifier]
        if result['ai_status'] in {'completed','cache_hit'}:
            rd=result['ai_relevance']!=source['relevance'];cd=(result['ai_primary_category'] or '')!=source['primary_category']
            if rd or cd or identifier in REVIEW_NOTES:
                rows.append(dict(source,**result,relevance_disagreement=rd,category_disagreement=cd,
                    provisional_review_observation=REVIEW_NOTES.get(identifier,'Inspect full duties and dominant category.'),
                    human_relevance='',human_primary_category='',human_role_scope='',human_review_notes='',human_verified=''))
        else:
            related=[x for x in state['ledger'] if x['advertisement_id']==identifier]
            for item in related:
                diagnostic=item.get('model_response_diagnostic',{});content=diagnostic.get('content') or ''
                failures.append(dict(advertisement_id=identifier,job_title=source['job_title'],status=item['status'],
                    error=diagnostic.get('validation_error') or item.get('network_error_type'),
                    finish_reason=diagnostic.get('finish_reason'),completion_tokens=item.get('completion_tokens'),
                    output_cap=old_client.config.max_output_tokens,captured_characters=len(content),
                    whitespace_fraction=sum(c.isspace() for c in content)/len(content) if content else None,
                    diagnostic_content_truncated=diagnostic.get('content_truncated'),
                    accounted_cost_usd=item['accounted_cost_usd'],usage_basis=item['usage_basis']))
            if any(x['status'] in {'request_outcome_unknown','reserved_before_request'} for x in related):
                blocked.append(dict(advertisement_id=identifier,source_sha256=source_hash(source['job_title'],source['description_text']),
                    old_cache_key=result['ai_cache_key'],reason='Unknown billing: do not retry under any prompt version without reconciliation/approval'))
            else:retry_ids.append(identifier)
    review=pd.DataFrame(rows)
    for col in review:
        review[col]=review[col].map(lambda v:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v)
    review.to_csv(output/'disagreements_and_ambiguities.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(failures).to_csv(output/'failures.csv',index=False,encoding='utf-8-sig')
    atomic_json(output/'unknown_outcome_guard.json',dict(blocked_sources=blocked,automatically_retry=False))
    # Only paid-response technical failures are candidate inputs; not approvals or executed retries.
    candidates=frame[frame.advertisement_id.isin(retry_ids)].copy()
    candidates.to_csv(output/'known_outcome_retry_candidates.csv',index=False,encoding='utf-8-sig')
    new_client=Step3FReliableClassifier(Config(old_client.config.model,.4,1.6,max_retries=0),
        POLICY_PATH.read_text(encoding='utf-8'),output/'unused-cache',output/'unused-checkpoint.json',1,
        execute=False,blocked_source_hashes=[x['source_sha256'] for x in blocked])
    plans=[dict(advertisement_id=r.advertisement_id,**new_client.estimate(r.job_title,r.description_text)) for r in candidates.itertuples()]
    pd.DataFrame(plans).to_csv(output/'candidate_dry_run_costs.csv',index=False,encoding='utf-8-sig')
    reserve=prepare_human_reserve(reserve_source,reserve_manifest,frame,output)
    summary=dict(original_pilot_total=len(frame),accepted_preserved=len(accepted),unresolved=len(failures),
        relevance_disagreements=int(review.relevance_disagreement.sum()),category_disagreements=int(review.category_disagreement.sum()),
        unique_disagreement_ads=int((review.relevance_disagreement|review.category_disagreement).sum()),
        policy_version=POLICY_VERSION,new_prompt_version=PROMPT_VERSION,new_validation_version=VALIDATION_VERSION,
        policy_changed=False,old_predictions_compatible_with_new_cache=False,
        original_accounted_cost_usd=report['accounted_cost_usd'],
        original_usage_priced_cost_usd=report['provider_reported_usage_estimated_cost_usd'],
        unknown_usage_hold_usd=report['unreported_usage_reserved_usd'],
        unknown_outcomes_blocked=len(blocked),known_outcome_retry_candidates=len(candidates),
        candidate_expected_cost_usd=sum(x['expected_cost_usd'] for x in plans),
        candidate_conservative_cost_usd=sum(x['reserved_cost_per_attempt_usd'] for x in plans),
        independent_human_labeling_rows=len(reserve),automatically_populated_human_labels=0,
        new_paid_requests=0,classification_quality_metrics_calculated=False,
        new_version_live_validated=False,ready_for_full_dataset=False,
        original_checkpoint_sha256=sha256(run_dir/'checkpoint.json'),
        new_classifier_sha256=sha256(ROOT/'src/step3f_reliable_classifier.py'),new_prompt_sha256=sha256(PROMPT_PATH))
    atomic_json(output/'reliability_summary.json',summary)
    changed=[p for p,digest in protected.items() if not (ROOT/p).is_file() or sha256(ROOT/p)!=digest]
    if changed:raise RuntimeError('Original data preservation failed: '+str(changed))
    atomic_json(output/'preservation_manifest.json',dict(files=protected,unchanged_files_verified=len(protected)))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,default=ROOT/'data/processed/step3f-span-pilot20-20261022')
    parser.add_argument('--run',type=Path,default=ROOT/'data/processed/step3f-span-pilot20-recovered-20261022')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.prepared,args.run,args.prepared/'independent_evaluation_reserve.csv',
        args.prepared/'evaluation_reserve_manifest.json',args.output),ensure_ascii=True,indent=2))


if __name__=='__main__':main()
