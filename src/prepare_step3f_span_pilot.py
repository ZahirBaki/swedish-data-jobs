"""Prepare a targeted 20-ad technical pilot and an isolated, blind review reserve.

This module is offline. It never calls the model or changes the frozen classifier.
Selection reasons are diagnostic hypotheses, not verified occupational labels.
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from src.llm_classifier import Config, stable_hash
from src.recover_large_pilot import resilient_atomic_json as atomic_json
from src.run_step3f import validate_prepared
from src.step3f_classifier import Step3FClassifier, source_spans, POLICY_PATH
from src.validate_classification import ROOT, read_csv, representative_sample, sha256

SEED=20261022
SELECTION=[
    ('31454174','previous_evidence_failure','BI/ETL; Swedish title and punctuation'),
    ('31215842','previous_evidence_failure','Academic AI/digital twins; Swedish diacritics and dash'),
    ('31153263','previous_evidence_failure','Cooking duties versus incidental municipal AI branding'),
    ('31276346','previous_evidence_failure','Data engineering; previously rejected quotation'),
    ('31427098','previous_evidence_failure','GIS borderline occupation'),
    ('31365156','previous_evidence_failure','AI Act governance/compliance boundary'),
    ('31505124','new_core_candidate','Production AI versus general internal software'),
    ('31083431','new_core_candidate','Dedicated data engineering duties'),
    ('31150357','new_core_candidate','Warehouse duties and Swedish text'),
    ('31460068','new_core_candidate','BI and analytical platform duties'),
    ('31557998','new_unrelated_candidate','Sales of non-AI products despite employer AI activities'),
    ('31093621','new_unrelated_candidate','Teaching versus employer AI branding'),
    ('31539022','new_academic_candidate','Doctoral ML algorithms; independent opportunity flag'),
    ('30973459','new_academic_candidate','Postdoctoral statistics/data science boundary'),
    ('31395973','new_borderline_candidate','AI development-tool platform ownership/management'),
    ('31454872','new_borderline_candidate','Test-data/QA versus analytical data engineering'),
    ('31575787','new_borderline_candidate','Domain doctoral research versus AI methodology'),
    ('31338012','new_rule_uncertain','General C#/.NET software versus dedicated Data/AI duties'),
    ('31574285','new_rule_uncertain','Generic software title with substantive AI duties to inspect'),
    ('31518660','new_opportunity_candidate','Deep-learning thesis; separate thesis attribute'),
]


def content_key(row):
    return stable_hash(dict(job_title=row['job_title'],description_text=row['description_text']))


def coverage_audit(frame,client):
    """One explicit routing row per ID; same-content reuse never removes a record."""
    if frame.advertisement_id.duplicated().any():raise ValueError('Duplicate advertisement IDs')
    first={}; rows=[]
    for row in frame.to_dict('records'):
        key=client.cache_key(row['job_title'],row['description_text'])
        identifier=row['advertisement_id']
        if not row['description_text'].strip():
            action,representative='local_uncertain_missing_description',identifier
        elif key in first:
            action,representative='reuse_if_compatible_success_available',first[key]
        else:
            action,representative='distinct_nonempty_input',identifier
            first[key]=identifier
        rows.append(dict(advertisement_id=identifier,cache_key=key,coverage_action=action,
                         representative_advertisement_id=representative))
    summary=dict(total_advertisements=len(frame),unique_ids=frame.advertisement_id.nunique(),
        missing_descriptions=sum(r['coverage_action']=='local_uncertain_missing_description' for r in rows),
        distinct_nonempty_inputs=len(first),
        additional_same_content_advertisements=sum(r['coverage_action']=='reuse_if_compatible_success_available' for r in rows),
        omitted_advertisements=0,
        note='Reuse requires a valid compatible success; failures remain explicit, never dropped or classified irrelevant.')
    assert summary['total_advertisements']==sum(summary[k] for k in [
        'missing_descriptions','distinct_nonempty_inputs','additional_same_content_advertisements'])
    return rows,summary


def select_pilot(frame,checkpoint):
    indexed=frame.set_index('advertisement_id',drop=False)
    selected=[]
    for identifier,group,reason in SELECTION:
        if identifier not in indexed.index:raise ValueError('Required pilot ID missing: '+identifier)
        row=indexed.loc[identifier].to_dict()
        if not row['description_text'].strip():raise ValueError('Live pilot requires full descriptions')
        errors=[x.get('model_response_diagnostic',{}).get('validation_error','')
            for x in checkpoint['ledger'] if x['advertisement_id']==identifier
            and x['status']=='invalid_model_response']
        if group=='previous_evidence_failure':
            if 'Evidence is not a verbatim excerpt of title/description' not in errors:
                raise ValueError('Evidence failure not confirmed in checkpoint: '+identifier)
        elif any(row.get(c,'') for c in ['previous_step3e_ai_status','previous_step3d_ai_status']):
            raise ValueError('New pilot case was already processed: '+identifier)
        row.update(pilot_selection_group=group,pilot_selection_reason=reason,
            pilot_previous_validation_errors=json.dumps(errors,ensure_ascii=False),
            pilot_purpose='targeted_technical_validation_not_independent_evaluation')
        selected.append(row)
    import pandas as pd
    pilot=pd.DataFrame(selected)
    if len(pilot)!=20 or pilot.advertisement_id.nunique()!=20:raise ValueError('Pilot must have 20 unique IDs')
    if len({content_key(row) for row in selected})!=20:raise ValueError('Pilot must exercise 20 distinct inputs')
    return pilot


def evaluation_reserve(frame,pilot,size=200,seed=SEED):
    # Earlier 200/30 have influenced policy. Exclude all same-content copies, not just IDs.
    historical=frame[[c for c in ['previous_step3e_ai_status','previous_step3d_ai_status'] if c in frame]].ne('').any(axis=1)
    excluded=set(content_key(row) for row in frame[historical].to_dict('records'))
    excluded.update(content_key(row) for row in pilot.to_dict('records'))
    eligible=frame.copy()
    eligible['reserve_content_key']=[content_key(row) for row in eligible.to_dict('records')]
    eligible=eligible[~eligible.reserve_content_key.isin(excluded)&eligible.description_text.str.strip().ne('')]
    eligible=eligible.sort_values('advertisement_id').drop_duplicates('reserve_content_key')
    if len(eligible)<size:raise ValueError('Insufficient untouched distinct inputs for review reserve')
    sample=representative_sample(eligible,size=size,seed=seed)
    if set(sample.reserve_content_key)&excluded:raise AssertionError('Holdout content leakage')
    # No predictions or selection strata appear in the human sheet.
    fields=[c for c in ['advertisement_id','job_title','description_text','employer','publication_date','municipality'] if c in sample]
    blind=sample[fields].copy()
    for field in ['human_relevance','human_primary_category','human_review_notes','reviewer','human_verified']:
        blind[field]=''
    selected_keys=set(sample.reserve_content_key)
    all_reserved_ids=sorted(r['advertisement_id'] for r in frame.to_dict('records') if content_key(r) in selected_keys)
    return blind,dict(seed=seed,size=len(blind),sampling_unit='distinct original title/description content',
        sampling_frame_size=len(eligible),design='proportional rule-stratified distinct-content reserve, not all-ad population sample',
        reserved_ids=blind.advertisement_id.tolist(),content_keys=sorted(selected_keys),
        all_same_content_reserved_ids=all_reserved_ids,development_content_exclusions=len(excluded),
        human_verified_labels=0,pilot_overlap_ids=0,pilot_overlap_content=0,
        note='Blind future human evaluation reserve; never supplied as labels or included in this technical pilot.')


def formatting_audit(frame):
    spans_checked=0; flags=Counter()
    for row in frame.to_dict('records'):
        spans=source_spans(row['job_title'],row['description_text'])
        for field in ['job_title','description_text']:
            text=row[field]; parts=[s for s in spans if s['source_field']==field]
            if ''.join(s['text'] for s in parts)!=text:raise AssertionError('Incomplete source coverage')
            for span in parts:
                if text[span['start']:span['end']]!=span['text']:raise AssertionError('Invalid source offset')
                spans_checked+=1
        text=row['job_title']+row['description_text']
        flags['ads_with_swedish_characters']+=any(c in text for c in 'åäöÅÄÖ')
        flags['ads_with_line_breaks']+=any(c in text for c in '\r\n\u2028\u2029')
        flags['ads_with_punctuation']+=any(c in text for c in '.,;:!?–—-/()')
    return dict(advertisements=len(frame),source_spans_checked=spans_checked,
        full_text_roundtrip=True,all_offsets_exact=True,normalization_applied=False,**flags)


def prepare(full_prepared,output,checkpoint_path,reserve_size=200):
    full_prepared,output,checkpoint_path=map(Path,[full_prepared,output,checkpoint_path])
    if output.exists():raise FileExistsError('Use a new pilot preparation output')
    manifest,frame=validate_prepared(full_prepared)
    checkpoint=json.loads(checkpoint_path.read_text(encoding='utf-8'))
    pilot=select_pilot(frame,checkpoint)
    blind,reserve=evaluation_reserve(frame,pilot,size=reserve_size)
    protected={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'data').rglob('*') if p.is_file()}
    output.mkdir(parents=True)
    pilot.to_csv(output/'prepared_jobs.csv',index=False,encoding='utf-8-sig')
    for filename in ['policy.md','prompt.txt','schema.json']:
        shutil.copyfile(full_prepared/filename,output/filename)
    new_manifest=dict(manifest,total_advertisements=len(pilot),prepared_sha256=sha256(output/'prepared_jobs.csv'),
        purpose='targeted_source_span_technical_pilot_not_evaluation',
        parent_preparation_path=str(full_prepared.resolve()),parent_manifest_sha256=sha256(full_prepared/'manifest.json'),
        pilot_selection_groups=dict(Counter(pilot.pilot_selection_group)),
        previous_failure_checkpoint=str(checkpoint_path.resolve()),previous_failure_checkpoint_sha256=sha256(checkpoint_path),
        selection_code_sha256=sha256(Path(__file__)))
    atomic_json(output/'manifest.json',new_manifest)
    blind.to_csv(output/'independent_evaluation_reserve.csv',index=False,encoding='utf-8-sig')
    atomic_json(output/'evaluation_reserve_manifest.json',reserve)
    client=Step3FClassifier(Config('gpt-4.1-mini-2025-04-14',.4,1.6,max_retries=0),
        POLICY_PATH.read_text(encoding='utf-8'),output/'audit-cache',output/'unused-checkpoint.json',1,execute=False)
    routing,coverage=coverage_audit(frame,client)
    import pandas as pd
    pd.DataFrame(routing).to_csv(output/'full_dataset_coverage.csv',index=False,encoding='utf-8-sig')
    atomic_json(output/'full_dataset_coverage.json',coverage)
    atomic_json(output/'source_span_formatting_audit.json',formatting_audit(pilot))
    changed=[p for p,digest in protected.items() if not (ROOT/p).is_file() or sha256(ROOT/p)!=digest]
    if changed:raise RuntimeError('Preservation check failed: '+str(changed))
    atomic_json(output/'preservation_manifest.json',dict(files=protected,unchanged_files_verified=len(protected)))
    return dict(prepared=str(output),pilot_size=20,confirmed_previous_evidence_failures=6,new_cases=14,
        independent_evaluation_reserve=len(blind),coverage=coverage,paid_requests=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-prepared',type=Path,default=ROOT/'data/processed/step3f-prepared-20261022')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'data/processed/step3e-live-recovered-20261022/checkpoint.json')
    args=parser.parse_args()
    print(json.dumps(prepare(args.full_prepared,args.output,args.checkpoint),indent=2,ensure_ascii=True))


if __name__=='__main__':main()
