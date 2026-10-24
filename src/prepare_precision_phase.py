"""Offline preparation of a three-case technical test and independent evaluation."""
import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from src.llm_classifier import Config
from src.recover_large_pilot import resilient_atomic_json as atomic_json
from src.run_precision_reliability import ALLOWED_IDS,code_hashes
from src.step3f_reliable_classifier import POLICY_PATH,PROMPT_PATH,source_hash
from src.validate_classification import ROOT,read_csv,sha256

BASE=ROOT/'data/processed/step3f-reliability-v1_1-20261023'
POLICY=ROOT/'docs/precision_first_inclusion_policy_v1.md'


def prepare(output):
    output=Path(output)
    if output.exists():raise FileExistsError('Use a new preparation folder')
    history=ROOT/'data/processed/step3f-span-pilot20-recovered-20261022/checkpoint.json'
    checkpoint=json.loads(history.read_text(encoding='utf-8'))
    candidates=read_csv(BASE/'known_outcome_retry_candidates.csv')
    if len(candidates)!=3 or set(candidates.advertisement_id)!=ALLOWED_IDS:raise ValueError('Known-outcome cases changed')
    for identifier in ALLOWED_IDS:
        attempts=[r for r in checkpoint['ledger'] if r['advertisement_id']==identifier]
        if not attempts or any(r['status']!='invalid_model_response' or r['usage_basis']!='provider_reported_tokens' for r in attempts):
            raise ValueError('Candidate has uncertain or changed historical request outcome')
    full=read_csv(ROOT/'data/processed/step3f-span-pilot20-20261022/prepared_jobs.csv').set_index('advertisement_id')
    blocked=[]
    for r in checkpoint['ledger']:
        if r['status'] in {'request_outcome_unknown','reserved_before_request'}:
            source=full.loc[r['advertisement_id']]
            blocked.append(dict(advertisement_id=r['advertisement_id'],source_sha256=source_hash(source.job_title,source.description_text),old_cache_key=r['cache_key']))
    for r in candidates.itertuples():
        original=full.loc[r.advertisement_id]
        if (r.job_title,r.description_text)!=(original.job_title,original.description_text):raise ValueError('Candidate source mismatch')
    if {source_hash(r.job_title,r.description_text) for r in candidates.itertuples()}&{b['source_sha256'] for b in blocked}:raise ValueError('Unknown-outcome content must not be retried')
    protected={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'data').rglob('*') if p.is_file()}
    output.mkdir(parents=True)
    candidates.to_csv(output/'reliability_cases.csv',index=False,encoding='utf-8-sig')
    atomic_json(output/'unknown_outcome_guard.json',dict(blocked_sources=blocked,automatically_retry=False))
    shutil.copyfile(POLICY,output/'precision_first_policy.md')
    shutil.copyfile(POLICY_PATH,output/'model_policy.md');shutil.copyfile(PROMPT_PATH,output/'model_prompt.txt')
    atomic_json(output/'pilot_manifest.json',dict(stage='precision-reliability-three',
        input_sha256=sha256(output/'reliability_cases.csv'),guard_sha256=sha256(output/'unknown_outcome_guard.json'),
        config=asdict(Config('gpt-4.1-mini-2025-04-14',.4,1.6,max_retries=0)),
        policy_sha256=sha256(POLICY_PATH),prompt_sha256=sha256(PROMPT_PATH),code_hashes=code_hashes(),
        history_checkpoint=str(history),history_checkpoint_sha256=sha256(history),
        history_accounted_cost_usd=sum(r['accounted_cost_usd'] for r in checkpoint['ledger']),
        precision_policy_sha256=sha256(POLICY),precision_policy_version='precision-first-v1'))
    human=read_csv(BASE/'human_labeling_200.csv')
    # Preserve existing manual entries, add only a blank optional cause field.
    if 'human_false_positive_cause' not in human:human['human_false_positive_cause']=''
    human.to_csv(output/'human_labeling_200.csv',index=False,encoding='utf-8-sig')
    reserve=json.loads((ROOT/'data/processed/step3f-span-pilot20-20261022/evaluation_reserve_manifest.json').read_text(encoding='utf-8'))
    reserve['source_by_id']={r.advertisement_id:source_hash(r.job_title,r.description_text) for r in human.itertuples()}
    if set(human.advertisement_id)!=set(reserve['reserved_ids']) or set(reserve['source_by_id'].values())!=set(reserve['content_keys']):raise ValueError('Independent reserve mismatch')
    atomic_json(output/'evaluation_reserve_manifest.json',reserve)
    audit=human[['advertisement_id','job_title','description_text']].copy()
    for c in ['duties_assessment','category_assessment','duties_assessor','inclusion_audit_notes','inclusion_review_resolved']:audit[c]=''
    audit.to_csv(output/'inclusion_audit_template.csv',index=False,encoding='utf-8-sig')
    atomic_json(output/'evaluation_input_contract.json',dict(
        predictions='Separate independent-reserve prediction CSV: original text, IDs, Step 3F 1.0/1.1 ai_* fields and exact source spans. No human label columns.',
        labels='Only human_labeling_200.csv, explicitly human verified with named independent reviewer.',
        audit='Separate optional inclusion audit, frozen before ground truth; auditor must differ from truth reviewer.',
        confidence='Qualitative and uncalibrated; select thresholds on separate calibration data, not final holdout.',
        duties_assessment_values=['clear','incidental_only','unrelated_duties','insufficient','contradictory'],
        category_assessment_values=['clear','insufficient','contradictory'],
        inclusion_review_resolved='true only after the named inclusion auditor resolves review flags; do not change model fields.'))
    changed=[p for p,h in protected.items() if not (ROOT/p).is_file() or sha256(ROOT/p)!=h]
    if changed:raise RuntimeError('Preservation failed: '+str(changed))
    atomic_json(output/'preservation_manifest.json',dict(files=protected,unchanged_files_verified=len(protected)))
    return dict(prepared=str(output),reliability_cases=len(candidates),unknown_outcomes_excluded=len(blocked),
        independent_labeling_rows=len(human),automatic_human_labels=0,paid_requests=0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    print(json.dumps(prepare(p.parse_args().output),indent=2,ensure_ascii=True))


if __name__=='__main__':main()
