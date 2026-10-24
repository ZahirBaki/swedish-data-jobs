"""Approved v3 publication: recompute frozen duties rules and extract without APIs."""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src import preview_duties_v3 as rules
from src.dashboard_data import load_dataset, summary, validate
from src.extract_requirements import extract_job, VERSION as EXTRACTOR_VERSION
from src.llm_classifier import stable_hash
from src.recover_large_pilot import resilient_atomic_json as atomic_json, writable_probe
from src.run_step4 import write_csv, json_rows
from src.step4_jobs import SOURCE, load_existing
from src.step6 import BASE, merge_frames
from src.validate_classification import ROOT, read_csv, sha256

POLICY_VERSION='precision-first-v3'
APPROVED_PREVIEW=ROOT/'data/processed/step6-v3-policy-preview-final-20261024'
ACTIVE_POLICY=ROOT/'docs/precision_first_inclusion_policy_v3.md'
DEFAULT_OUTPUT=ROOT/'data/processed/step6-v3-active-20261024'
AUTHORIZATION='User explicitly approved activation of proposed v3 for qualifying jobs from the 37-job preview; recompute all gates and do not force count.'


def approved_inputs(prepared,run_dir,preview=APPROVED_PREVIEW):
    prepared,run_dir,preview=Path(prepared),Path(run_dir),Path(preview)
    approved=json.loads((preview/'preview_summary.json').read_text(encoding='utf-8'))
    if approved['hashes']['rules']!=sha256(Path(rules.__file__)) or approved['hashes']['policy']!=sha256(rules.POLICY_PATH):
        raise ValueError('Rules/proposal differ from approved preview')
    if sha256(preview/'policy_proposal.md')!=approved['hashes']['policy']:
        raise ValueError('Frozen approved policy mismatch')
    state=json.loads((run_dir/'checkpoint.json').read_text(encoding='utf-8'))
    manifest=json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    identity=json.loads((run_dir/'identity.json').read_text(encoding='utf-8'))
    if (identity['preparation']!=sha256(prepared/'manifest.json') or
        manifest['input_sha256']!=sha256(prepared/'selected_jobs.csv') or
        approved['hashes']['selection']!=sha256(prepared/'selected_jobs.csv') or
        approved['hashes']['checkpoint']!=sha256(run_dir/'checkpoint.json')):
        raise ValueError('Saved run/source differs from approved preview')
    frame=read_csv(prepared/'selected_jobs.csv')
    if frame.advertisement_id.duplicated().any() or set(frame.advertisement_id)!=set(state['results']):
        raise ValueError('Incomplete or duplicate saved IDs')
    frozen=[json.loads(line) for line in (preview/'audit_receipts.jsonl').read_text(encoding='utf-8').splitlines()]
    frozen_by_id={r['advertisement_id']:r for r in frozen}
    if len(frozen_by_id)!=len(frozen) or set(frozen_by_id)!=set(frame.advertisement_id):raise ValueError('Frozen audit IDs mismatch')
    approved_ids={i for i,r in frozen_by_id.items() if r['preview_decision']=='would_qualify_v3'}
    # Integrity-check receipts by reproducing every decision from the canonical source.
    canonical=read_csv(SOURCE).set_index('advertisement_id').to_dict('index')
    originals={}
    for r in frame.to_dict('records'):
        identifier=r['advertisement_id']
        if any(r[k]!=canonical[identifier][k] for k in ['job_title','description_text']):raise ValueError('Canonical source mismatch')
        original=dict(r,**state['results'][identifier]);originals[identifier]=original
        if rules.decision(original)!=frozen_by_id[identifier]:raise ValueError('Frozen audit receipt differs from recomputation')
    return state,originals,approved_ids,approved


def evaluate(originals,approved_ids):
    receipts=[];included=[]
    for identifier,row in originals.items():
        assessment=rules.decision(row)  # No preview CSV is treated as a sufficient approval.
        qualifies=assessment['preview_decision']=='would_qualify_v3'
        allowed=qualifies and identifier in approved_ids
        if qualifies and not allowed:disposition='not_in_approved_preview'
        elif allowed:disposition='included'
        elif assessment['preview_decision']=='separate_lane':disposition=assessment['reporting_lane']
        elif assessment['preview_decision']=='abstain':disposition='abstention'
        elif assessment['preview_decision']=='technical_failure':disposition='technical_failure'
        else:disposition=assessment['saved_relevance'] or 'not_executed'
        receipt=dict(assessment,policy_version=POLICY_VERSION,activated=True,approved=allowed,
            inclusion_status=disposition,inclusion_policy_version=POLICY_VERSION,
            inclusion_method='automated_deterministic' if allowed else 'v3_abstention_or_exclusion',
            human_reviewed=False,authorization=AUTHORIZATION)
        receipt['audit_receipt_sha256']=stable_hash(receipt)
        receipts.append(receipt)
        if allowed:included.append((row,receipt))
    return receipts,included


def run(output=DEFAULT_OUTPUT,prepared=rules.PREPARED,run_dir=rules.RUN):
    output=Path(output);run_dir=Path(run_dir)
    if output.exists():raise FileExistsError('Publication requires a NEW directory')
    protected={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'data').rglob('*') if p.is_file()}
    state,originals,approved_ids,preview=approved_inputs(prepared,run_dir)
    receipts,included=evaluate(originals,approved_ids)
    base_jobs,base_facts=load_dataset(BASE)
    jobs=[];facts=[]
    for row,receipt in included:
        enriched=dict(row,classification_source_valid=True,classification_human_verified=False,
            classification_selected_run=str(run_dir.resolve()),step4_disposition='eligible_relevant',
            reporting_scope='core_industry',inclusion_method='automated_deterministic',
            inclusion_policy_version=POLICY_VERSION,inclusion_human_reviewed=False,
            inclusion_audit_receipt_sha256=receipt['audit_receipt_sha256'],
            inclusion_source_sha256=receipt['source_sha256'],inclusion_duties_evidence=receipt['duty_evidence'],
            inclusion_audit_engine_version=rules.VERSION)
        job,evidence=extract_job(enriched);jobs.append(job);facts.extend(evidence)
    new_jobs=pd.DataFrame(jobs)
    new_facts=pd.DataFrame(facts) if facts else base_facts.iloc[:0].copy()
    # Only additive metadata on copied historical rows. No retroactive v3 approval.
    historical=base_jobs.copy()
    historical['inclusion_method']='historical_source_validated_subset'
    historical['inclusion_policy_version']='historical-step5-accepted-subset'
    historical['inclusion_human_reviewed']=False
    merged,evidence=merge_frames(historical,base_facts,new_jobs,new_facts)
    validate(merged,evidence)
    baseline_rows=merged.set_index('advertisement_id').loc[base_jobs.advertisement_id,base_jobs.columns.drop('advertisement_id')].reset_index()
    pd.testing.assert_frame_equal(baseline_rows[base_jobs.columns],base_jobs,check_dtype=False)
    output.mkdir(parents=True);writable_probe(output)
    activation=dict(policy_version=POLICY_VERSION,active=True,authorization=AUTHORIZATION,
        approved_preview=str(APPROVED_PREVIEW),approved_preview_sha256=sha256(APPROVED_PREVIEW/'preview_summary.json'),
        approved_proposal_sha256=preview['hashes']['policy'],active_policy_sha256=sha256(ACTIVE_POLICY),
        rules_sha256=sha256(Path(rules.__file__)),publisher_sha256=sha256(Path(__file__)),
        extractor_version=EXTRACTOR_VERSION,extractor_sha256=sha256(ROOT/'src/extract_requirements.py'),
        selected_sha256=sha256(Path(prepared)/'selected_jobs.csv'),checkpoint_sha256=sha256(run_dir/'checkpoint.json'),
        no_api_requests=True,approved_preview_jobs=len(approved_ids),recomputed_passing_jobs=len(included),
        new_unique_jobs=len(merged)-len(base_jobs))
    atomic_json(output/'policy_activation.json',activation)
    (output/'active_policy.md').write_bytes(ACTIVE_POLICY.read_bytes())
    (output/'approved_proposal.md').write_bytes(rules.POLICY_PATH.read_bytes())
    with (output/'inclusion_audit_receipts.jsonl').open('w',encoding='utf-8') as f:
        for r in receipts:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    write_csv(pd.DataFrame(receipts),output/'classification_decisions.csv')
    for disposition in sorted({r['inclusion_status'] for r in receipts}):
        write_csv(pd.DataFrame([r for r in receipts if r['inclusion_status']==disposition]),output/('group_'+disposition+'.csv'))
    write_csv(merged,output/'jobs_requirements.csv')
    with (output/'jobs_requirements.jsonl').open('w',encoding='utf-8') as f:
        for r in json_rows(merged):f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    write_csv(evidence,output/'requirements_evidence.csv')
    provenance=merged[['advertisement_id','inclusion_method','inclusion_policy_version','inclusion_human_reviewed','classification_selected_run','ai_prompt_version','ai_model']].copy()
    write_csv(provenance,output/'job_inclusion_provenance.csv')
    registry,_,_=load_existing()
    counts=Counter(registry.step4_disposition);groups=Counter(r['inclusion_status'] for r in receipts)
    report=summary(merged,evidence)
    report.update(active_inclusion_policy=POLICY_VERSION,total_collected=len(registry),
        total_successfully_classified=int(registry.classification_source_valid.eq(True).sum())+
            sum(r.get('ai_status') in {'completed','cache_hit'} for r in state['results'].values()),
        total_relevant_analyzed=len(merged),excluded_irrelevant=counts['excluded_irrelevant']+groups['irrelevant'],
        uncertain=counts['set_aside_uncertain']+groups['uncertain']+groups['abstention'],
        semantic_uncertain=counts['set_aside_uncertain']+groups['uncertain'],
        technical_failures=counts['set_aside_technical_failure']+groups['technical_failure'],
        expansion_statuses=dict(groups),baseline_analyzed=len(base_jobs),new_analyzed=len(merged)-len(base_jobs),
        new_usable_requirements_jobs=sum(bool(j['requirements_data_usable']) for j in jobs),
        newly_extracted_jobs=len(jobs),classification_requests_this_publication=0,
        historical_classification_requests=len(state['ledger']),
        historical_classification_accounted_cost_usd=sum(r['accounted_cost_usd'] for r in state['ledger']),
        approval_methods=dict(Counter(merged.inclusion_method)),
        policy_note='Original 108 retain historical acceptance provenance; new jobs use approved automatic v3. Neither is independently human-verified ground truth.')
    write_csv(pd.DataFrame(report['coverage']),output/'requirement_coverage.csv')
    atomic_json(output/'step6_summary.json',report)
    # Re-read serialization and re-check every exact evidence offset before discovery.
    saved_jobs,saved_facts=load_dataset(output)
    if len(saved_jobs)!=len(merged):raise ValueError('Publication readback count mismatch')
    by_id=saved_jobs.set_index('advertisement_id').to_dict('index')
    for fact in saved_facts.to_dict('records'):
        source=by_id[fact['advertisement_id']][fact['source_field']]
        for prefix,column in [('evidence','evidence_text'),('match','matched_text')]:
            if source[int(float(fact[prefix+'_start'])):int(float(fact[prefix+'_end']))]!=fact[column]:
                raise ValueError('Extraction evidence offset mismatch')
    if any(not (ROOT/p).is_file() or sha256(ROOT/p)!=h for p,h in protected.items()):raise RuntimeError('Historical data preservation failed')
    atomic_json(output/'preservation_manifest.json',dict(files=protected,unchanged_files_verified=len(protected)))
    # A final atomic manifest is the existing dashboard discovery commit marker.
    atomic_json(output/'manifest.json',dict(stage='step6-combined',active_inclusion_policy=POLICY_VERSION,
        jobs_sha256=sha256(output/'jobs_requirements.jsonl'),evidence_sha256=sha256(output/'requirements_evidence.csv'),
        baseline_sha256=sha256(BASE/'jobs_requirements.jsonl'),activation_sha256=sha256(output/'policy_activation.json')))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    a=p.parse_args();print(json.dumps(run(a.output),indent=2))

if __name__=='__main__':main()
