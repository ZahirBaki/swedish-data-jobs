"""Non-paid Step 4 pilot: freeze existing classifications and extract requirements."""
import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.extract_requirements import extract_job,VERSION
from src.requirements_summary import summarize
from src.step4_jobs import load_existing,select_pilot,SOURCE
from src.recover_large_pilot import resilient_atomic_json as atomic_json
from src.validate_classification import ROOT,sha256


def write_csv(frame,path):
    serial=frame.copy()
    for column in serial:
        serial[column]=serial[column].map(lambda x:json.dumps(x,ensure_ascii=False,allow_nan=False) if isinstance(x,(list,dict)) else x)
    serial.to_csv(path,index=False,encoding='utf-8-sig')


def json_rows(frame):
    return json.loads(frame.to_json(orient='records',force_ascii=False))


def run(output,size=50,seed=20261023,dry_run=False,plan_llm=False,llm_plan_max_ads=10):
    output=Path(output)
    if output.exists():raise FileExistsError('Step 4 requires a new versioned output folder')
    if type(size) is not int or size<1:raise ValueError('Pilot size must be positive')
    if type(llm_plan_max_ads) is not int or llm_plan_max_ads<0:raise ValueError('LLM planning count must be nonnegative')
    registry,history,sources=load_existing()
    pilot=select_pilot(registry,size,seed)
    if pilot.empty:raise ValueError('No accepted, source-validated relevant jobs available')
    protected={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'data').rglob('*') if p.is_file()}
    output.mkdir(parents=True)
    write_csv(registry,output/'classification_registry.csv')
    atomic_json(output/'classification_history.json',history)
    write_csv(pilot,output/'pilot_input.csv')
    count=dict(Counter(registry.step4_disposition))
    plan=dict(total_source_advertisements=len(registry),classification_dispositions=count,
        available_accepted_relevant=count.get('eligible_relevant',0),selected_pilot_jobs=len(pilot),
        selection_seed=seed,selection='fixed-seed category-stratified sample of existing relevant AI pilot decisions',
        classifier_requests=0,extraction_method='deterministic_rules',paid_extraction_requests=0,
        expected_api_cost_usd=0.0,conservative_api_cost_usd=0.0,
        llm_execution_enabled=False,requires_complete_corpus_classification=False,
        accepted_is_human_verified=False,
        scope='Existing accepted relevant pilot subset; no full Swedish job-market demand or accuracy claims.')
    atomic_json(output/'dry_run_report.json',plan)
    code_paths=[ROOT/'src'/name for name in ['step4_jobs.py','extract_requirements.py','requirements_summary.py','run_step4.py']]
    manifest=dict(stage='step4',extractor_version=VERSION,source_path=str(SOURCE),source_sha256=sha256(SOURCE),
        classification_sources=sources,code_sha256={p.name:sha256(p) for p in code_paths},
        pilot_input_sha256=sha256(output/'pilot_input.csv'),seed=seed,dry_run=dry_run,
        no_classification_retries=True,api_requests_made=0)
    if not dry_run:
        jobs=[];records=[]
        for original in json_rows(pilot):
            job,facts=extract_job(original);jobs.append(job);records.extend(facts)
        jobs=pd.DataFrame(jobs);facts=pd.DataFrame(records)
        if facts.empty:facts=pd.DataFrame(columns=['requirement_id','advertisement_id','kind','value','modality'])
        write_csv(jobs,output/'jobs_requirements.csv')
        with (output/'jobs_requirements.jsonl').open('w',encoding='utf-8') as file:
            for row in json_rows(jobs):file.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        write_csv(facts,output/'requirements_evidence.csv')
        for filename,kinds in [('job_skills.csv',['programming_language','cloud_platform','data_tool']),
            ('job_language_requirements.csv',['language']),('job_education_requirements.csv',['education']),
            ('job_experience_requirements.csv',['experience']),('job_seniority.csv',['seniority']),
            ('job_employment_location.csv',['employment_type','working_time','work_arrangement','source_metadata'])]:
            write_csv(facts[facts.kind.isin(kinds)],output/filename)
        summary=summarize(jobs,facts);atomic_json(output/'requirements_summary.json',summary)
        # Full-table manual review supports validity precision; missing facts need
        # an independent per-ad review and cannot be inferred from output precision.
        review=facts.copy()
        for c in ['human_valid','human_modality','human_correct_value','human_notes','human_verified','reviewer']:review[c]=''
        write_csv(review,output/'manual_review_requirements.csv')
        review_jobs=jobs[['advertisement_id','job_title','description_text']].copy()
        for c in ['human_missing_requirements','human_notes','human_verified','reviewer']:review_jobs[c]=''
        write_csv(review_jobs,output/'manual_review_jobs.csv')
        queue=jobs[jobs.requirements_review_required|~jobs.requirements_data_usable].copy()
        write_csv(queue,output/'ambiguous_requirements_queue.csv')
        optional_plan=[]
        if plan_llm:
            for row in json_rows(queue.head(llm_plan_max_ads)):
                chars=len(row['job_title'])+len(row['description_text'])+2500
                expected_input=(chars+3)//4+256
                conservative_input=len((row['job_title']+row['description_text']).encode('utf-8'))+5000
                optional_plan.append(dict(advertisement_id=row['advertisement_id'],expected_input_tokens=expected_input,
                    conservative_input_tokens=conservative_input,expected_output_tokens=1000,output_cap=4096,
                    expected_cost_usd=(expected_input*.4+1000*1.6)/1e6,
                    conservative_cost_usd=(conservative_input*.4+4096*1.6)/1e6))
        atomic_json(output/'optional_llm_plan.json',dict(enabled=False,planning_only=True,
            planned_jobs=len(optional_plan),draft_model='gpt-4.1-mini-2025-04-14',
            configured_input_usd_per_million=.4,configured_output_usd_per_million=1.6,
            expected_cost_usd=sum(r['expected_cost_usd'] for r in optional_plan),
            conservative_cost_usd=sum(r['conservative_cost_usd'] for r in optional_plan),requests=optional_plan,
            caveat='Provisional planning heuristic only; no LLM extraction backend or execution flag. A future API phase needs explicit approval, final prompt/schema, budget and resumable receipts. Deterministic extraction remains $0.'))
        plan.update(extracted_requirement_records=len(facts),usable_requirements_jobs=int(jobs.requirements_data_usable.sum()),
            jobs_queued_for_review=len(queue),human_validated_extraction_accuracy=None)
        atomic_json(output/'run_summary.json',plan)
    changed=[p for p,h in protected.items() if not (ROOT/p).is_file() or sha256(ROOT/p)!=h]
    if changed:raise RuntimeError('Original dataset preservation check failed: '+str(changed))
    atomic_json(output/'preservation_manifest.json',dict(files=protected,unchanged_files_verified=len(protected)))
    atomic_json(output/'manifest.json',manifest)
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--size',type=int,default=50);parser.add_argument('--seed',type=int,default=20261023)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--plan-llm',action='store_true',help='Cost planning only; cannot send API calls')
    parser.add_argument('--llm-plan-max-ads',type=int,default=10)
    args=parser.parse_args()
    print(json.dumps(run(args.output,args.size,args.seed,args.dry_run,args.plan_llm,args.llm_plan_max_ads),indent=2,ensure_ascii=True))


if __name__=='__main__':main()
