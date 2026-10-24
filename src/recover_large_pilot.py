"""Read-only audit and new-directory recovery of interrupted Step 3E runs."""
import argparse
import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

from src.llm_classifier import AI_FIELDS,SCHEMA,VALIDATION_VERSION,Config,cost_usd,stable_hash,validate_prediction
from src.run_ai_classification import write_results
from src.run_large_pilot import validate_prepared,operational_summary
from src.step3e_classifier import Step3EClassifier,POLICY_PATH,PROMPT_PATH,PROMPT_VERSION
from src.validate_classification import ROOT,sha256


def resilient_atomic_json(path,value,attempts=6,sleep=time.sleep):
    """Retry only Windows-style replacement errors, never a model request."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('w',encoding='utf-8') as file:
            json.dump(value,file,ensure_ascii=False,indent=2,allow_nan=False)
            file.flush();os.fsync(file.fileno())
        for attempt in range(attempts):
            try:
                os.replace(temporary,path)
                break
            except PermissionError:
                if attempt==attempts-1:raise
                sleep(.05*2**attempt)
    finally:
        try:temporary.unlink(missing_ok=True)
        except OSError:pass  # Do not hide the original replacement error.


def writable_probe(directory):
    directory=Path(directory)
    target=directory/('.checkpoint-write-probe-'+uuid.uuid4().hex+'.json')
    try:
        resilient_atomic_json(target,{'probe':1})
        resilient_atomic_json(target,{'probe':2})
        if json.loads(target.read_text(encoding='utf-8'))!={'probe':2}:
            raise ValueError('Write/replace verification failed')
    finally:target.unlink(missing_ok=True)
    return True


@contextmanager
def resilient_checkpoint_writes():
    # Keep existing classifier/runner source hashes and request identity intact.
    # Changes only this process's JSON file writer, not transport/API retry logic.
    import src.llm_classifier as transport
    import src.run_large_pilot as runner
    previous=(transport.atomic_json,runner.atomic_json)
    transport.atomic_json=runner.atomic_json=resilient_atomic_json
    try:yield
    finally:transport.atomic_json,runner.atomic_json=previous


def inspect(run_dir,budget_usd,cache_dir=None):
    run_dir=Path(run_dir).resolve()
    plan=json.loads((run_dir/'plan.json').read_text(encoding='utf-8'))
    identity=plan['identity'];config=Config(**identity['config'])
    frame,manifest=validate_prepared(plan['prepared']);frame=frame.iloc[:identity['max_ads']].copy()
    expected=dict(stage='step3e',config=asdict(config),sample_sha256=manifest['sample_sha256'],
        sampling_manifest_sha256=sha256(Path(plan['prepared'])/'sampling_manifest.json'),ids=frame.advertisement_id.tolist(),
        policy_sha256=sha256(POLICY_PATH),prompt_sha256=sha256(PROMPT_PATH),prompt_version=PROMPT_VERSION,
        schema_sha256=stable_hash(SCHEMA),validation_version=VALIDATION_VERSION,max_ads=identity['max_ads'],
        code_hashes={n:sha256(ROOT/'src'/n) for n in ['run_large_pilot.py','step3e_classifier.py','llm_classifier.py']})
    if expected!=identity:raise ValueError('Original run and current prompt/schema/policy/code/config are incompatible')
    if sha256(run_dir/'pilot_input.csv')!=plan['pilot_input_sha256']:
        raise ValueError('Frozen run input hash changed')
    for name,digest in identity['code_hashes'].items():
        if sha256(run_dir/name)!=digest:raise ValueError('Source snapshot differs from original identity')
    if sha256(run_dir/'policy_snapshot.md')!=identity['policy_sha256'] or sha256(run_dir/'prompt_snapshot.txt')!=identity['prompt_sha256']:
        raise ValueError('Policy/prompt snapshot differs')
    if stable_hash(json.loads((run_dir/'output_schema.json').read_text(encoding='utf-8')))!=identity['schema_sha256']:
        raise ValueError('Schema snapshot differs')
    cache_dir=Path(cache_dir) if cache_dir else ROOT/'data/processed/.llm_cache'
    client=Step3EClassifier(config,POLICY_PATH.read_text(encoding='utf-8'),cache_dir,
        run_dir/'checkpoint.json',budget_usd,execute=False)
    if client.system_prompt!=(run_dir/'system_prompt.txt').read_text(encoding='utf-8'):
        raise ValueError('Rendered system prompt differs')
    checkpoint=client.state;by_id=frame.set_index('advertisement_id');attempts={}
    for item in checkpoint['ledger']:
        identifier=item['advertisement_id']
        if identifier not in by_id.index or item['attempt_id'] in attempts:
            raise ValueError('Unexpected ID or duplicate attempt in checkpoint')
        source=by_id.loc[identifier]
        if item['cache_key']!=client.cache_key(source.job_title,source.description_text):
            raise ValueError('Ledger request identity mismatch')
        if item['usage_basis']=='provider_reported_tokens':
            tokens=[item.get('prompt_tokens'),item.get('completion_tokens')]
            if any(type(n) is not int or n<0 for n in tokens) or abs(cost_usd(*tokens,config)-item['accounted_cost_usd'])>1e-12:
                raise ValueError('Reported usage/cost inconsistent')
        attempts[item['attempt_id']]=item
    accepted=[];rejected=[]
    for identifier,result in checkpoint['results'].items():
        if identifier not in by_id.index:raise ValueError('Checkpoint result outside frozen sample')
        source=by_id.loc[identifier]
        if result['ai_cache_key']!=client.cache_key(source.job_title,source.description_text):
            raise ValueError('Checkpoint result identity mismatch')
        if result.get('ai_human_verified') is not False:raise ValueError('AI result must not claim human verification')
        if result.get('ai_prompt_version')!=identity['prompt_version']:raise ValueError('Result prompt version mismatch')
        if result['ai_status'] in {'completed','cache_hit'}:
            validate_prediction({k:result[k] for k in AI_FIELDS},source.job_title,source.description_text,allow_formatting=True)
            accepted.append(identifier)
        else:rejected.append(identifier)
    receipts=[];pending=[];estimates=[];additional_cache=[];unknown_receipts=[]
    for source in frame.itertuples():
        key=client.cache_key(source.job_title,source.description_text)
        receipt=client.cached(source.job_title,source.description_text)
        marker=cache_dir/(key+'.pending.json')
        if marker.exists():pending.append(dict(advertisement_id=source.advertisement_id,marker=str(marker)))
        if receipt:
            known=receipt['attempt_id'] in attempts
            receipts.append(dict(advertisement_id=source.advertisement_id,attempt_id=receipt['attempt_id'],known_ledger_attempt=known))
            if not known:unknown_receipts.append(receipts[-1])
            if source.advertisement_id not in checkpoint['results']:additional_cache.append(source.advertisement_id)
        if source.advertisement_id not in checkpoint['results'] and not receipt and not marker.exists():
            estimates.append(client.estimate(source.job_title,source.description_text))
    if unknown_receipts:
        raise ValueError('Cache receipts have attempts absent from this ledger; inspect before recovery to avoid losing costs')
    # Unfinished reservations/pending markers are preserved, never implicitly cleared.
    unfinished=[x for x in checkpoint['ledger'] if x['status'] in {'reserved_before_request','request_outcome_unknown'}]
    breakdown=client.cost_breakdown()
    report=dict(selected=len(frame),ledger_entries=len(checkpoint['ledger']),result_entries=len(checkpoint['results']),
        accepted_ids=accepted,rejected_ids=rejected,accepted=len(accepted),rejected=len(rejected),
        remaining_unrecorded=len(frame)-len(checkpoint['results']),additional_cached_ids=additional_cache,
        potential_uncached_ads=len(estimates),unique_cache_receipt_attempts=len({r['attempt_id'] for r in receipts}),
        pending_markers=pending,unfinished_ledger_attempts=len(unfinished),cache_receipts=receipts,
        accounted_cost_usd=client.spent,**breakdown,total_budget_usd=budget_usd,
        remaining_budget_usd=float(Decimal(str(budget_usd))-Decimal(str(client.spent))),
        remaining_expected_usd=sum(e['expected_cost_usd'] for e in estimates),
        remaining_conservative_usd=sum(e['reserved_cost_per_attempt_usd'] for e in estimates),
        checkpoint_sha256=sha256(run_dir/'checkpoint.json'),compatibility_verified=True,new_paid_requests=0)
    if report['remaining_budget_usd']<0:raise ValueError('Budget below already accounted cost')
    return report,frame,checkpoint,plan


def recover(source,output,budget_usd=2,cache_dir=None):
    source,output=Path(source).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('Recovery requires a new destination directory')
    report,frame,checkpoint,plan=inspect(source,budget_usd,cache_dir)
    if report['pending_markers'] or report['unfinished_ledger_attempts']:
        raise ValueError('Unknown request outcomes require inspection before automatic recovery')
    output.mkdir(parents=True,exist_ok=False)
    archive=output/'source_artifacts';archive.mkdir()
    for path in source.iterdir():
        if path.is_file() and path.name!='run.lock':
            shutil.copyfile(path,archive/path.name)
            shutil.copyfile(path,output/path.name)
    # The entire checkpoint, including diagnostics and costs, remains byte-identical.
    assert sha256(output/'checkpoint.json')==report['checkpoint_sha256']
    report.update(source_run=str(source),recovered_run=str(output),
        writable_destination=writable_probe(output),writable_cache=writable_probe(cache_dir or ROOT/'data/processed/.llm_cache'))
    rows=[]
    for source_row in frame.to_dict('records'):
        result=checkpoint['results'].get(source_row['advertisement_id'])
        if result is None:result=dict(ai_status='not_attempted',ai_review_required=True,ai_human_verified=False)
        rows.append({**source_row,**dict.fromkeys(AI_FIELDS),**result,'ai_source_run':str(output),
                     'ai_recovery_origin':str(source)})
    write_results(output,rows)
    resilient_atomic_json(output/'recovery_manifest.json',report)
    resilient_atomic_json(output/'run_summary.json',dict(stage='step3e',status='recovered_not_executed',
        **operational_summary(rows,True),**{k:report[k] for k in ['accounted_cost_usd','provider_reported_usage_estimated_cost_usd',
        'unreported_usage_reserved_usd','remaining_budget_usd','new_paid_requests']},
        provenance='Checkpoint authoritative; original stale summary archived under source_artifacts'))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--budget-usd',type=float,default=2)
    args=parser.parse_args();report=recover(args.source,args.output,args.budget_usd)
    print(json.dumps({k:report[k] for k in ['recovered_run','accepted','rejected','remaining_unrecorded',
        'potential_uncached_ads','accounted_cost_usd','remaining_budget_usd','writable_destination','new_paid_requests']},indent=2))


if __name__=='__main__':main()
