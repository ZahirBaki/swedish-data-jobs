"""Prepare a new failed-only pilot; never mutate the parent run or call an API."""
import argparse
import json
from pathlib import Path

from src.llm_classifier import AI_FIELDS, PROMPT_VERSION, SCHEMA, VALIDATION_VERSION, atomic_json, stable_hash, validate_prediction
from src.validate_classification import read_csv, sha256


def prepare_retry(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Retry preparation output must be a new folder')
    plan = json.loads((source/'plan.json').read_text(encoding='utf-8'))
    summary = json.loads((source/'run_summary.json').read_text(encoding='utf-8'))
    checkpoint = json.loads((source/'checkpoint.json').read_text(encoding='utf-8'))
    if sha256(source/'pilot_input.csv') != plan['pilot_input_sha256']:
        raise ValueError('Parent frozen input was modified')
    frozen = read_csv(source/'pilot_input.csv')
    if frozen.advertisement_id.duplicated().any() or set(AI_FIELDS) & set(frozen):
        raise ValueError('Parent input must have unique original records')
    lines = (source/'pilot_results.jsonl').read_text(encoding='utf-8').splitlines(keepends=True)
    results = [json.loads(line) for line in lines]
    ids = [row['advertisement_id'] for row in results]
    if len(set(ids)) != len(ids) or set(ids) != set(frozen.advertisement_id):
        raise ValueError('Parent results must cover each frozen input exactly once')
    original = frozen.set_index('advertisement_id')
    successes, failed = [], []
    for row in results:
        identifier = row['advertisement_id']
        record = original.loc[identifier]
        for name in ['job_title', 'description_text', 'relevance', 'primary_category']:
            if row[name] != record[name]:
                raise ValueError('Parent results do not preserve original fields')
        if row.get('ai_status') in {'completed','cache_hit'}:
            # The evidence/schema guardrails are unchanged across 1.1 and 1.2.
            if plan['identity']['schema_sha256'] != stable_hash(SCHEMA):
                raise ValueError('Parent schema differs; explicitly review migration before preserving results')
            validate_prediction({name:row[name] for name in AI_FIELDS},record.job_title,record.description_text,
                allow_formatting=plan['identity'].get('validation_version')==VALIDATION_VERSION)
            if row.get('ai_human_verified') is not False:
                raise ValueError('AI result must not claim human verification')
            saved = checkpoint['results'].get(identifier, {})
            if any(saved.get(name) != row.get(name) for name in AI_FIELDS+['ai_cache_key','ai_prompt_version']):
                raise ValueError('Parent checkpoint and result disagree')
            successes.append(identifier)
        elif row.get('ai_status') == 'invalid_model_response':
            failed.append(identifier)
        else:
            raise ValueError('Only known invalid_model_response failures may be retried; inspect other outcomes first')
    if not failed:
        raise ValueError('No invalid model responses to retry')
    output.mkdir(parents=True,exist_ok=False)
    retry_input = frozen.loc[frozen.advertisement_id.isin(failed)].copy()
    retry_input['retry_parent_run'] = str(source)
    retry_input['retry_parent_prompt_version'] = plan['identity']['prompt_version']
    retry_input.to_csv(output/'retry_failed_input.csv',index=False,encoding='utf-8-sig')
    # Copy original JSONL lines, retaining per-record model/prompt/cache provenance.
    (output/'preserved_successful_results.jsonl').write_text(
        ''.join(line for line,row in zip(lines,results) if row['advertisement_id'] in successes),encoding='utf-8')
    read_csv(source/'pilot_results.csv').loc[lambda f:f.advertisement_id.isin(successes)].to_csv(
        output/'preserved_successful_results.csv',index=False,encoding='utf-8-sig')
    manifest = dict(source_run=str(source),successful_ids=successes,retry_ids=failed,
        parent_prompt_version=plan['identity']['prompt_version'],retry_prompt_version=PROMPT_VERSION,
        parent_validation_version=plan['identity'].get('validation_version','exact-legacy'),
        retry_validation_version=VALIDATION_VERSION,
        parent_schema_sha256=plan['identity']['schema_sha256'],retry_schema_sha256=stable_hash(SCHEMA),
        parent_config=plan['identity']['config'],
        same_prompt_schema=(plan['identity'].get('validation_version')==VALIDATION_VERSION and
                            plan['identity']['prompt_version']==PROMPT_VERSION and
                            plan['identity']['schema_sha256']==stable_hash(SCHEMA)),
        source_hashes={name:sha256(source/name) for name in
            ['pilot_input.csv','pilot_results.csv','pilot_results.jsonl','checkpoint.json','plan.json']},
        source_usage_estimated_cost_usd=summary.get('provider_reported_usage_estimated_cost_usd'),
        source_unreported_usage_reserved_usd=summary.get('unreported_usage_reserved_usd'),
        new_paid_requests=0,output_scope='Failed-only diagnostic sample; not a representative population sample',
        merge_policy='Keep parent successes and retry results separate. Any later union must retain per-row prompt/model/schema and report mixed-version provenance; do not label all rows with the new prompt.')
    atomic_json(output/'retry_manifest.json',manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    manifest = prepare_retry(args.run,args.output)
    print(json.dumps(dict(output=str(args.output),preserved=len(manifest['successful_ids']),
                         retry_only=len(manifest['retry_ids']),new_paid_requests=0),indent=2))


if __name__ == '__main__':
    main()
