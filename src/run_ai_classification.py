"""Step 3D: plan or explicitly execute a capped 30-ad pilot, never the full corpus."""
import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.classification_policy import policy_flags
from src.classify_jobs_v2 import CATEGORIES, RELEVANCES
from src.llm_classifier import (AI_FIELDS, SCHEMA, PROMPT_VERSION, VALIDATION_VERSION, BudgetExceeded, CredentialMissing, Config,
    LLMClassifier, atomic_json, file_lock, stable_hash)
from src.validate_classification import HUMAN_FIELDS, ROOT, latest_step3b, read_csv, sha256


def find_sample():
    step3b = latest_step3b()
    digest = sha256(step3b/'jobs_classified_v2.csv')
    workspaces = []
    for path in (ROOT/'data'/'processed').glob('step3c-*/step3c_summary.json'):
        if json.loads(path.read_text(encoding='utf-8'))['source_sha256'] == digest:
            if (path.parent/'manual_review_representative.csv').is_file():
                workspaces.append(path.parent)
    return (max(workspaces,key=lambda p:p.name) if workspaces else step3b)/'manual_review_representative.csv'


def select_pilot(sample, max_ads=30, seed=47):
    if not 1 <= max_ads <= 30:
        raise ValueError('Step 3D is pilot-only: --max-ads must be 1..30; full-corpus execution is disabled')
    required = {'advertisement_id', 'job_title', 'description_text', 'relevance', 'primary_category'}
    if missing := required-set(sample):
        raise ValueError(f'Missing rule-based sample columns: {sorted(missing)}')
    if len(sample) > 200 or len(sample) < max_ads:
        raise ValueError('Select the existing representative sample (at most 200 rows), not the full corpus')
    if sample.advertisement_id.eq('').any() or sample.advertisement_id.duplicated().any():
        raise ValueError('Sample IDs must be nonempty and unique')
    if set(sample.relevance)-set(RELEVANCES):
        raise ValueError('Invalid rule-based relevance values')
    if set(AI_FIELDS) & set(sample):
        raise ValueError('Input must be original rule-based records, not AI suggestions/results')
    shuffled = sample.sample(frac=1,random_state=seed)
    groups = [group for _,group in shuffled.groupby('relevance',sort=True)]
    indices,depth = [],0
    while len(indices) < max_ads:
        for group in groups:
            if depth < len(group) and len(indices) < max_ads:
                indices.append(group.index[depth])
        depth += 1
    selected = sample.loc[indices].copy()
    selected['ai_pilot_sampling_design'] = 'balanced_rule_relevance_pilot_not_population_representative'
    selected['ai_pilot_seed'] = seed
    return selected


def discover_suggestions():
    name = 'manual_review_representative_ai_suggestions.csv'
    paths = ([ROOT/name] if (ROOT/name).is_file() else [])+list((ROOT/'data').rglob(name))
    if len(paths) > 1:
        raise ValueError('Multiple suggestions files found; select one with --suggestions')
    return paths[0] if paths else None


def load_suggestions(path, id_column=None, relevance_column=None, category_column=None):
    if path is None:
        return None
    frame = read_csv(path)
    def choose(explicit, candidates, purpose, optional=False):
        if explicit:
            if explicit not in frame:
                raise ValueError(f'Suggestions column missing: {explicit}')
            return explicit
        matches = [col for col in candidates if col in frame]
        if len(matches) == 1:
            return matches[0]
        if not matches and optional:
            return None
        raise ValueError(f'Inspect suggestions headers {list(frame)} and configure the {purpose} column explicitly')
    ids = choose(id_column,['advertisement_id','id'],'ID')
    relevance = choose(relevance_column,['ai_relevance','suggested_relevance','ai_suggested_relevance'],'relevance')
    category = choose(category_column,['ai_primary_category','suggested_primary_category','ai_suggested_primary_category'],
                      'category',optional=True)
    if frame[ids].eq('').any() or frame[ids].duplicated().any():
        raise ValueError('Provisional reference IDs must be nonempty and unique')
    if set(frame[relevance])-set(RELEVANCES)-{''}:
        raise ValueError('Unknown provisional relevance label')
    if category and set(frame[category])-set(CATEGORIES)-{''}:
        raise ValueError('Unknown provisional category; map it explicitly to the policy first')
    # Intentionally discard ALL human/reviewer/verified columns in this AI reference.
    result = pd.DataFrame(dict(advertisement_id=frame[ids],
                              provisional_ai_reference_relevance=frame[relevance]))
    result['provisional_ai_reference_category'] = frame[category] if category else ''
    result['provisional_reference_provenance'] = 'AI suggestions; never independently verified ground truth'
    return result


def write_results(output, rows):
    serial = pd.DataFrame(rows)
    if 'ai_secondary_categories' in serial:
        serial['ai_secondary_categories'] = [json.dumps(row.get('ai_secondary_categories'),ensure_ascii=False) for row in rows]
    temporary = output/'pilot_results.csv.tmp'
    serial.to_csv(temporary,index=False,encoding='utf-8-sig')
    os.replace(temporary,output/'pilot_results.csv')
    temporary = output/'pilot_results.jsonl.tmp'
    with temporary.open('w',encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    os.replace(temporary,output/'pilot_results.jsonl')


def compare_rows(rows):
    frame = pd.DataFrame(rows)
    completed = frame.loc[frame.ai_status.isin(['completed','cache_hit'])].copy()
    if completed.empty:
        return dict(model_classified=0,rule_ai_relevance_disagreements=0,
                    rule_ai_category_disagreements=0,uncertain_resolved=0,
                    ai_relevance_counts={},ai_category_counts={},human_accuracy_status='Not calculated')
    return dict(model_classified=len(completed),
        rule_ai_relevance_disagreements=int(completed.relevance.ne(completed.ai_relevance).sum()),
        rule_ai_category_disagreements=int(completed.primary_category.fillna('').ne(completed.ai_primary_category.fillna('')).sum()),
        uncertain_resolved=int((completed.relevance.eq('uncertain') & completed.ai_relevance.isin(['relevant','irrelevant'])).sum()),
        ai_relevance_counts={k:int(v) for k,v in completed.ai_relevance.value_counts().items()},
        ai_category_counts={k:int(v) for k,v in completed.ai_primary_category.fillna('(null)').value_counts().items()},
        review_required=int(completed.ai_review_required.sum()),
        human_accuracy_status='No automatic accuracy/precision/recall/F1 against rule-based or AI reference labels')


def run(config, budget_usd, execute=False, max_ads=30, seed=47, sample_path=None,
        suggestions_path=None, suggestion_columns=None, output=None, resume=None,
        cache_dir=None, policy_path=None, retry_unknown=False, session=None, sleep=None):
    if output and resume:
        raise ValueError('Use either a new --output folder or --resume, not both')
    if not 1 <= max_ads <= 30:
        raise ValueError('Only a 1..30-ad pilot is authorized by this runner')
    policy_path = Path(policy_path) if policy_path else ROOT/'docs'/'classification_policy.md'
    policy = policy_path.read_text(encoding='utf-8')
    old_plan = json.loads((Path(resume)/'plan.json').read_text(encoding='utf-8')) if resume else None
    sample_path = Path(sample_path) if sample_path else Path(old_plan['sample_path']) if old_plan else find_sample()
    if 'ai_suggestions' in sample_path.name.casefold():
        raise ValueError('AI suggestions must be supplied with --suggestions, never as the source/human review sample')
    sample = read_csv(sample_path)
    pilot = select_pilot(sample,max_ads,seed)
    suggestions_path = Path(suggestions_path) if suggestions_path else (
        Path(old_plan['suggestions_path']) if old_plan and old_plan.get('suggestions_path') else discover_suggestions())
    if old_plan and not any((suggestion_columns or {}).values()):
        suggestion_columns = old_plan['identity'].get('suggestion_columns', {})
    suggestion_columns = {key:value for key,value in (suggestion_columns or {}).items() if value is not None}
    references = load_suggestions(suggestions_path,**suggestion_columns)
    if references is not None:
        pilot = pilot.merge(references,on='advertisement_id',how='left',validate='one_to_one').fillna('')
    policy_hash, sample_hash = sha256(policy_path), sha256(sample_path)
    identity = dict(config=asdict(config),sample_sha256=sample_hash,policy_sha256=policy_hash,
        prompt_version=PROMPT_VERSION,validation_version=VALIDATION_VERSION,
        schema_sha256=stable_hash(SCHEMA),pilot_ids=pilot.advertisement_id.tolist(),
        max_ads=max_ads,seed=seed,suggestions_sha256=sha256(suggestions_path) if suggestions_path else None,
        suggestion_columns=suggestion_columns)
    output = Path(resume) if resume else Path(output) if output else ROOT/'data'/'processed'/(
        'step3d-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if resume:
        if old_plan['identity'] != identity:
            raise ValueError('Resume identity changed (input, policy, model, prices, limits, seed or reference); use a new run')
        if sha256(output/'pilot_input.csv') != old_plan['pilot_input_sha256']:
            raise ValueError('Frozen pilot input was modified')
    else:
        output.mkdir(parents=True,exist_ok=False)
        pilot.to_csv(output/'pilot_input.csv',index=False,encoding='utf-8-sig')
        human_review = pilot.copy(deep=True)
        for name in HUMAN_FIELDS:
            if name not in human_review:
                human_review[name] = ''
        human_review.to_csv(output/'human_review_pilot.csv',index=False,encoding='utf-8-sig')
        atomic_json(output/'plan.json',dict(identity=identity,sample_path=str(sample_path.resolve()),
            suggestions_path=str(suggestions_path.resolve()) if suggestions_path else None,
            pilot_input_sha256=sha256(output/'pilot_input.csv'),initial_budget_usd=budget_usd))
        (output/'classification_policy_snapshot.md').write_text(policy,encoding='utf-8')
        atomic_json(output/'output_schema.json',SCHEMA)
        for name in ['llm_classifier.py','run_ai_classification.py']:
            (output/name).write_bytes((ROOT/'src'/name).read_bytes())
    with file_lock(output/'run.lock'):
        kwargs = dict(session=session) if session is not None else {}
        if sleep is not None:
            kwargs['sleep'] = sleep
        classifier = LLMClassifier(config,policy,cache_dir or ROOT/'data'/'processed'/'.llm_cache',
            output/'checkpoint.json',budget_usd,execute=execute,retry_unknown=retry_unknown,**kwargs)
        (output/'system_prompt.txt').write_text(classifier.system_prompt,encoding='utf-8') if not resume else None
        estimates = []
        for row in pilot.itertuples():
            estimate = classifier.estimate(row.job_title,row.description_text)
            cached = classifier.cached(row.job_title,row.description_text)
            needs_request = bool(row.description_text.strip()) and not cached and not estimate['input_limit_exceeded']
            estimates.append(dict(advertisement_id=row.advertisement_id,**estimate,
                request_needed=needs_request,cache_available=bool(cached)))
        estimate_frame = pd.DataFrame(estimates)
        estimate_frame.to_csv(output/'token_cost_plan.csv',index=False)
        pending_estimates = [item for item in estimates if item['request_needed']]
        forecast = dict(expected_input_tokens=sum(item['expected_input_tokens'] for item in pending_estimates),
            reserved_input_tokens=sum(item['reserved_input_tokens'] for item in pending_estimates),
            expected_output_tokens=sum(item['expected_output_tokens'] for item in pending_estimates),
            reserved_output_tokens=sum(item['reserved_output_tokens'] for item in pending_estimates),
            expected_first_pass_usd=sum(item['expected_cost_usd'] for item in pending_estimates),
            conservative_first_pass_usd=sum(item['reserved_cost_per_attempt_usd'] for item in pending_estimates),
            conservative_all_attempts_usd=sum(item['reserved_cost_per_attempt_usd'] for item in pending_estimates)*(config.max_retries+1),
            uncached_requests_needed=len(pending_estimates),
            token_estimation='Expected chars/3.5+256; reservation UTF-8 payload bytes+2048 and capped output. No truncation or paid counting requests.')
        atomic_json(output/'cost_forecast.json',forecast)
        print(json.dumps(dict(mode='execute' if execute else 'dry_run',provider=config.provider,model=config.model,
            pilot_advertisements=len(pilot),rule_relevance_strata=pilot.relevance.value_counts().to_dict(),
            budget_usd=budget_usd,prior_accounted_cost_usd=classifier.spent,forecast=forecast),indent=2))
        ledger_at_start = len(classifier.state['ledger'])
        rows,status = [],'completed' if execute else 'dry_run_only'
        for row in pilot.to_dict('records'):
            if status == 'authentication_stopped':
                rows.append({**row,**dict.fromkeys(AI_FIELDS),
                    'ai_status':'not_attempted_authentication_failure',
                    'ai_review_required':True,'ai_human_verified':False,
                    'ai_provider':config.provider,'ai_model':config.model,
                    'ai_prompt_version':PROMPT_VERSION,'ai_role_scope':'not_classified'})
                continue
            try:
                prediction = classifier.classify(row['advertisement_id'],row['job_title'],row['description_text'])
            except CredentialMissing:
                prediction = dict(ai_status='credential_required',ai_review_required=True,ai_human_verified=False)
                status = 'credential_required'
            except BudgetExceeded:
                status = 'budget_stopped'
                prediction = dict(ai_status='budget_stopped',ai_review_required=True,ai_human_verified=False)
            if status == 'budget_stopped' and prediction['ai_status'] == 'budget_stopped':
                # No more attempts in this invocation; preserve all remaining source rows.
                for remainder in pilot.to_dict('records')[len(rows):]:
                    result = {**remainder,**dict.fromkeys(AI_FIELDS),'ai_status':'budget_stopped',
                        'ai_review_required':True,'ai_human_verified':False,'ai_provider':config.provider,
                        'ai_model':config.model,'ai_prompt_version':PROMPT_VERSION,'ai_role_scope':'not_classified'}
                    rows.append(result)
                break
            result = {**row,**dict.fromkeys(AI_FIELDS),**prediction}
            if result.get('ai_review_required') is None:
                result['ai_review_required'] = True  # Unclassified, never a completed AI decision.
            result.setdefault('ai_provider',config.provider)
            result.setdefault('ai_model',config.model)
            result.setdefault('ai_prompt_version',PROMPT_VERSION)
            if result.get('ai_relevance') in RELEVANCES:
                flags = policy_flags(row['job_title'],result['ai_relevance'],result['ai_primary_category'])
                result.update({'ai_'+key:value for key,value in flags.items()})
            else:
                result['ai_role_scope'] = 'not_classified'
            rows.append(result)
            if prediction['ai_status'] in {'http_401', 'http_403'}:
                status = 'authentication_stopped'
            write_results(output,rows)  # Regenerable export; paid completions also live in cache/checkpoint.
        write_results(output,rows)
        if execute and status == 'completed' and any(row['ai_status'] not in {
                'completed','cache_hit','missing_description'} for row in rows):
            status = 'partial_requires_inspection'
        comparison = compare_rows(rows)
        summary = dict(mode='execute' if execute else 'dry_run',status=status,
            pilot_size=len(pilot),source_representative_size=len(sample),
            source_path=str(sample_path.resolve()),source_sha256=sample_hash,policy_sha256=policy_hash,
            model=config.model,provider=config.provider,prompt_version=PROMPT_VERSION,validation_version=VALIDATION_VERSION,
            ai_human_verified_assigned=False,api_credentials_logged=False,
            pricing_usd_per_million=dict(input=config.input_usd_per_million,output=config.output_usd_per_million),
            budget_usd=budget_usd,accounted_cost_usd=classifier.spent,
            **classifier.cost_breakdown(),
            attempts_reserved_total=len(classifier.state['ledger']),
            attempts_reserved_this_invocation=len(classifier.state['ledger'])-ledger_at_start,
            reported_input_tokens=sum(item['prompt_tokens'] or 0 for item in classifier.state['ledger']),
            reported_output_tokens=sum(item['completion_tokens'] or 0 for item in classifier.state['ledger']),
            unknown_or_unreported_cost_attempts=sum(item['usage_basis']!='provider_reported_tokens' for item in classifier.state['ledger']),
            forecast=forecast,comparison=comparison,
            ai_status_counts=dict(pd.Series([row['ai_status'] for row in rows]).value_counts().items()),
            suggestions_available=references is not None,suggestions_provenance='Provisional AI reference, never human ground truth',
            sample_scope='Balanced pilot from the existing representative sample, not representative of population prevalence',
            cost_scope='Estimates at configured prices; missing usage/timeouts retain full reservation, including retries. Provider invoice/fees may differ.',
            resume_arguments=['python','-m','src.run_ai_classification','--resume',str(output.resolve()),
                '--provider',config.provider,'--model',config.model,'--base-url',config.base_url,
                '--api-key-env',config.api_key_env,'--input-usd-per-million',str(config.input_usd_per_million),
                '--output-usd-per-million',str(config.output_usd_per_million),'--budget-usd',str(budget_usd),
                '--max-ads',str(max_ads),'--seed',str(seed),'--max-output-tokens',str(config.max_output_tokens),
                '--max-input-tokens',str(config.max_input_tokens),'--timeout',str(config.timeout),
                '--max-retries',str(config.max_retries),'--min-interval',str(config.min_interval),
                '--completion-token-field',config.completion_token_field,
                *(['--cache-dir',str(cache_dir)] if cache_dir else []),
                *[part for key,value in suggestion_columns.items() for part in ['--suggestion-'+key.replace('_','-'),value]],
                '--execute'],
            inputs_unchanged=sha256(sample_path)==sample_hash and sha256(policy_path)==policy_hash)
        # pandas counts may be numpy integers: normalize before strict JSON output.
        summary['ai_status_counts'] = {k:int(v) for k,v in summary['ai_status_counts'].items()}
        atomic_json(output/'run_summary.json',summary)
        report = f'''# Step 3D pilot

Mode: {summary['mode']}; status: {status}. Provider/model: {config.provider} / {config.model}. Prompt: {PROMPT_VERSION}.
Pilot: {len(pilot)} of {len(sample)} representative-sample advertisements, balanced by rule relevance (seed {seed}). Full original text and classifications remain in pilot_input.csv and pilot_results.csv/jsonl.

Expected first-pass cost: ${forecast['expected_first_pass_usd']:.6f}. Conservative first-pass reservation: ${forecast['conservative_first_pass_usd']:.6f}. All-attempt forecast: ${forecast['conservative_all_attempts_usd']:.6f}. Run budget: ${budget_usd:.4f}. Accounted cost so far: ${classifier.spent:.6f}. Reservations are durably checked before every attempt. Actual provider token usage is recorded when supplied; otherwise conservative reservations remain charged to the app budget.

Provider-reported token usage priced at configured rates: ${summary['provider_reported_usage_estimated_cost_usd']:.6f}. Budget held for attempts without reported usage: ${summary['unreported_usage_reserved_usd']:.6f}. These holds are not evidence of billed token usage; the provider invoice is authoritative. HTTP 401/403 stops further advertisements. Sanitized API error metadata is stored on checkpoint ledger entries; raw responses and credentials are never logged.

API attempts reserved this invocation: {summary['attempts_reserved_this_invocation']}. Completed model classifications: {comparison['model_classified']}. Rule/AI relevance disagreements: {comparison['rule_ai_relevance_disagreements']}. Previously uncertain rows resolved by the model: {comparison['uncertain_resolved']}. These are comparisons, not validated correctness.

No human metrics are calculated automatically. Optional AI suggestions are comparison-only and never supplied as ground truth or included in the classification prompt. Human fields from the original sample remain separate; no AI-generated label is marked human-verified. Use notebooks/04_evaluate_ai_classifier.ipynb for separate relevance/category evaluation after explicit verified human labels exist.

Dry runs do not call any LLM API. Add --execute only after checking the forecast and configuring the named credential environment variable. Use --resume on this folder with identical model, prices, limits, policy, seed and sample to continue. The budget may be explicitly increased on resume, but prior costs still count. Successful cached/checkpointed classifications are not re-requested. Timeouts or crashes with unknown outcomes block automatic retries; --retry-unknown explicitly accepts possible duplicate billing and retains prior reservations. Other terminal errors are preserved; a fresh pilot folder can retry failures while reusing successful shared cache entries. Do not run concurrent writers against one workspace.

The runner refuses more than 30 advertisements and refuses corpus-sized input. No full-market run is implemented or authorized. No source dataset is overwritten, deduplicated, or reclassified in place. Model confidence is qualitative and uncalibrated. Research/statistics/management and title-derived position scopes remain separately identifiable.
'''
        (output/'pilot_report.md').write_text(report,encoding='utf-8')
        print(json.dumps(dict(status=status,output=str(output.resolve()),accounted_cost_usd=classifier.spent,
            new_attempts_reserved=summary['attempts_reserved_this_invocation'],comparison=comparison),indent=2))
        if not summary['inputs_unchanged']:
            raise AssertionError('Source sample/policy changed during execution')
        return output,summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider',choices=['openai','openai-compatible'],default=os.getenv('LLM_PROVIDER','openai'))
    parser.add_argument('--model',default=os.getenv('LLM_MODEL'))
    parser.add_argument('--base-url',default=os.getenv('LLM_BASE_URL'))
    parser.add_argument('--api-key-env')
    parser.add_argument('--input-usd-per-million',type=float,default=os.getenv('LLM_INPUT_USD_PER_MILLION'))
    parser.add_argument('--output-usd-per-million',type=float,default=os.getenv('LLM_OUTPUT_USD_PER_MILLION'))
    parser.add_argument('--budget-usd',type=float,default=os.getenv('LLM_BUDGET_USD'))
    parser.add_argument('--max-ads',type=int,default=30)
    parser.add_argument('--seed',type=int,default=47)
    parser.add_argument('--max-output-tokens',type=int,default=1024)
    parser.add_argument('--max-input-tokens',type=int,default=64000)
    parser.add_argument('--timeout',type=float,default=60)
    parser.add_argument('--max-retries',type=int,default=2)
    parser.add_argument('--min-interval',type=float,default=.5)
    parser.add_argument('--completion-token-field',choices=['max_completion_tokens','max_tokens'])
    parser.add_argument('--sample',type=Path)
    parser.add_argument('--suggestions',type=Path)
    parser.add_argument('--suggestion-id-column')
    parser.add_argument('--suggestion-relevance-column')
    parser.add_argument('--suggestion-category-column')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--cache-dir',type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--execute',action='store_true')
    modes.add_argument('--dry-run',action='store_true',help='Default; explicitly forbids paid/network requests')
    parser.add_argument('--retry-unknown',action='store_true')
    args = parser.parse_args()
    if args.model is None or args.input_usd_per_million is None or args.output_usd_per_million is None or args.budget_usd is None:
        parser.error('Configure model, both USD-per-million prices and budget explicitly with flags or LLM_* environment variables')
    compatible = args.provider == 'openai-compatible'
    if compatible and not args.base_url:
        parser.error('OpenAI-compatible provider requires an explicit --base-url or LLM_BASE_URL')
    config = Config(model=args.model,input_usd_per_million=args.input_usd_per_million,
        output_usd_per_million=args.output_usd_per_million,provider=args.provider,
        base_url=args.base_url or 'https://api.openai.com/v1',api_key_env=args.api_key_env or ('LLM_API_KEY' if compatible else 'OPENAI_API_KEY'),
        max_output_tokens=args.max_output_tokens,max_input_tokens=args.max_input_tokens,timeout=args.timeout,
        max_retries=args.max_retries,min_interval=args.min_interval,
        completion_token_field=args.completion_token_field or ('max_tokens' if compatible else 'max_completion_tokens'))
    _,summary = run(config,args.budget_usd,execute=args.execute,max_ads=args.max_ads,seed=args.seed,
        sample_path=args.sample,suggestions_path=args.suggestions,
        suggestion_columns=dict(id_column=args.suggestion_id_column,relevance_column=args.suggestion_relevance_column,
                                category_column=args.suggestion_category_column),
        output=args.output,resume=args.resume,cache_dir=args.cache_dir,retry_unknown=args.retry_unknown)
    return 0 if summary['status'] in {'completed','dry_run_only'} else 2


if __name__=='__main__':
    raise SystemExit(main())
