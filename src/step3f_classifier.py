"""Step 3F: isolated versioned span-selection schema and proven budgeted transport.

The request flow is intentionally isolated from historical engines: old code hashes,
prompts, schemas and cache identities remain unchanged. No calls without execute.
"""
import json
import os
import time
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

from src.classify_jobs_v2 import CATEGORIES
from src.llm_classifier import (LLMClassifier, ClassificationError, BudgetExceeded,
    CredentialMissing, Config, stable_hash, strict_json, validate_schema, file_lock)
from src.recover_large_pilot import resilient_atomic_json as atomic_json, writable_probe
from src.validate_classification import ROOT

PROMPT_VERSION = 'step3f-1.0'
POLICY_VERSION = 'step3f-v1'
VALIDATION_VERSION = 'indexed-spans-1'
POLICY_PATH = ROOT/'docs/classification_policy_step3f_v1.md'
PROMPT_PATH = ROOT/'prompts/step3f-v1.txt'
SCOPES = ['core_industry','data_ai_governance','academic_ai_ml_research',
          'adjacent_borderline','unrelated','unresolved']
OPPORTUNITIES = ['internship','thesis','doctoral','postdoctoral','student']
SCHEMA = dict(type='object',additionalProperties=False,properties={
    'ai_relevance':dict(type='string',enum=['relevant','irrelevant','uncertain']),
    'ai_primary_category':dict(type=['string','null'],enum=CATEGORIES+[None]),
    'ai_secondary_categories':dict(type='array',items=dict(type='string',enum=CATEGORIES),maxItems=5),
    'ai_reasoning':dict(type='string',minLength=1,maxLength=1000),
    'ai_evidence_span_ids':dict(type='array',items=dict(type='string'),maxItems=3),
    'ai_confidence':dict(type='string',enum=['high','medium','low']),
    'ai_review_required':dict(type='boolean'),
    'ai_scope':dict(type='string',enum=SCOPES),
    'ai_opportunity_types':dict(type='array',items=dict(type='string',enum=OPPORTUNITIES),maxItems=5),
    'ai_multi_role':dict(type='boolean')})
AI_FIELDS = list(SCHEMA['properties'])
SCHEMA['required'] = AI_FIELDS


def source_spans(title, description, width=240):
    """Full coverage, no normalization; offsets count Python Unicode characters."""
    if width < 1:
        raise ValueError('Span width must be positive')
    spans=[]
    for prefix,field,text in [('T','job_title',title),('D','description_text',description)]:
        for index,start in enumerate(range(0,len(text),width)):
            spans.append(dict(id=f'{prefix}{index:05d}',source_field=field,start=start,
                              end=min(start+width,len(text)),text=text[start:start+width]))
    return spans


def span_match(ids,title,description,allow_formatting=False):
    spans={s['id']:s for s in source_spans(title,description)}
    if len(ids)!=len(set(ids)):
        raise ClassificationError('Duplicate evidence span ID')
    if any(i not in spans for i in ids):
        raise ClassificationError('Unknown evidence span ID; source-grounded evidence required')
    selected=[spans[i] for i in ids]
    for span in selected:
        original=title if span['source_field']=='job_title' else description
        if original[span['start']:span['end']]!=span['text']:
            raise ClassificationError('Source span offsets do not match original text')
    return dict(method='source_span_selection',spans=selected)


def validate_span_prediction(value,title,description,allow_formatting=False):
    validate_schema(value,SCHEMA)
    result=deepcopy(value)
    if not result['ai_reasoning'].strip():
        raise ClassificationError('A nonblank decision explanation is required')
    relevance=result['ai_relevance']; category=result['ai_primary_category']
    secondary=result['ai_secondary_categories']; scope=result['ai_scope']
    if len(secondary)!=len(set(secondary)) or category in secondary:
        raise ClassificationError('Primary/secondary categories must be distinct')
    if len(result['ai_opportunity_types'])!=len(set(result['ai_opportunity_types'])):
        raise ClassificationError('Duplicate opportunity type')
    if relevance!='relevant' and (category is not None or secondary):
        raise ClassificationError('Non-relevant decisions must abstain from categories')
    if relevance=='irrelevant' and scope!='unrelated':
        raise ClassificationError('Irrelevant decisions require unrelated scope')
    if relevance=='uncertain' and scope not in {'unresolved','adjacent_borderline'}:
        raise ClassificationError('Uncertain decisions require unresolved or borderline scope')
    if relevance=='relevant' and scope=='unrelated':
        raise ClassificationError('Relevant decisions cannot have unrelated scope')
    if relevance=='relevant' and scope=='data_ai_governance' and category!='Data Governance / Data Quality':
        raise ClassificationError('Governance scope requires governance category')
    if relevance=='relevant' and scope=='academic_ai_ml_research' and category!='AI / ML Research':
        raise ClassificationError('Academic AI research scope requires AI / ML Research category')
    if relevance=='relevant' and category in {'Statistician','Data / AI Management'} and scope!='adjacent_borderline':
        raise ClassificationError('Statistics/management require separately tracked adjacent scope')
    ids=result['ai_evidence_span_ids']
    evidence=span_match(ids,title,description)
    if relevance!='uncertain' and not ids:
        raise ClassificationError('Resolved relevance requires a supporting source span')
    # Derived excerpts are auditable selections, never replacements for rejected quotes.
    result['ai_evidence']=[s['text'] for s in evidence['spans']]
    if (relevance=='uncertain' or category is None or scope!='core_industry'
            or result['ai_confidence']!='high' or result['ai_multi_role']
            or result['ai_opportunity_types'] or all(i.startswith('T') for i in ids)):
        result['ai_review_required']=True
    return result


class Step3FClassifier(LLMClassifier):
    def __init__(self,*args,prompt_text=None,**kwargs):
        super().__init__(*args,**kwargs)
        if self.config.max_retries != 0 or self.retry_unknown:
            raise ValueError('Step 3F does not automatically retry failed or unknown requests')
        self.system_prompt=((prompt_text if prompt_text is not None else PROMPT_PATH.read_text(encoding='utf-8'))
            +'\nCANONICAL CATEGORIES:\n'+json.dumps(CATEGORIES,ensure_ascii=False)
            +'\nPOLICY:\n'+self.policy_text)
        if self.execute:
            writable_probe(self.checkpoint_path.parent)
            writable_probe(self.cache_dir)

    def save(self):
        atomic_json(self.checkpoint_path,self.state)

    def payload(self,title,description):
        payload=dict(model=self.config.model,messages=[dict(role='system',content=self.system_prompt),
            dict(role='user',content=json.dumps(dict(source_spans=source_spans(title,description)),ensure_ascii=False))],
            response_format=dict(type='json_schema',json_schema=dict(name='step3f_classification',strict=True,schema=SCHEMA)))
        payload[self.config.completion_token_field]=self.config.max_output_tokens
        if self.config.provider=='openai':payload['store']=False
        return payload

    def cache_key(self,title,description):
        return stable_hash(dict(stage='step3f',prompt_version=PROMPT_VERSION,policy_version=POLICY_VERSION,
            validation_version=VALIDATION_VERSION,policy_hash=self.policy_hash,
            provider=self.config.provider,endpoint=self.config.base_url.rstrip('/'),payload=self.payload(title,description)))

    def cached(self,title,description):
        key=self.cache_key(title,description); path=self.cache_dir/(key+'.json')
        if not path.exists():return None
        entry=json.loads(path.read_text(encoding='utf-8'))
        if entry.get('cache_key')!=key:raise ClassificationError('Cache identity mismatch')
        entry['prediction']=validate_span_prediction({k:entry['prediction'][k] for k in AI_FIELDS},title,description)
        entry['evidence_match']=span_match(entry['prediction']['ai_evidence_span_ids'],title,description)
        return entry

    def metadata(self,key,entry,cache_hit=False):
        result=super().metadata(key,entry,cache_hit)
        result.update(ai_prompt_version=PROMPT_VERSION,ai_policy_version=POLICY_VERSION,
            ai_validation_version=VALIDATION_VERSION,ai_schema_sha256=stable_hash(SCHEMA),
            ai_validation_status='accepted_source_spans',
            classification_status=entry['prediction']['ai_relevance'].capitalize(),
            ai_source_run=str(self.checkpoint_path.parent.resolve()))
        return result

    def failure(self,identifier,key,status):
        result=dict.fromkeys(AI_FIELDS)
        result.update(ai_status=status,ai_review_required=True,ai_cache_key=key,ai_evidence=None,
            ai_provider=self.config.provider,ai_model=self.config.model,ai_prompt_version=PROMPT_VERSION,
            ai_policy_version=POLICY_VERSION,ai_policy_sha256=self.policy_hash,ai_validation_version=VALIDATION_VERSION,
            ai_schema_sha256=stable_hash(SCHEMA),ai_validation_status='rejected_or_not_completed',
            ai_human_verified=False,classification_status='Technical failure / requires retry',
            ai_source_run=str(self.checkpoint_path.parent.resolve()))
        self.state['results'][str(identifier)]=result
        self.save()
        return result

    def classify(self, identifier, title, description):
        if not isinstance(title, str) or not isinstance(description, str):
            raise ValueError('Pass original title/description strings; missing values must be empty strings')
        key = self.cache_key(title, description)
        prior = self.state['results'].get(str(identifier))
        if prior:
            if prior['ai_cache_key'] != key:
                raise ClassificationError('Checkpoint input/prompt changed')
            if prior['ai_status'] in {'completed', 'cache_hit', 'missing_description'}:
                validated = validate_span_prediction({name:prior[name] for name in AI_FIELDS}, title, description,allow_formatting=True)
                prior.update(validated)
                if prior['ai_evidence']:
                    prior.update({'ai_evidence_'+name:value for name,value in
                        span_match(prior['ai_evidence_span_ids'],title,description).items()})
                if prior.get('ai_human_verified') is not False:
                    raise ClassificationError('Checkpoint must not claim human verification')
            if prior['ai_status'] != 'request_outcome_unknown' or not self.retry_unknown:
                return prior
        if not description.strip():
            result = dict(ai_relevance='uncertain', ai_primary_category=None, ai_secondary_categories=[],
                ai_reasoning='Description missing; no LLM request was made.', ai_evidence='',
                ai_confidence='low', ai_review_required=True, ai_status='missing_description',
                ai_cache_key=key, ai_provider=self.config.provider, ai_model=self.config.model,
                ai_prompt_version=PROMPT_VERSION, ai_policy_sha256=self.policy_hash,
                ai_input_tokens=0, ai_output_tokens=0, ai_estimated_cost_usd=0.0, ai_human_verified=False,
                ai_evidence_span_ids=[], ai_scope='unresolved', ai_opportunity_types=[], ai_multi_role=False)
            result.update(ai_policy_version=POLICY_VERSION, ai_validation_version=VALIDATION_VERSION,
                classification_status='Uncertain', ai_validation_status='local_missing_description',
                ai_source_run=str(self.checkpoint_path.parent.resolve()))
            if self.execute:
                self.state['results'][str(identifier)] = result
                self.save()
            return result
        cached = self.cached(title, description)
        if cached:
            # If a process crashed after atomic cache commit, settle its existing
            # reservation from the receipt instead of repeating a completed request.
            for item in self.state['ledger']:
                if item['attempt_id'] == cached.get('attempt_id') and item['status'] == 'reserved_before_request':
                    self.settle(item, dict(usage=cached.get('usage', {})), 'completed_recovered_from_cache')
            result = self.metadata(key, cached, cache_hit=True)
            if self.execute:
                self.state['results'][str(identifier)] = result
                self.save()
            return result
        if not self.execute:
            return dict(ai_status='dry_run', ai_cache_key=key, ai_human_verified=False, **self.estimate(title, description))
        if not self.retry_unknown and any(item['cache_key'] == key and item['status'] in {
                'reserved_before_request', 'request_outcome_unknown'} for item in self.state['ledger']):
            return self.failure(identifier, key, 'request_outcome_unknown')
        estimate = self.estimate(title, description)
        if estimate['input_limit_exceeded']:
            return self.failure(identifier, key, 'input_limit_exceeded')
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise CredentialMissing(f'Missing credential environment variable {self.config.api_key_env}')
        # A second workspace cannot concurrently pay for the same prompt/content.
        with file_lock(self.cache_dir/(key+'.lock')):
            cached = self.cached(title, description)
            if cached:
                result = self.metadata(key, cached, cache_hit=True)
                self.state['results'][str(identifier)] = result
                self.save()
                return result
            pending = self.cache_dir/(key+'.pending.json')
            if pending.exists():
                if not self.retry_unknown:
                    return self.failure(identifier, key, 'request_outcome_unknown')
                pending.unlink()  # Explicit user flag accepts possible duplicate billing.
            for attempt in range(self.config.max_retries+1):
                item = self.reserve(key, str(identifier), estimate)
                atomic_json(pending, dict(attempt_id=item['attempt_id'], cache_key=key,
                                         checkpoint_path=str(self.checkpoint_path.resolve())))
                if self.last_call is not None:
                    self.sleep(max(0, self.config.min_interval-(self.clock()-self.last_call)))
                try:
                    self.last_call = self.clock()
                    response = self.session.post(self.config.base_url.rstrip('/')+'/chat/completions',
                        headers={'Authorization':'Bearer '+api_key, 'Content-Type':'application/json'},
                        json=self.payload(title, description), timeout=(10, self.config.timeout), allow_redirects=False)
                except requests.RequestException as exc:
                    # A timeout may already have been billed. Keep reservation and
                    # global pending marker; no automatic ambiguous-outcome retry.
                    item['status'] = 'request_outcome_unknown'
                    item['network_error_type'] = type(exc).__name__  # No raw exception/URL/headers.
                    self.save()
                    return self.failure(identifier, key, 'request_outcome_unknown')
                try:
                    # JSON API text is UTF-8. Do not let an absent/misleading
                    # HTTP charset select Latin-1 and corrupt Swedish characters.
                    response.encoding = 'utf-8'
                    body = response.json()
                except (ValueError, TypeError):
                    body = {}
                retryable = response.status_code in {408, 409, 429, 500, 502, 503, 504}
                status = 'http_'+str(response.status_code)
                item['http_status'] = response.status_code
                if response.status_code != 200:
                    item['api_error'] = self.error_diagnostic(body, response, api_key)
                valid_entry = None
                if response.status_code == 200:
                    stage = 'response_envelope'
                    try:
                        if not isinstance(body, dict) or not isinstance(body.get('choices'), list):
                            raise ClassificationError('Response must contain a choices array')
                        choice = body['choices'][0]
                        if not isinstance(choice, dict) or not isinstance(choice.get('message'), dict):
                            raise ClassificationError('Response choice must contain a message object')
                        stage = 'completion_status'
                        if choice.get('finish_reason') != 'stop':
                            raise ClassificationError('Response did not finish with stop')
                        if choice['message'].get('refusal'):
                            raise ClassificationError('Model refused the request')
                        stage = 'json_parsing'
                        parsed = strict_json(choice['message']['content'])
                        stage = 'prediction_validation'
                        prediction = validate_span_prediction(parsed, title, description,allow_formatting=True)
                        valid_entry = dict(cache_key=key, prediction=prediction,
                            usage=body.get('usage', {}), model_returned=body.get('model', self.config.model),
                            attempt_id=item['attempt_id'],
                            evidence_match=span_match(prediction['ai_evidence_span_ids'],title,description)
                                if prediction['ai_evidence'] else {})
                        status = 'completed'
                    except (KeyError, IndexError, TypeError, ClassificationError) as exc:
                        status, retryable = 'invalid_model_response', True
                        item['model_response_diagnostic'] = self.invalid_response_diagnostic(
                            body, response, api_key, exc, stage)
                self.settle(item, body, status)
                if valid_entry:
                    valid_entry['usage'] = (dict(prompt_tokens=item['prompt_tokens'], completion_tokens=item['completion_tokens'])
                                            if item['prompt_tokens'] is not None else {})
                    valid_entry.update(accounted_cost_usd=item['accounted_cost_usd'], usage_basis=item['usage_basis'])
                    atomic_json(self.cache_dir/(key+'.json'), valid_entry)
                pending.unlink(missing_ok=True)
                if self.spent > self.budget_usd:
                    # Unexpected provider usage beyond reservation: preserve the
                    # receipt and stop all further requests; never conceal overrun.
                    raise BudgetExceeded('Provider-reported usage exceeded reserved budget; no further requests allowed')
                if valid_entry:
                    result = self.metadata(key, valid_entry)
                    self.state['results'][str(identifier)] = result
                    self.save()
                    return result
                if not retryable or attempt == self.config.max_retries:
                    return self.failure(identifier, key, status)
                delay = 2**attempt
                header = response.headers.get('Retry-After')
                if header:
                    try:
                        delay = max(delay, float(header))
                    except ValueError:
                        try:
                            delay = max(delay, (parsedate_to_datetime(header)-datetime.now(timezone.utc)).total_seconds())
                        except (ValueError, TypeError):
                            pass
                if delay > self.config.max_retry_delay:
                    return self.failure(identifier, key, 'retry_after_requires_later_run')
                self.sleep(delay)
        raise RuntimeError('Unreachable classification path')

