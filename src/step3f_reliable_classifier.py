"""Step 3F 1.1 reliability experiment, isolated from all frozen earlier engines.

Keeps strict original-source evidence and policy semantics; constrains atomic IDs
in the request schema and places a short explanation last. No automatic API retries.
"""
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests

from src.llm_classifier import (ClassificationError,BudgetExceeded,CredentialMissing,
    validate_schema,stable_hash,strict_json,file_lock)
from src.recover_large_pilot import resilient_atomic_json as atomic_json
from src.step3f_classifier import (Step3FClassifier,SCHEMA as OLD_SCHEMA,AI_FIELDS,
    POLICY_PATH,POLICY_VERSION,source_spans,span_match,validate_span_prediction)
from src.validate_classification import ROOT

PROMPT_VERSION='step3f-1.1'
VALIDATION_VERSION='indexed-span-enum-2'
PROMPT_PATH=ROOT/'prompts/step3f-v1.1.txt'
SCHEMA=deepcopy(OLD_SCHEMA)
SCHEMA['properties']['ai_reasoning']['maxLength']=400
# Structured Outputs emits properties in schema order; keep all labels and IDs
# ahead of the explanation without ever accepting an unfinished response.
SCHEMA['properties']={k:v for k,v in SCHEMA['properties'].items() if k!='ai_reasoning'}|{
    'ai_reasoning':SCHEMA['properties']['ai_reasoning']}
SCHEMA['required']=list(SCHEMA['properties'])


def schema_for(title,description):
    schema=deepcopy(SCHEMA)
    ids=[span['id'] for span in source_spans(title,description)]
    # Static category/scope enums also count toward the API's 1,000-value limit.
    static_count=sum(len(p.get('enum',p.get('items',{}).get('enum',[]))) for p in schema['properties'].values())
    if len(ids)+static_count>1000:raise ClassificationError('Per-ad source-span enum exceeds API limit; do not truncate source')
    if ids:schema['properties']['ai_evidence_span_ids']['items']['enum']=ids
    else:schema['properties']['ai_evidence_span_ids']['maxItems']=0
    return schema


def validate_reliable_prediction(value,title,description,allow_formatting=False):
    validate_schema(value,schema_for(title,description))
    # Exact source offsets, duplicate IDs, scope/category mappings and abstention
    # invariants remain the original strict validator's responsibility.
    return validate_span_prediction(value,title,description)


def source_hash(title,description):
    return stable_hash(dict(job_title=title,description_text=description))


class Step3FReliableClassifier(Step3FClassifier):
    def __init__(self,*args,blocked_source_hashes,prompt_text=None,**kwargs):
        super().__init__(*args,prompt_text=prompt_text if prompt_text is not None else PROMPT_PATH.read_text(encoding='utf-8'),**kwargs)
        self.blocked_source_hashes=set(blocked_source_hashes)
        # Checkpoint compatibility across versions, including terminal failures.
        for result in self.state['results'].values():
            if result.get('ai_prompt_version')!=PROMPT_VERSION or result.get('ai_validation_version')!=VALIDATION_VERSION:
                raise ClassificationError('Historical checkpoint is incompatible; never migrate old predictions into this cache')

    def payload(self,title,description):
        payload=super().payload(title,description)
        payload['response_format']['json_schema'].update(name='step3f_atomic_span_classification',schema=schema_for(title,description))
        return payload

    def cache_key(self,title,description):
        return stable_hash(dict(stage='step3f-reliability-1.1',prompt_version=PROMPT_VERSION,
            validation_version=VALIDATION_VERSION,policy_version=POLICY_VERSION,policy_hash=self.policy_hash,
            provider=self.config.provider,endpoint=self.config.base_url.rstrip('/'),payload=self.payload(title,description)))

    def cached(self,title,description):
        key=self.cache_key(title,description);path=self.cache_dir/(key+'.json')
        if not path.exists():return None
        entry=json.loads(path.read_text(encoding='utf-8'))
        if entry.get('cache_key')!=key:raise ClassificationError('Cache identity mismatch')
        entry['prediction']=validate_reliable_prediction({k:entry['prediction'][k] for k in AI_FIELDS},title,description)
        if entry.get('response_schema_sha256')!=stable_hash(schema_for(title,description)):
            raise ClassificationError('Per-ad response schema identity mismatch')
        entry['evidence_match']=span_match(entry['prediction']['ai_evidence_span_ids'],title,description)
        return entry

    def metadata(self,key,entry,cache_hit=False):
        result=super().metadata(key,entry,cache_hit)
        result.update(ai_prompt_version=PROMPT_VERSION,ai_validation_version=VALIDATION_VERSION,
            ai_schema_sha256=stable_hash(SCHEMA),ai_response_schema_sha256=entry.get('response_schema_sha256'))
        return result

    def failure(self,identifier,key,status):
        result=dict.fromkeys(AI_FIELDS)
        result.update(ai_status=status,ai_review_required=True,ai_evidence=None,ai_cache_key=key,
            ai_provider=self.config.provider,ai_model=self.config.model,ai_prompt_version=PROMPT_VERSION,
            ai_policy_version=POLICY_VERSION,ai_policy_sha256=self.policy_hash,ai_validation_version=VALIDATION_VERSION,
            ai_human_verified=False,classification_status='Technical failure / requires retry',
            ai_validation_status='rejected_or_not_completed',ai_source_run=str(self.checkpoint_path.parent.resolve()))
        self.state['results'][str(identifier)]=result;self.save()
        return result

    def invalid_response_diagnostic(self,body,response,api_key,error,stage):
        result=super().invalid_response_diagnostic(body,response,api_key,error,stage)
        content=result.get('content') or ''
        result.update(configured_output_token_cap=self.config.max_output_tokens,
            reported_completion_tokens=body.get('usage',{}).get('completion_tokens') if isinstance(body,dict) else None,
            captured_content_characters=len(content),
            captured_whitespace_fraction=sum(c.isspace() for c in content)/len(content) if content else 0,
            incomplete_response=result.get('finish_reason')!='stop')
        return result

    def classify(self,identifier,title,description):
        # Unknown billing is tied to source content, not just the old prompt key.
        # Callers pass hashes from the preserved history guard file (also for new IDs).
        if source_hash(title,description) in self.blocked_source_hashes:
            if self.execute:return self.failure(identifier,self.cache_key(title,description),'historical_request_outcome_unknown')
            return dict(ai_status='historical_request_outcome_unknown',ai_relevance=None,ai_primary_category=None,
                ai_review_required=True,ai_human_verified=False)
        return self._classify(identifier,title,description)

    def _classify(self, identifier, title, description):
        if not isinstance(title, str) or not isinstance(description, str):
            raise ValueError('Pass original title/description strings; missing values must be empty strings')
        key = self.cache_key(title, description)
        prior = self.state['results'].get(str(identifier))
        if prior:
            if prior['ai_cache_key'] != key:
                raise ClassificationError('Checkpoint input/prompt changed')
            if prior['ai_status'] in {'completed', 'cache_hit', 'missing_description'}:
                validated = validate_reliable_prediction({name:prior[name] for name in AI_FIELDS}, title, description,allow_formatting=True)
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
                        prediction = validate_reliable_prediction(parsed, title, description,allow_formatting=True)
                        valid_entry = dict(cache_key=key, prediction=prediction,
                            usage=body.get('usage', {}), model_returned=body.get('model', self.config.model),
                            attempt_id=item['attempt_id'],response_schema_sha256=stable_hash(schema_for(title,description)),
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

