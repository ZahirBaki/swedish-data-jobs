"""Conservative post-classification gate; never overwrites model predictions."""
from dataclasses import asdict,dataclass
import json

from src.classify_jobs_v2 import CATEGORIES
from src.step3f_classifier import AI_FIELDS,validate_span_prediction,span_match
from src.step3f_reliable_classifier import validate_reliable_prediction

VERSION='precision-first-v1'


@dataclass(frozen=True)
class InclusionConfig:
    confidence_levels: tuple=('high',)
    scopes: tuple=('core_industry','data_ai_governance')
    require_no_review: bool=True
    require_duties_audit: bool=True
    allow_other_category: bool=False
    allow_opportunities: bool=False

    def __post_init__(self):
        if not self.confidence_levels or set(self.confidence_levels)-{'high','medium','low'}:raise ValueError('Invalid confidence levels')
        if not self.scopes or set(self.scopes)-{'core_industry','data_ai_governance','academic_ai_ml_research','adjacent_borderline'}:
            raise ValueError('Invalid reporting scopes')


def decode_prediction(row):
    out=dict(row)
    for k in ['ai_secondary_categories','ai_evidence_span_ids','ai_opportunity_types','ai_evidence','ai_evidence_spans']:
        if isinstance(out.get(k),str):
            out[k]=json.loads(out[k]) if out[k] else []
    for k in ['ai_review_required','ai_multi_role','ai_human_verified']:
        if isinstance(out.get(k),str):
            if out[k].lower() not in {'true','false'}:raise ValueError('Invalid boolean '+k)
            out[k]=out[k].lower()=='true'
    if out.get('ai_primary_category')=='':out['ai_primary_category']=None
    return out


def inclusion_decision(row,config=InclusionConfig()):
    def result(status,*reasons):
        return dict(inclusion_status=status,inclusion_reasons=list(reasons),inclusion_policy_version=VERSION,
                    inclusion_config=asdict(config))
    status=row.get('ai_status')
    if status in {None,'','not_executed','dry_run'}:return result('not_executed','predictions_unavailable')
    if status not in {'completed','cache_hit'}:
        if status=='missing_description':return result('abstained','missing_description')
        return result('technical_failure','technical_status:'+str(status or 'not_executed'))
    try:
        r=decode_prediction(row)
        validator=validate_reliable_prediction if r.get('ai_prompt_version')=='step3f-1.1' else validate_span_prediction
        if r.get('ai_prompt_version') not in {'step3f-1.0','step3f-1.1'}:raise ValueError('Unknown prompt version')
        checked=validator({k:r[k] for k in AI_FIELDS},r['job_title'],r['description_text'])
        r.update(checked)
        spans=span_match(r['ai_evidence_span_ids'],r['job_title'],r['description_text'])['spans']
        if r.get('ai_evidence')!=[s['text'] for s in spans] or r.get('ai_evidence_spans',[])!=spans:
            raise ValueError('Stored source excerpts/offsets differ')
    except (ValueError,TypeError,KeyError):return result('technical_failure','invalid_source_evidence')
    if r['ai_relevance']=='uncertain':return result('abstained','semantic_uncertainty')
    assessment=r.get('duties_assessment','')
    if assessment in {'incidental_only','unrelated_duties'}:
        if not r.get('duties_assessor','').strip() or not r.get('inclusion_audit_notes','').strip():
            return result('abstained','unattributed_exclusion_audit')
        return result('excluded','incidental_tools_only' if assessment=='incidental_only' else 'unrelated_duties')
    if r['ai_relevance']=='irrelevant':return result('excluded','model_unrelated_duties_unverified')
    if r['ai_scope'] not in config.scopes:return result('out_of_scope','scope_not_selected')
    if r['ai_opportunity_types'] and not config.allow_opportunities:return result('out_of_scope','opportunity_not_selected')
    reasons=[]
    if not r['ai_primary_category']:reasons.append('category_unresolved')
    elif r['ai_primary_category'] not in CATEGORIES:reasons.append('category_unresolved')
    elif r['ai_primary_category']=='Other relevant Data & AI roles' and not config.allow_other_category:reasons.append('generic_category_requires_review')
    if assessment in {'insufficient','contradictory'}:reasons.append('duties_insufficient_or_contradictory')
    if r.get('category_assessment') in {'insufficient','contradictory'}:reasons.append('category_not_supported')
    if config.require_duties_audit and (assessment!='clear' or r.get('category_assessment')!='clear'
        or not r.get('duties_assessor','').strip() or not r.get('inclusion_audit_notes','').strip()):reasons.append('duties_not_confirmed')
    if r['ai_confidence'] not in config.confidence_levels:reasons.append('confidence_below_threshold')
    resolved=str(r.get('inclusion_review_resolved','')).lower()=='true' and assessment=='clear' and r.get('category_assessment')=='clear' and bool(r.get('duties_assessor','').strip()) and bool(r.get('inclusion_audit_notes','').strip())
    if config.require_no_review and r['ai_review_required'] and not resolved:reasons.append('review_outstanding')
    if not any(s['source_field']=='description_text' for s in spans):reasons.append('description_evidence_required')
    if reasons:return result('abstained',*reasons)
    return result('included','clear_duties_supported_category')
