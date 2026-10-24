"""Offline duties candidate audit. v1 approvals remain human-only; no execution switch."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src.precision_inclusion import InclusionConfig, inclusion_decision, decode_prediction
from src.recover_large_pilot import resilient_atomic_json as atomic_json
from src.run_step4 import write_csv
from src.step4_jobs import validate_classification
from src.step6 import publish
from src.step3f_reliable_classifier import source_hash
from src.validate_classification import ROOT, read_csv, sha256

VERSION = 'duties-candidate-rules-1.1'
PROPOSED_POLICY = 'precision-first-v2-auto-draft'
PROPOSAL = ROOT/'docs/precision_first_inclusion_policy_v2_auto_DRAFT.md'
PREPARED = ROOT/'data/processed/step6-prepared500-final-20261024'
RUN = ROOT/'data/processed/step6-live500-20261024'

# Narrow category-specific responsibility objects; ordinary SQL/software/AI use
# cannot qualify. Unsupported categories abstain, never get a fallback match.
OBJECTS = {
    'Data Engineer': r'data\s*pipelines?|data\s*platforms?|analytical\s+(?:data\s+)?pipelines?|datapipelines?|dataplattform(?:ar|en)?',
    'Data Warehouse / ETL Developer': r'data\s*warehouses?|ETL(?:[- ](?:pipelines?|flows?|workflows?))?|ELT[- ](?:pipelines?|workflows?)|datalager|ETL[- ]flöden',
    'BI Developer / BI Analyst': r'BI[- ](?:solutions?|systems?|models?)|Power\s*BI\s+(?:dashboards?|reports?|solutions?)|BI[- ](?:lösningar|system)|dashboards?',
    'ML Engineer': r'(?:machine[- ]learning|ML)[- ](?:models?|systems?|pipelines?)|model\s+(?:deployment|monitoring)|maskininlärningsmodeller',
    'AI Engineer': r'(?:LLM|generative\s+AI|AI)[- ](?:applications?|agents?|systems?|integrations?)|AI[- ](?:applikationer|agenter|lösningar)',
    'Data Scientist': r'predictive\s+models?|statistical\s+models?|prediktiva\s+modeller|statistiska\s+modeller',
    'Analytics Engineer': r'analytical\s+data\s+models?|dbt\s+(?:models?|transformations?)|analytiska\s+datamodeller',
    'Data Architect': r'(?:enterprise|analytical|data)[- ](?:data\s+)?architectures?|dataarkitektur(?:en)?',
}
ACTION = r'(?:design|build|develop|implement|operate|maintain|deploy|evaluate|monitor|own|utveckla|bygga|bygg|implementera|driva|förvalta|designa|utforma)(?:s|ing|r)?'
ACTOR = re.compile(r'\b(?:you\s+(?:(?:will|would)\s+)?(?:be\s+responsible\s+for\s+)?|du\s+(?:(?:kommer\s+att|ska|skall)\s+)?)',re.I)
QUALIFICATION = re.compile(r'\b(?:experience|experienced|skills?|knowledge|proficien\w*|familiar\w*|qualifications?|requirements?|meritorious|meriterande|erfarenhet|kunskap\w*|kompetens\w*|krav|vi\s+söker|we\s+(?:seek|are\s+looking))\b',re.I)
NEGATION = re.compile(r"\b(?:not|never|no|inte|ej|aldrig|without|utan|if|may|might|could|optional|\u00f6nskv\u00e4rt|won['\u2019]t|don['\u2019]t|vill)\b",re.I)
DUTY_HEADING = re.compile(r'^\s*(?:responsibilities|your\s+(?:responsibilities|tasks)|what\s+you(?:\s+will|’ll|\x27ll)\s+do|arbetsuppgifter|dina\s+arbetsuppgifter|ansvarsområden)\s*[:]?\s*$',re.I)
STOP_HEADING = re.compile(r'^\s*(?:qualifications|requirements|about\s+(?:us|you)|your\s+(?:profile|background)|we\s+offer|krav|kvalifikationer|din\s+(?:profil|bakgrund)|om\s+oss|vi\s+erbjuder)\s*[:]?\s*$',re.I)
TITLE_RESTRICTION = re.compile(r'\b(?:intern(?:ship)?|thesis|doctoral|postdoc\w*|PhD|student\w*|praktik\w*|examensarbete|doktorand\w*|forskare|researcher|professor)\b',re.I)


def duty_matches(description, category):
    """Exact clause offsets with actor/action/object; no model reasoning consumed."""
    if category not in OBJECTS:return []
    pattern=re.compile(r'\b'+ACTION+r'\b[^.;!?\n]{0,100}?\b(?:'+OBJECTS[category]+r')\b',re.I)
    result=[];in_duties=False;seen=set()
    for line in re.finditer(r'[^\r\n]+',description):
        text=line.group()
        if DUTY_HEADING.fullmatch(text):in_duties=True;continue
        if STOP_HEADING.fullmatch(text):in_duties=False;continue
        for clause in re.finditer(r'[^.;!?]+',text):
            value=clause.group();base=line.start()+clause.start()
            if QUALIFICATION.search(value) or NEGATION.search(value):continue
            match=pattern.search(value)
            if not match:continue
            prefix=value[:match.start()]
            # An explicit addressed employee, or an imperative responsibility bullet.
            actor=ACTOR.search(prefix)
            bullet=in_duties and bool(re.match(r'^\s*[-*•]\s*'+ACTION+r'\b',value,re.I))
            if not actor and not bullet:continue
            if actor and len(prefix[actor.end():])>70:continue
            if re.search(r'\b(?:team|colleagues?|company|clients?|employer|kolleg\w*|företag\w*)\b',prefix,re.I):continue
            signature=value.strip().casefold()
            if signature in seen:continue
            seen.add(signature)
            start=base;end=base+len(value)
            result.append(dict(source_field='description_text',start=start,end=end,text=description[start:end],
                action_object_start=base+match.start(),action_object_end=base+match.end(),
                action_object_text=match.group(),rule_category=category,
                actor_basis='employee_address' if actor else 'duties_bullet'))
    return result


def compound_duty(matches, category):
    """Two concrete operations on the same category object in a single clause."""
    if category not in OBJECTS:return False
    operations = r'(?:design(?:ing)?\s+and\s+(?:build(?:ing)?|implement(?:ing)?)|build(?:ing)?\s+and\s+operat(?:e|ing)|develop(?:ing)?\s+and\s+deploy(?:ing)?|designa,?\s+bygga\s+och\s+vidareutveckla)'
    pattern = re.compile(r'\b'+operations+r'\s+(?:(?:the|our|big|modern|scalable)\s+){0,3}(?:'+OBJECTS[category]+r')\b', re.I)
    return any(pattern.search(m['text']) for m in matches)


def audit_row(row):
    """Candidate assessment is distinct from approval; never creates human audit fields."""
    r=decode_prediction(row)
    valid,error=validate_classification(r,r)
    current=inclusion_decision(r,InclusionConfig())
    reasons=[];matches=[]
    if not valid:
        disposition='technical_failure';reasons=['invalid_or_incomplete_classification',error]
    elif r['ai_relevance']=='irrelevant':
        disposition='excluded_by_policy';reasons=['accepted_irrelevant_prediction_unverified']
    elif r['ai_relevance']=='uncertain':
        disposition='human_review';reasons=['semantic_uncertainty']
    elif r['ai_opportunity_types'] or TITLE_RESTRICTION.search(r['job_title']):
        disposition='excluded_by_policy';reasons=['restricted_opportunity_or_research_title']
    else:
        disposition='human_review'
        if r.get('ai_prompt_version')!='step3f-1.1':reasons.append('classifier_version_not_supported_by_automation')
        if r['ai_scope']!='core_industry':reasons.append('special_scope_requires_human_review')
        if r['ai_multi_role']:reasons.append('multi_role_requires_human_review')
        if r['ai_secondary_categories']:reasons.append('hybrid_categories_require_human_review')
        if r['ai_confidence']!='high':reasons.append('confidence_below_threshold')
        if r['ai_review_required']:reasons.append('mandatory_review_flag')
        if r['ai_primary_category'] not in OBJECTS:reasons.append('category_not_supported_by_automation')
        matches=duty_matches(r['description_text'],r['ai_primary_category'])
        if len(matches)<2 and not compound_duty(matches,r['ai_primary_category']):reasons.append('insufficient_distinct_explicit_duty_operations')
        spans=r.get('ai_evidence_spans',[])
        if not any(m['start']<s['end'] and s['start']<m['end'] for m in matches for s in spans if s['source_field']=='description_text'):
            reasons.append('duty_rules_not_corroborated_in_selected_description_spans')
        if not reasons:
            disposition='candidate_pending_policy_approval'
            reasons=['multiple_category_specific_employee_operations','v1_person_audit_required','v2_not_approved']
    return dict(advertisement_id=r['advertisement_id'],job_title=r['job_title'],
        ai_relevance=r.get('ai_relevance'),ai_primary_category=r.get('ai_primary_category'),
        ai_scope=r.get('ai_scope'),ai_confidence=r.get('ai_confidence'),
        ai_review_required=r.get('ai_review_required'),ai_multi_role=r.get('ai_multi_role'),
        ai_opportunity_types=r.get('ai_opportunity_types'),
        pending_246=bool(valid and r.get('ai_relevance')=='relevant' and not r.get('ai_opportunity_types')),
        candidate_disposition=disposition,decision_reasons=reasons,duty_evidence=matches,
        active_policy_status=current['inclusion_status'],active_policy_reasons=current['inclusion_reasons'],
        automatic_approval=False,human_reviewed=False,approval_method='none',
        audit_method='deterministic_candidate_analysis',audit_engine_version=VERSION,
        active_policy_version='precision-first-v1',proposed_policy_version=PROPOSED_POLICY,
        source_sha256=source_hash(r['job_title'],r['description_text']))


def run(output,prepared=PREPARED,run_dir=RUN):
    output,prepared,run_dir=Path(output),Path(prepared),Path(run_dir)
    if output.exists():raise FileExistsError('Use a new versioned output')
    # Protect all existing source datasets, runs, pending markers and cache receipts.
    protected={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'data').rglob('*') if p.is_file()}
    manifest=json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    if manifest['input_sha256']!=sha256(prepared/'selected_jobs.csv'):raise ValueError('Selection identity changed')
    if manifest['identities'][str(ROOT/'docs/precision_first_inclusion_policy_v1.md')]!=sha256(ROOT/'docs/precision_first_inclusion_policy_v1.md'):
        raise ValueError('Active policy changed; re-evaluate policy boundary')
    frame=read_csv(prepared/'selected_jobs.csv')
    if frame.advertisement_id.duplicated().any():raise ValueError('Duplicate selected IDs')
    state=json.loads((run_dir/'checkpoint.json').read_text(encoding='utf-8'))
    if set(state['results'])!=set(frame.advertisement_id):raise ValueError('Incomplete or unexpected saved results')
    decisions=[audit_row(dict(r,**state['results'][r['advertisement_id']])) for r in frame.to_dict('records')]
    # Deliberately no audit CSV is supplied: candidates cannot populate person-audit fields.
    publication=publish(prepared,run_dir,output)
    write_csv(pd.DataFrame(decisions),output/'automatic_duties_decisions.csv')
    for name,status in [('automatic_candidates','candidate_pending_policy_approval'),('human_review','human_review'),('policy_excluded','excluded_by_policy')]:
        rows=[d for d in decisions if d['candidate_disposition']==status]
        write_csv(pd.DataFrame(rows,columns=pd.DataFrame(decisions).columns),output/(name+'.csv'))
    with (output/'automatic_duties_decisions.jsonl').open('w',encoding='utf-8') as f:
        for decision in decisions:f.write(json.dumps(decision,ensure_ascii=False,allow_nan=False)+'\n')
    pending=[d for d in decisions if d['pending_246']]
    report=dict(audit_engine_version=VERSION,active_policy='precision-first-v1',proposed_policy=PROPOSED_POLICY,
        policy_change_approved=False,total_saved_results=len(decisions),pending_relevant=len(pending),
        automatically_eligible_under_active_policy=0,automatically_approved=0,
        candidates_pending_policy_approval=sum(d['candidate_disposition']=='candidate_pending_policy_approval' for d in decisions),
        dispositions=dict(Counter(d['candidate_disposition'] for d in decisions)),
        pending_246_dispositions=dict(Counter(d['candidate_disposition'] for d in pending)),
        pending_246_reason_counts=dict(Counter(reason for d in pending for reason in d['decision_reasons'])),
        new_analyzed=publication['new_analyzed'],final_analyzed=publication['analyzed_jobs'],paid_requests=0,
        provenance=dict(checkpoint_sha256=sha256(run_dir/'checkpoint.json'),selection_sha256=sha256(prepared/'selected_jobs.csv'),
            active_policy_sha256=sha256(ROOT/'docs/precision_first_inclusion_policy_v1.md'),
            candidate_rules_sha256=sha256(Path(__file__)),proposal_sha256=sha256(PROPOSAL)),
        limitations='Rule matches are uncalibrated candidate evidence, not independent correctness proof or a human audit. No accuracy claim. Absence of a match is not occupational irrelevance.')
    atomic_json(output/'automatic_duties_summary.json',report)
    (output/'proposed_policy.md').write_bytes(PROPOSAL.read_bytes())
    changed=[p for p,h in protected.items() if not (ROOT/p).is_file() or sha256(ROOT/p)!=h]
    if changed:raise RuntimeError('Earlier data changed: '+str(changed))
    atomic_json(output/'automatic_audit_preservation.json',dict(unchanged_files_verified=len(protected),files=protected))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepared',type=Path,default=PREPARED);p.add_argument('--run-dir',type=Path,default=RUN)
    a=p.parse_args();print(json.dumps(run(a.output,a.prepared,a.run_dir),indent=2))

if __name__=='__main__':main()
