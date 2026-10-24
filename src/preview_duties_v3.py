"""Preview-only v3 duties policy. Never classifies, approves, publishes or calls APIs."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src.automatic_duties_audit import PREPARED, RUN
from src.dashboard_data import latest_dataset
from src.precision_inclusion import decode_prediction
from src.recover_large_pilot import resilient_atomic_json as atomic_json
from src.run_step4 import write_csv
from src.step3f_reliable_classifier import source_hash
from src.step4_jobs import SOURCE, validate_classification
from src.validate_classification import ROOT, read_csv, sha256

VERSION='duties-v3-preview-1.0'
POLICY='precision-first-v3-draft'
POLICY_PATH=ROOT/'docs/precision_first_inclusion_policy_v3_DRAFT.md'

# Each object names occupational work rather than a generic tool or an AI mention.
OBJECTS={
 'Data Engineer':r'data[- ]?pipelines?|data[- ]?platforms?|data ingestion|analytical data infrastructure|datapipelines?|dataplattform\w*|analytiska dataflöden',
 'Data Warehouse / ETL Developer':r'data warehouses?|ETL(?:[- ](?:pipelines?|processes?|flows?))?|ELT(?:[- ](?:pipelines?|processes?))?|datalager\w*|datavaruhus\w*',
 'Analytics Engineer':r'dbt[- ](?:models?|transformations?)|analytical data models?|analytics (?:models?|transformations?)|analytiska datamodeller|dbt[- ]modeller',
 'BI Developer / BI Analyst':r'BI[- ](?:solutions?|systems?|models?|dashboards?|reports?)|business intelligence (?:solutions?|systems?)|Power\s*BI(?:[- ](?:reports?|dashboards?|models?))?|BI[- ](?:lösningar|system|modeller|rapporter)|dashboards?|visualiseringar',
 'Data Analyst':r'(?:business |customer |measurement |sales )?data (?:analysis|analytics|insights)|data into (?:actionable )?insights|analys(?:is|es) (?:of|on) (?:business |customer |measurement )?data|dataanalys\w*|analysera (?:stora |komplexa )?data|data till (?:konkreta )?insikter|statistik|analysverktyg',
 'Data Scientist':r'predictive models?|statistical models?|machine learning models?|prediktiva modeller|statistiska modeller|prognosmodeller|prediktiv\w* (?:analys|modell\w*)|bildanalysmodeller',
 'ML Engineer':r'machine[- ]learning(?:[- ](?:models?|systems?|pipelines?|workflows?))?|ML[- ](?:models?|systems?|pipelines?)|MLOps(?:[- ]pipelines?)?|model deployment|maskininlärnings(?:modeller|system|algoritmer)|ML[- ]modeller|neural networks',
 'AI Engineer':r'AI[- ](?:applications?|agents?|systems?|solutions?|integrations?)|LLM[- ](?:applications?|systems?|solutions?|integrations?)|RAG[- ](?:pipelines?|systems?|solutions?)|generative AI (?:applications?|solutions?)|AI[- ](?:applikationer|agenter|system|lösningar|integrationer)|agentiska system',
 'Data Architect':r'data architectures?|enterprise data models?|dataarkitektur\w*|analytical architectures?|arkitektur\w* för (?:data|AI och data)',
 'Data Governance / Data Quality':r'data governance|data quality (?:rules?|frameworks?|management)|metadata management|data stewardship|datastyrning|datakvalitet\w*|masterdatahantering',
 'AI / ML Research':r'machine[- ]learning (?:methods?|algorithms?|models?)|AI (?:methods?|algorithms?)|maskininlärnings(?:metoder|algoritmer|modeller)',
 'Statistician':r'statistical (?:analysis|modelling|modeling)|statistisk(?:a)? (?:analys\w*|modell\w*|metod\w*)|regressionsanalys',
 'Data / AI Management':r'data (?:engineering |science |analytics )?teams?|AI teams?|analytics teams?|data och AI[- ](?:team|verksamhet)|data[- ]och AI[- ](?:team|verksamhet)',
}
ACTION=r'(?:build\w*|develop\w*|design\w*|implement\w*|deploy\w*|operat\w*|maintain\w*|optimi[sz]\w*|creat\w*|evaluat\w*|train\w*|monitor\w*|own\w*|architect\w*|analy[sz]\w*|transform\w*|automat\w*|lead\w*|manag\w*|utveckl\w*|vidareutveckl\w*|bygg\w*|design\w*|implementer\w*|förvalt\w*|underhåll\w*|utforma\w*|produktionssätt\w*|träna\w*|analysera\w*|omvandl\w*|visualiser\w*|skapa\w*|ansvar\w*|leda|leder|genomför\w*)'
NOUN_ACTION=r'(?:development|deployment|maintenance|design|implementation|ownership|utveckling|vidareutveckling|förvaltning|underhåll|utformning|modellering|implementering|ansvar)\s+(?:of|for|av|för)'
START=re.compile(r'\b(?:your responsibilities|key responsibilities|responsibilities|your tasks|what you(?: will|[’\x27]ll) do|you will(?=:)|you[’\x27]ll(?=:)|arbetsuppgifter(?:\s*/\s*ansvar)?|ansvarsområden|ditt uppdrag|dina uppgifter|role overview|about the role|om rollen|om tjänsten)\b',re.I)
STOP=re.compile(r'\b(?:qualifications|requirements|who (?:we are|we[’\x27]re) looking for|what we offer|we offer|your profile|your background|about (?:us|the company)|kvalifikationer|din profil|din bakgrund|vi söker dig|vi erbjuder|vad vi erbjuder|om oss|krav(?=[A-ZÅÄÖ\s:]))\b',re.I)
QUAL=re.compile(r'\b(?:experience(?:d)?|skills?|knowledge|familiar\w*|proficien\w*|meritorious|meriterande|erfarenhet|kunskap\w*|kompetens\w*|qualification\w*|certif\w*|degree|examen|utbildning|intresse för)\b',re.I)
NEG=re.compile(r'\b(?:not|never|without|inte|ej|utan|if|might|could|may|optional|önskvärt|won[’\x27]t|don[’\x27]t)\b',re.I)
ACTOR=re.compile(r'\b(?:you|your (?:role|responsibilities|tasks)|du|dina (?:arbetsuppgifter|uppgifter)|ditt (?:arbete|uppdrag)|rollen|tjänsten|the (?:role|position|candidate))\b',re.I)
THIRD=re.compile(r'\b(?:our company|our clients|our customers|our team|colleagues|the team|våra kunder|vårt företag|våra kollegor|kollegorna|företaget)\b',re.I)
COLLAB=re.compile(r'\b(?:collaborat\w*|work(?:ing)? with|samarbet\w*|arbeta(?:r)? (?:nära|med))\b.{0,60}\b(?:scientists?|engineers?|analysts?|utvecklare|forskare|analytiker)\b',re.I)
RESTRICTED_TITLE=re.compile(r'\b(?:intern(?:ship)?|thesis|doctoral|postdoc\w*|PhD|student\w*|praktik\w*|examensarbete|doktorand\w*)\b',re.I)
RESEARCH_TITLE=re.compile(r'\b(?:researcher|professor|forskare|forskarassistent)\b',re.I)


def clauses(description):
    """Segment at headings and punctuation without changing original character offsets."""
    markers=[]
    for typ,pattern in [('duties',START),('other',STOP)]:
        for m in pattern.finditer(description):
            if typ=='duties':
                previous=max(description.rfind(c,0,m.start()) for c in ['\n','.','!','?'])
                prefix=description[previous+1:m.start()].strip()
                # A mention of someone else's responsibilities is not a heading.
                if prefix and not ACTOR.search(prefix) and not re.match(r'you',m.group(),re.I):continue
            markers.append((m.start(),m.end(),typ))
    # Headings provide context only until the next heading; never beyond a stop.
    markers.sort(key=lambda x:(x[0],0 if x[2]=='other' else 1))
    boundaries=sorted({0,len(description),*(x for m in markers for x in m[:2])})
    duties=False
    for a,b in zip(boundaries,boundaries[1:]):
        heading=next((m for m in markers if m[0]==a and m[1]==b),None)
        if heading:
            duties=heading[2]=='duties';continue
        for m in re.finditer(r'[^.;!?\r\n]+',description[a:b]):
            start=a+m.start();end=a+m.end()
            yield start,end,description[start:end],duties


def evidence(description,category):
    if category not in OBJECTS:return []
    # Both active employee duties and noun-form duty lists are supported.
    pattern=re.compile(r'\b(?:'+NOUN_ACTION+'|'+ACTION+r')\b[^.;!?\r\n]{0,100}?\b(?:'+OBJECTS[category]+r')\b',re.I)
    matches=[]
    for start,end,text,in_duties in clauses(description):
        if QUAL.search(text) or NEG.search(text) or COLLAB.search(text):continue
        match=pattern.search(text)
        if not match:continue
        actor=ACTOR.search(text[:match.start()])
        if not actor and not in_duties:continue
        if actor and match.start()-actor.end()>180:continue
        prefix=text[:match.start()]
        if actor and THIRD.search(prefix[actor.end():]):continue
        if not actor and THIRD.search(prefix):continue
        if re.search(r'\b(?:we|our|vi|vår\w*)\b',prefix,re.I) and not actor:continue
        # Questions/promotional invitations do not assert assigned duties.
        if re.search(r'\b(?:vill du|would you|do you want|want to)\b',prefix,re.I):continue
        matches.append(dict(source_field='description_text',start=start,end=end,text=text,
            duty_start=start+match.start(),duty_end=start+match.end(),duty_text=match.group(),
            category=category,context='employee_statement' if actor else 'duties_section',
            context_is_not_independent_truth=True))
    return matches


def corroborated(hit,spans):
    # Require the whole concrete action/object finding in the union of validated
    # selected description intervals, not an unrelated part of the same sentence.
    cursor=hit['duty_start']
    for span in sorted((s for s in spans if s['source_field']=='description_text'),key=lambda s:s['start']):
        if span['start']>cursor:break
        if span['end']>cursor:cursor=span['end']
        if cursor>=hit['duty_end']:return True
    return False


def decision(row):
    r=decode_prediction(row);valid,error=validate_classification(r,r)
    hits=evidence(r['description_text'],r.get('ai_primary_category')) if valid and r.get('ai_relevance')=='relevant' else []
    reasons=[]
    opportunities=r.get('ai_opportunity_types') or []
    scope=r.get('ai_scope')
    lane=('opportunity' if opportunities or RESTRICTED_TITLE.search(r['job_title']) else
          'academic_research' if scope=='academic_ai_ml_research' or RESEARCH_TITLE.search(r['job_title']) else
          'adjacent' if scope=='adjacent_borderline' else 'governance' if scope=='data_ai_governance' or r.get('ai_primary_category')=='Data Governance / Data Quality' else
          'core_industry' if scope=='core_industry' else 'unresolved')
    overlaps=[h for h in hits if corroborated(h,r.get('ai_evidence_spans') or [])]
    if not valid:outcome='technical_failure';reasons=['source_validation_failed',error]
    elif r.get('ai_relevance')!='relevant':outcome='not_relevant_candidate';reasons=['saved_'+r['ai_relevance']]
    else:
        if r.get('ai_prompt_version')!='step3f-1.1':reasons.append('unsupported_classifier_version')
        if lane!='core_industry':reasons.append('separate_reporting_lane_not_in_core_auto_scope')
        if r.get('ai_confidence')!='high':reasons.append('confidence_below_high_screen')
        if r.get('ai_review_required') is not False:reasons.append('outstanding_model_review')
        if r.get('ai_multi_role'):reasons.append('multiple_occupations_require_review')
        if r.get('ai_primary_category') not in OBJECTS:reasons.append('unresolved_or_generic_primary_category')
        if not hits:reasons.append('no_category_supported_assigned_duties')
        elif not overlaps:reasons.append('duties_not_corroborated_by_validated_description_evidence')
        if reasons:outcome='separate_lane' if lane!='core_industry' else 'abstain'
        else:outcome='would_qualify_v3';reasons=['explicit_assigned_category_duties','validated_description_evidence','high_confidence_screen_and_no_mandatory_review']
    return dict(advertisement_id=r['advertisement_id'],job_title=r['job_title'],saved_relevance=r.get('ai_relevance'),
        saved_category=r.get('ai_primary_category'),saved_scope=scope,reporting_lane=lane,
        opportunity_types=opportunities,saved_review_required=r.get('ai_review_required'),
        saved_confidence=r.get('ai_confidence'),saved_multi_role=r.get('ai_multi_role'),
        saved_secondary_categories=r.get('ai_secondary_categories'),preview_decision=outcome,reasons=reasons,
        duty_evidence=hits,corroborated_evidence_count=len(overlaps),source_sha256=source_hash(r['job_title'],r['description_text']),
        policy_version=POLICY,engine_version=VERSION,activated=False,approved=False,human_reviewed=False,
        assessment_method='deterministic_source_duties_preview')


def run(output,prepared=PREPARED,run_dir=RUN):
    output,prepared,run_dir=Path(output),Path(prepared),Path(run_dir)
    if output.exists():raise FileExistsError('Use a new preview directory')
    before=latest_dataset(ROOT/'data/processed')
    protected={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'data').rglob('*') if p.is_file()}
    state=json.loads((run_dir/'checkpoint.json').read_text(encoding='utf-8'))
    manifest=json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    identity=json.loads((run_dir/'identity.json').read_text(encoding='utf-8'))
    if identity['preparation']!=sha256(prepared/'manifest.json') or manifest['input_sha256']!=sha256(prepared/'selected_jobs.csv'):
        raise ValueError('Run/preparation mismatch')
    frame=read_csv(prepared/'selected_jobs.csv')
    canonical=read_csv(SOURCE).set_index('advertisement_id').to_dict('index')
    if frame.advertisement_id.duplicated().any() or set(frame.advertisement_id)!=set(state['results']):raise ValueError('Saved IDs mismatch')
    rows=[]
    for r in frame.to_dict('records'):
        c=canonical[r['advertisement_id']]
        if any(r[k]!=c[k] for k in ['job_title','description_text']):raise ValueError('Original source changed')
        rows.append(decision(dict(r,**state['results'][r['advertisement_id']])))
    relevant=[r for r in rows if r['saved_relevance']=='relevant']
    eligible=[r for r in relevant if r['preview_decision']=='would_qualify_v3']
    output.mkdir(parents=True)
    write_csv(pd.DataFrame(rows),output/'all_saved_decisions.csv')
    write_csv(pd.DataFrame(relevant),output/'relevant294_preview.csv')
    write_csv(pd.DataFrame(eligible,columns=pd.DataFrame(rows).columns),output/'would_qualify.csv')
    with (output/'audit_receipts.jsonl').open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    report=dict(policy=POLICY,engine=VERSION,preview_only=True,activated=False,paid_requests=0,
        total_saved_results=len(rows),relevant=len(relevant),would_qualify=len(eligible),
        relevant_decisions=dict(Counter(r['preview_decision'] for r in relevant)),
        relevant_reporting_lanes=dict(Counter(r['reporting_lane'] for r in relevant)),
        qualifying_categories=dict(Counter(r['saved_category'] for r in eligible)),
        reason_counts=dict(Counter(reason for r in relevant if r['preview_decision']!='would_qualify_v3' for reason in r['reasons'])),
        technical_failures=sum(r['preview_decision']=='technical_failure' for r in rows),
        existing_analyzed=108,new_jobs_published=0,potential_combined_analyzed=108+len(eligible),
        dashboard_dataset=str(before),dashboard_changed=False,
        evidence_note='Uncalibrated deterministic duty audit, not independent human ground truth. Counts describe coverage, not accuracy.',
        hashes=dict(checkpoint=sha256(run_dir/'checkpoint.json'),selection=sha256(prepared/'selected_jobs.csv'),
            policy=sha256(POLICY_PATH),rules=sha256(Path(__file__))))
    (output/'policy_proposal.md').write_bytes(POLICY_PATH.read_bytes())
    if any(not (ROOT/p).is_file() or sha256(ROOT/p)!=h for p,h in protected.items()):raise RuntimeError('Original records changed')
    if latest_dataset(ROOT/'data/processed')!=before:raise RuntimeError('Preview changed dashboard discovery')
    atomic_json(output/'preservation.json',dict(unchanged_files_verified=len(protected),files=protected))
    atomic_json(output/'preview_summary.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.output),indent=2))

if __name__=='__main__':main()
