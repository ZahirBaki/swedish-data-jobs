"""Proposed DE-only v3.1 audit; no activation, requests, reclassification or publication."""
import re
from copy import deepcopy

from src import preview_duties_v3 as v3
from src.precision_inclusion import decode_prediction
from src.step4_jobs import validate_classification

VERSION='de-duties-v3.1-preview-1.0'
POLICY='precision-first-v3.1-DE-draft'

START=re.compile(r'(?:what you(?: will|[’\x27]ll) (?:work on|be doing|do|own)|(?:your |key |main |core )?responsibilities|your tasks|your duties|(?:about |your |the )?(?:role|assignment)(?: overview| description)?|(?:dina (?:huvudsakliga )?)?arbetsuppgifter(?:\s*/\s*ansvar)?|vad du (?:kommer (?:att )?)?(?:arbeta|jobba) med|ansvarsområden|ditt uppdrag|din roll|om (?:rollen|tjänsten|uppdraget)|(?:till exempel|bland annat) kommer du att|you will(?=:)|you[’\x27]ll(?=:))',re.I)
STOP=re.compile(r'(?:requirements|qualifications|what you bring|what we(?:[’\x27]re| are) looking for|who we(?:[’\x27]re| are) looking for|your (?:profile|background|skills)|who you are|what you have|what we expect|what skills|skills and experience|technical skills|tech(?:nical)? stack|education|certifications|about (?:us|the company)|company description|our services|what we offer|we offer|benefits|vi söker dig|vi tror att du|din (?:profil|bakgrund)|kvalifikationer|kompetenskrav|krav|meriterande|teknisk miljö|om (?:oss|företaget)|vad vi erbjuder|vi erbjuder)',re.I)
ACTOR=re.compile(r'\b(?:you(?:[’\x27]ll)?|your (?:role|responsibilities|tasks)|du|din(?:a|t)? (?:roll|uppdrag|arbete|ansvar|arbetsuppgifter)|rollen|tjänsten|the (?:role|candidate|position))\b',re.I)
QUAL=re.compile(r'\b(?:experience\s+(?:in|of|with|building|developing|designing|maintaining)|(?:previous|prior|proven|hands-on|professional)\s+experience|knowledge\s+of|skills\s+in|familiarity\s+with|erfarenhet\s+(?:av|inom|som)|kunskap\w*\s+(?:i|om|inom)|meriterande|qualifications?|certification|degree|examen|intresse\s+för)\b',re.I)
VETO=re.compile(r'\b(?:not|never|without|inte|ej|utan|may|might|could|optional|if|önskvärt|won[’\x27]t|don[’\x27]t|vill du|would you|do you want|kan komma att)\b',re.I)
THIRD=re.compile(r'\b(?:our (?:company|clients?|customers?|team)|colleagues?|the team|we|vi|våra (?:kunder|kollegor)|vårt (?:team|företag)|företaget)\b',re.I)
OTHER_DUTIES=re.compile(r'\b(?:collaborat\w*|work(?:ing)? with|samarbet\w*|arbeta(?:r)? (?:med|nära))\b.{0,70}\b(?:data engineers?|data scientists?|utvecklare|forskare|analytiker)\b',re.I)
ACTION=re.compile(r'\b(?:build\w*|develop\w*|design\w*|maintain\w*|migrat\w*|operat\w*|orchestrat\w*|ingest\w*|integrat\w*|transform\w*|process\w*|implement\w*|deploy\w*|monitor\w*|troubleshoot\w*|optimi[sz]\w*|automat\w*|creat\w*|manage|configure|support|own|development|maintenance|migration|orchestration|operation|bygg\w*|utveckl\w*|vidareutveckl\w*|design\w*|underhåll\w*|förvalt\w*|migrer\w*|migrering|orkestrer\w*|läs\w*\s+in|integrer\w*|transformer\w*|bearbet\w*|implementer\w*|produktionssätt\w*|övervak\w*|felsök\w*|optimera\w*|automatiser\w*|skapa\w*|utform\w*|drift\w*|ansvar\w*)\b',re.I)
OBJECTS=[
 ('analytical_pipeline',r'(?:data|analytical)[-\s]*pipelines?|datapipelines?|analytiska dataflöden|data (?:models?|modeller)\s+(?:and|och)\s+(?:data\s+)?pipelines?'),
 ('etl_elt',r'(?:ETL|ELT)(?:\s*[/&-]\s*(?:ETL|ELT))?(?:[-\s]+(?:pipelines?|flows?|workflows?|process\w*|flöden))?'),
 ('analytical_storage',r'data[-\s]*warehouses?|(?:data[-\s]*)?lakehouses?|data lakes?|datalag(?:er|r)\w*|datavaruhus\w*|datasjö\w*'),
 ('data_platform',r'data[-\s]*platforms?|analytical data infrastructure|dataplattform\w*|analytiska dataplattform\w*'),
 ('qualified_ingestion',r'data ingestion|ingestion of data|datainläsning|datainhämtning'),
 ('spark_engineering',r'(?:Spark|PySpark|Databricks)[-\s]+(?:jobs?|pipelines?|transformations?|data processing|workflows?|jobb|dataflöden)'),
 ('qualified_transformation',r'data transformations?|transformations? (?:of|av) (?:analytical )?data|datatransformation\w*'),
]
OBJECTS=[(name,re.compile(r'\b(?:'+pattern+r')\b',re.I)) for name,pattern in OBJECTS]
ANALYTICAL_DESTINATION=re.compile(r'\b(?:analytics|analytical|warehouse|lakehouse|data platform|data lake|ETL|ELT|Spark|PySpark|Databricks|analys|analytisk\w*|datalag(?:er|r)\w*|dataplattform\w*)\b',re.I)
INTEGRATION=re.compile(r'\b(?:data integration|dataintegration\w*|dataflöden|data flows?|(?:integrat\w*|ingest\w*|transform\w*|integrer\w*|bearbet\w*)\s+data)\b',re.I)
UNRELATED_HEAD=re.compile(r'\b(?:CI/CD|CICD|backend|frontend|API\w*|web\w*|ORM|sales|selling|sell|procurement|recruit\w*|roadmap|budget|training|courses?|documentation|presentations?|försäljning|marknadsföring)\b',re.I)


def _boundary(description,start):
    previous=max(description.rfind(c,0,start) for c in ['\n','.','!','?',';'])
    return not description[previous+1:start].strip(' \t\r:-•*')


def segments(description):
    """Heading boundaries retain exact original offsets, even joined heading/body text."""
    markers=[]
    for kind,pattern in [('duties',START),('other',STOP)]:
        for m in pattern.finditer(description):
            # Guard embedded words, e.g. requirements vs requirement-setting duty text.
            if m.start() and description[m.start()-1].isalnum():continue
            tail=description[m.end():m.end()+1]
            if tail and tail.islower():continue
            if not _boundary(description,m.start()):continue
            rest=description[m.end():].split('\n',1)[0]
            if kind=='duties' and re.match(r'\s+(?:of|for)\s+(?:our|the|other|colleagues)',rest,re.I):kind='other'
            markers.append((m.start(),m.end(),kind))
    markers.sort(key=lambda m:(m[0],-(m[1]-m[0]),m[2]))
    nonoverlap=[]
    for m in markers:
        if not nonoverlap or m[0]>=nonoverlap[-1][1]:nonoverlap.append(m)
    cursor=0;heading=None
    for start,end,kind in nonoverlap+[(len(description),len(description),'end')]:
        for match in re.finditer(r'[^.;!?\r\n]+',description[cursor:start]):
            a=cursor+match.start();b=cursor+match.end()
            yield a,b,description[a:b],heading
        if kind!='end':heading=dict(start=start,end=end,text=description[start:end]) if kind=='duties' else None
        cursor=end


def evidence(description):
    found=[];seen=set()
    for start,end,clause,heading in segments(description):
        if QUAL.search(clause) or VETO.search(clause) or OTHER_DUTIES.search(clause):continue
        actions=list(ACTION.finditer(clause));actor=ACTOR.search(clause)
        if not actions:continue
        objects=[]
        for name,pattern in OBJECTS:
            objects.extend((name,m) for m in pattern.finditer(clause))
        if ANALYTICAL_DESTINATION.search(clause):
            for match in INTEGRATION.finditer(clause):
                # For verb-object forms, data is the object; the verb is the action.
                # Keep original match offsets rather than linking an action to itself.
                obj = re.compile(r'\bdata\b', re.I).search(clause, match.start(), match.end()) if ACTION.match(clause, match.start()) else match
                if obj is not None: objects.append(('qualified_integration', obj))
        for rule,obj in objects:
            preceding=[action for action in actions if action.end()<=obj.start() and obj.start()-action.end()<=110]
            if not preceding:continue
            action=preceding[-1]
            gap=clause[action.end():obj.start()]
            if re.match(r'\s+(?:dashboards?|reports?|UI|front[- ]?end|training|courses?)\b',clause[obj.end():],re.I):continue
            if UNRELATED_HEAD.search(gap):continue
            # Do not link an API/CI-CD action to the platform it merely supports.
            if action.group().casefold() in {'manage','support','own'} and re.search(r'\b(?:development|delivery|strategy|utveckling|leverans)\b',gap,re.I):continue
            prefix=clause[:action.start()]
            if actor:
                if abs(action.start()-actor.start())>180:continue
                # Third parties own the action if they occur after the employee subject.
                if THIRD.search(prefix[actor.end():]):continue
                if THIRD.search(prefix[:actor.start()]):continue
                attribution='employee_statement'
            elif heading:
                if THIRD.search(prefix):continue
                # Lists must begin with an engineering action/noun, not arbitrary prose.
                cleaned=prefix.strip(' \t\r:-•*0123456789()')
                if cleaned and len(cleaned)>45:continue
                attribution='assigned_duties_section'
            else:continue
            a=start+action.start();b=start+obj.end()
            key=(a,b,rule)
            if key in seen:continue
            seen.add(key)
            found.append(dict(source_field='description_text',start=start,end=end,text=description[start:end],
                duty_start=a,duty_end=b,duty_text=description[a:b],action_start=start+action.start(),action_end=start+action.end(),
                object_start=start+obj.start(),object_end=start+obj.end(),rule_id=rule,attribution=attribution,
                heading=deepcopy(heading),evidence_method='deterministic_full_original_source',
                source_evidence_is_not_independent_ground_truth=True))
    return found


def decision(row):
    r=decode_prediction(row);old=v3.decision(r)
    valid,error=validate_classification(r,r)
    out=dict(advertisement_id=r['advertisement_id'],policy_version=POLICY,engine_version=VERSION,
        preview_only=True,activated=False,approved=False,human_reviewed=False,
        ai_primary_category=r.get('ai_primary_category'),ai_relevance=r.get('ai_relevance'),
        original_v3_decision=old['preview_decision'],original_v3_reasons=old['reasons'],
        original_selected_evidence=deepcopy(r.get('ai_evidence_spans') or []),
        source_sha256=old['source_sha256'],reporting_lane=old['reporting_lane'])
    if r.get('ai_primary_category')!='Data Engineer' or r.get('ai_relevance')!='relevant':
        return dict(out,preview_decision='outside_DE_proposal',reasons=['not_relevant_primary_data_engineer'],duty_evidence=[])
    if not valid:
        return dict(out,preview_decision='technical_failure',reasons=['invalid_source_evidence',error],duty_evidence=[])
    blockers=[]
    if r.get('ai_prompt_version')!='step3f-1.1':blockers.append('legacy_classifier_not_compatible')
    if old['reporting_lane']!='core_industry':blockers.append('separate_reporting_lane')
    if r.get('ai_confidence')!='high':blockers.append('confidence_below_high')
    if r.get('ai_review_required') is not False:blockers.append('mandatory_model_review')
    if r.get('ai_multi_role') is not False:blockers.append('multi_role_or_missing_flag')
    spans=r.get('ai_evidence_spans') or []
    if not any(s['source_field']=='description_text' for s in spans):blockers.append('validated_description_span_required')
    hits=evidence(r['description_text'])
    retained=old['preview_decision']=='would_qualify_v3'
    # Preserve the already approved v3 route unchanged; the extension is additive.
    if not hits and retained:
        hits=[dict(h,rule_id='unchanged_validated_v3_route',evidence_method='frozen_v3_corroborated_duty') for h in old['duty_evidence'] if v3.corroborated(h,spans)]
    if not hits:blockers.append('no_explicit_assigned_data_engineering_duty')
    for hit in hits:
        if r['description_text'][hit['start']:hit['end']]!=hit['text'] or r['description_text'][hit['duty_start']:hit['duty_end']]!=hit['duty_text']:
            raise ValueError('Deterministic evidence offsets differ from original source')
        hit['selected_span_fully_covers_duty']=v3.corroborated(hit,spans)
    qualifies=not blockers
    return dict(out,preview_decision='would_qualify_v31' if qualifies else 'still_abstained',
        reasons=blockers or ['explicit_assigned_data_engineering_duty','source_validated_classification','full_source_duty_receipt'],
        duty_evidence=hits,retains_frozen_v3_route=retained,
        evidence_routes=sorted({h['evidence_method'] for h in hits}),
        uses_full_source_beyond_selected_spans=bool(qualifies and not any(h['selected_span_fully_covers_duty'] for h in hits)),
        risks=['Deterministic patterns can mistake supporting or incidental work for the dominant occupation; confidence and quotations do not independently prove correctness.'])
