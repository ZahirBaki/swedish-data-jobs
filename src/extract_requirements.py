"""Conservative deterministic requirements extraction with exact source offsets.

No text rewriting, external model, credentials or network. Ambiguous clauses are
retained for inspection rather than silently interpreted as mandatory requirements.
"""
import hashlib
import json
import re

VERSION='requirements-rules-1.0'
SKILLS={
    'programming_language':{
        'Python':r'\bPython\b','SQL':r'\bSQL\b(?!\s+Server\b)',
        'R':r'\bR\b(?!\s*&\s*D)','Java':r'\bJava\b','JavaScript':r'\bJavaScript\b',
        'TypeScript':r'\bTypeScript\b','Scala':r'\bScala\b','C++':r'(?<!\w)C\+\+(?!\w)',
        'C#':r'(?<!\w)C\#(?!\w)','C':r'\bC\b(?!\s*[+#])',
        'Go':r'\bGolang\b|(?-i:\bGo)(?=\s*(?:programming|language|developer|[,/)\n]|$))',
        'Rust':r'\bRust\b','Julia':r'\bJulia\b','MATLAB':r'\bMATLAB\b',
        'Bash':r'\bBash\b','SAS':r'\bSAS\b','Kotlin':r'\bKotlin\b','Perl':r'\bPerl\b'},
    'cloud_platform':{
        'Azure':r'\b(?:Microsoft\s+)?Azure\b','AWS':r'\bAWS\b|\bAmazon Web Services\b',
        'Google Cloud':r'\bGCP\b|\bGoogle Cloud(?: Platform)?\b'},
    'data_tool':{
        'Databricks':r'\bDatabricks\b','Apache Spark':r'\b(?:Apache\s+)?Spark\b',
        'PySpark':r'\bPySpark\b','Snowflake':r'\bSnowflake\b','dbt':r'\bdbt(?:\s+Core)?\b',
        'Power BI':r'\bPower\s*BI\b','Tableau':r'\bTableau\b','Apache Airflow':r'\b(?:Apache\s+)?Airflow\b',
        'Apache Kafka':r'\b(?:Apache\s+)?Kafka\b','Hadoop':r'\bHadoop\b',
        'Docker':r'\bDocker\b','Kubernetes':r'\bKubernetes\b|\bK8s\b',
        'Git':r'\bGit\b','Terraform':r'\bTerraform\b','Azure Data Factory':r'\bAzure Data Factory\b|\bADF\b',
        'Azure Synapse':r'\b(?:Azure\s+)?Synapse\b','BigQuery':r'\bBigQuery\b',
        'Amazon Redshift':r'\b(?:Amazon\s+)?Redshift\b','SQL Server':r'\b(?:Microsoft\s+)?SQL Server\b',
        'PostgreSQL':r'\bPostgreSQL\b|\bPostgres\b','pandas':r'\bpandas\b','NumPy':r'\bNumPy\b',
        'scikit-learn':r'\bscikit[- ]learn\b','PyTorch':r'\bPyTorch\b','TensorFlow':r'\bTensorFlow\b',
        'MLflow':r'\bMLflow\b','.NET':r'(?<!\w)\.NET\b'},
}
REQ=re.compile(r'\brequire(?:d|s)?\b|\bmandatory\b|\bmust\b|\bessential\b|\bkrav\b|\bmåste\b|\bkrävs\b|\bkräver\b|\b(?:ska|skall) ha\b|\byou (?:have|bring|are|can|need)\b|\bdu (?:har|behärskar|kan|är)\b',re.I)
PREF=re.compile(r'\bpreferred\b|\bnice[- ]to[- ]have\b|\bdesirable\b|\badvantage\b|\ba plus\b|\bbonus\b|\bmeriterande\b|\bönskvärt\b|\bönskvärd\b|\bgärna\b|\bett plus\b',re.I)
NOT=re.compile(r'\bnot (?:required|necessary|mandatory|needed)\b|\bno\s+(?:\w+\s+){0,6}required\b|\binget krav\b|\binte (?:ett )?krav\b|\bkrävs inte\b|\bbehöver inte\b|\boptional\b',re.I)
COMPETENCE=re.compile(r'\bexperience (?:with|in|of)\b|\bknowledge of\b|\bproficien\w*\b|\bexpertise\b|\berfarenhet (?:av|inom)\b|\bkunskap\w* (?:i|inom|om)\b',re.I)
INCIDENTAL=re.compile(r'\bwe (?:offer|use)\b|\bour (?:team|stack)\b|\bvi (?:erbjuder|använder)\b|\bvår teknikstack\b|\bteamet använder\b|\bgain experience\b',re.I)
HEADINGS={
    'required':r'(?:requirements?|required(?: (?:skills|qualifications))?|qualifications|kvalifikationer|kvalifikationskrav|kravspecifikation|krav(?: för anställningen)?|baskrav|your background|din profil|who you are|what (?:we[’\x27]re|we are) looking for|vi söker dig som|du har)',
    'preferred':r'(?:preferred(?: (?:skills|qualifications))?|nice[- ]to[- ]have|meriterande(?: för anställningen)?|önskvärt|det är meriterande)',
    'alternative_required':r'(?:(?:experience with|erfarenhet av) (?:one or more|several|some|några|flera) (?:of )?(?:the )?following)',
    'mentioned_only':r'(?:responsibilities|what you will do|arbetsuppgifter(?: och ansvarsområden)?|om rollen|about (?:us|the role)|om oss|vi erbjuder|what we offer|why join us|om företaget|about the company|benefits)',
}
LANGUAGES={'Swedish':r'\bSwedish\b|\bsvenska\b','English':r'\bEnglish\b|\bengelska\b'}
ABILITY=re.compile(r'\bfluent\b|\bfluency\b|\bspeak\w*\b|\bspoken\b|\bwritten\b|\bcommunicat\w*\b|\bproficien\w*\b|\bbehärsk\w*\b|\btala\b|\btal och skrift\b|\bflytande\b|\bgoda? (?:kunskaper|förmåga)\b|\bkunskaper i\b',re.I)
EDUCATION={
    'bachelor':r"\bbachelor(?:'s|’s|s)?\b|\bB\.?Sc\.?\b|\bkandidatexamen\b|\bkandidatnivå\b|\bhögskoleingenjör\w*\b",
    'master':r"\bmaster(?:'s|’s|s)?\b|\bM\.?Sc\.?\b|\bmasterexamen\b|\bmagisterexamen\b|\bcivilingenjör\w*\b",
    'phd':r'\bPh\.?D\.?\b|\bdoctorate\b|\bdoktorsexamen\b',
    'degree_unspecified':r'\b(?:university|academic|relevant|higher education) degree\b|\bdegree\b(?!\s+of\b|\s+(?:Celsius|Fahrenheit)\b)|\bakademisk examen\b|\bhögskoleexamen\b|\buniversitetsexamen\b',
}
NUMBER_WORDS={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
    'ett':1,'en':1,'två':2,'tre':3,'fyra':4,'fem':5,'sex':6,'sju':7,'åtta':8,'nio':9,'tio':10}
NUMBER=r'(?:\d{1,2}|'+('|'.join(NUMBER_WORDS))+r')'
YEARS=re.compile(r'(?<!\w)(?P<low>'+NUMBER+r')\s*(?:(?P<range>[-–—]|to|till)\s*(?P<high>'+NUMBER+r'))?\s*(?P<plus>\+)?\s*(?:years?|års?|år)[’\x27]?(?:\s+(?:of|av))?\s*(?:(?:professional|work|relevant|commercial|software|development|data|engineering|industry|hands-on|research|yrkes|praktisk)\s+){0,4}(?:experience|erfarenhet|arbetslivserfarenhet)',re.I)
QUALIFIED={'required','preferred','alternative_required','explicitly_not_required'}


def units(text):
    section=None;heading=None
    for line in re.finditer(r'[^\r\n\u2028\u2029]+',text):
        raw=line.group();events=[]
        for kind,pattern in HEADINGS.items():
            # Collected HTML frequently glues a heading onto the preceding sentence
            # or the next word (RequirementsStrong...). Match those boundaries
            # without normalizing or moving any original character.
            for m in re.finditer(r'(?<!\w)(?i:'+pattern+r')(?=\s*:|\s*$|(?-i:[A-ZÅÄÖ]))\s*:?',raw):
                prefix=raw[:m.start()].strip()
                if m.group().strip().casefold() in {'required','preferred','krav','meriterande','du har'} and prefix and prefix[-1] not in '.!?':continue
                events.append((m.start(),m.end(),kind))
        events=sorted(events,key=lambda x:(x[0],-(x[1]-x[0])))
        spans=[];cursor=0
        for a,b,kind in events:
            if a<cursor:continue
            if a>cursor:spans.append((cursor,a,section,heading))
            section=kind;heading=(line.start()+a,line.start()+b);cursor=b
        if cursor<len(raw):spans.append((cursor,len(raw),section,heading))
        for a,b,context,context_heading in spans:
            for piece in re.finditer(r'(?:\.(?=\w)|[^;.!?])+(?:[.!?](?!\s|$)[^;.!?]*)*[;.!?]?(?:\s+|$)',raw[a:b]):
                start=line.start()+a+piece.start();end=line.start()+a+piece.end()
                while start<end and text[start].isspace():start+=1
                while end>start and text[end-1].isspace():end-=1
                if start<end:yield dict(start=start,end=end,text=text[start:end],section=context,heading=context_heading)


def modality(unit,hit_start,hit_end,force_competence=False):
    text=unit['text'];cues=[]
    negatives=list(NOT.finditer(text))
    for kind,pattern in [('explicitly_not_required',NOT),('preferred',PREF),('required',REQ)]:
        for m in pattern.finditer(text):
            if kind=='required' and any(a.start()<=m.start()<a.end() for a in negatives):continue
            distance=max(m.start()-hit_end,hit_start-m.end(),0)
            cues.append((distance,{'explicitly_not_required':0,'preferred':1,'required':2}[kind],kind))
    if cues:return min(cues)[2],'explicit_clause_cue'
    if INCIDENTAL.search(text):return 'mentioned_only','employer_or_training_context'
    if unit['section'] in {'required','preferred','alternative_required'}:return unit['section'],'qualification_heading'
    if force_competence or COMPETENCE.search(text):return 'ambiguous','competence_without_clear_requirement'
    return 'mentioned_only','no_qualification_cue'


def extract_job(job):
    identifier=str(job['advertisement_id']);text=job.get('description_text','') or '';title=job.get('job_title','') or ''
    records=[]
    def add(kind,value,start,end,unit,modality_value,rule,**extra):
        source_field=extra.pop('source_field','description_text');source=job.get(source_field,'') or ''
        a,b=unit['start'],unit['end'];heading=unit.get('heading')
        record=dict(advertisement_id=identifier,kind=kind,value=value,modality=modality_value,
            source_field=source_field,match_start=start,match_end=end,matched_text=source[start:end],
            evidence_start=a,evidence_end=b,evidence_text=source[a:b],rule_id=rule,extractor_version=VERSION,
            context_heading_start=heading[0] if heading else None,context_heading_end=heading[1] if heading else None,
            context_heading_text=source[heading[0]:heading[1]] if heading else None,
            extraction_review_required=modality_value in {'ambiguous','alternative_required'} or rule=='qualification_heading',**extra)
        record['requirement_id']=hashlib.sha256(json.dumps([identifier,kind,value,start,end,modality_value],ensure_ascii=False).encode('utf-8')).hexdigest()[:24]
        records.append(record)
    for unit in units(text):
        body=unit['text'];offset=unit['start']
        skills=[]
        for kind,catalog in SKILLS.items():
            for name,pattern in catalog.items():
                for m in re.finditer(pattern,body,re.I):skills.append((kind,name,m))
        skill_alternative=bool(re.search(r'\b(?:or|eller)\b',body,re.I) and len({name for _,name,_ in skills})>1)
        for kind,name,m in skills:
            mode,rule=modality(unit,m.start(),m.end())
            if skill_alternative and mode=='required':mode='alternative_required'
            add(kind,name,offset+m.start(),offset+m.end(),unit,mode,rule,alternative_group=skill_alternative)
        for language,pattern in LANGUAGES.items():
            for m in re.finditer(pattern,body,re.I):
                mode,rule=modality(unit,m.start(),m.end())
                # The ad language, application language and colleagues' nationality
                # do not establish applicant language requirements.
                irrelevant_context=re.search(r'(?:application|ansök\w*|cv|cover letter|annons|advertisement|speaking team|talande team)',body,re.I)
                direct_language_requirement=bool(re.search(r'(?:Swedish|English|svenska|engelska)\s+(?:language\s+)?(?:is\s+|är\s+)?(?:required|mandatory|ett krav|krävs)',body,re.I)
                    or re.search(r'(?:require|requires|mandatory)\s+(?:Swedish|English|svenska|engelska)',body,re.I))
                clear=(ABILITY.search(body) or direct_language_requirement or mode in {'preferred','explicitly_not_required'})
                if irrelevant_context or not clear:mode,rule='mentioned_only','not_applicant_language_qualification'
                elif mode in {'ambiguous','mentioned_only'} and re.search(r'^\s*(?:fluent|fluency|flytande)\b',body,re.I):
                    mode,rule='required','explicit_fluency_statement'
                if re.search(r'\b(?:or|eller)\b',body,re.I) and all(re.search(p,body,re.I) for p in LANGUAGES.values()) and mode=='required':
                    mode='alternative_required'
                add('language',language,offset+m.start(),offset+m.end(),unit,mode,rule)
        levels=[]
        for level,pattern in EDUCATION.items():
            for m in re.finditer(pattern,body,re.I):levels.append((level,m))
        for level,m in levels:
            if level=='degree_unspecified' and any(x!='degree_unspecified' for x,_ in levels):continue
            mode,rule=modality(unit,m.start(),m.end())
            if re.search(r'(?:thesis|examensarbete|PhD student|PhD candidate|doctoral position|supervis\w*|doktorand)',body,re.I) and not re.search(r'(?:completed|hold|degree|examen|avlagt)',body,re.I):
                mode,rule='mentioned_only','position_or_supervision_not_degree_requirement'
            alternative=bool(re.search(r'\b(?:or|eller)\b',body,re.I) and (len({l for l,_ in levels})>1 or re.search(r'equivalent (?:(?:work|professional|relevant) )?experience|motsvarande (?:arbetslivs)?erfarenhet',body,re.I)))
            if alternative and mode=='required':mode='alternative_required'
            add('education',level,offset+m.start(),offset+m.end(),unit,mode,rule,alternative_group=alternative)
        for m in YEARS.finditer(body):
            def number(value):return int(value) if value.isdigit() else NUMBER_WORDS[value.casefold()]
            low=number(m['low']);high=number(m['high']) if m['high'] else None
            if high is not None and high<low:continue
            mode,rule=modality(unit,m.start(),m.end(),True)
            explicit_min=bool(m['plus'] or re.search(r'(?:at least|minimum|minst|min\.?|minst cirka)\s*$',body[:m.start()],re.I))
            bound='range' if high is not None else 'at_least' if explicit_min else 'stated_duration'
            add('experience','professional_experience',offset+m.start(),offset+m.end(),unit,mode,rule,
                stated_years=low,maximum_years=high,minimum_years=low if explicit_min or high is not None else None,bound_type=bound)
    # Seniority is explicit own-role wording, never inferred from years/age or colleagues.
    seniority=[]
    for level,pattern in [('lead',r'\b(?:lead|teamledare|tech lead|ledande)\b'),('senior',r'\b(?:senior\w*|sr\.?)\b'),
        ('mid_level',r'\b(?:mid[- ]level|intermediate|midlevel)\b'),('junior',r'\b(?:junior\w*|entry[- ]level)\b')]:
        m=re.search(pattern,title,re.I)
        if m:
            add('seniority',level,m.start(),m.end(),dict(start=0,end=len(title),heading=None),'explicit','title_seniority',source_field='job_title')
            seniority.append(level)
    if not seniority:
        for unit in units(text):
            m=re.search(r'(?:we (?:are looking for|seek)|vi söker|as (?:a|an)|rollen som)\s+(?:(?:a|an|en)\s+)?(?P<level>junior|senior|lead|mid[- ]level)\s+(?:data|ml|ai|machine learning|analyst|engineer|utvecklare|analytiker|scientist)',unit['text'],re.I)
            if m:
                value=m['level'].casefold().replace('-','_').replace(' ','_')
                add('seniority',value,unit['start']+m.start('level'),unit['start']+m.end('level'),unit,'explicit','own_role_seniority')
                seniority.append(value);break
    employment_patterns={
        'employment_type':{'permanent':r'\btillsvidare\w*\b|\bpermanent (?:position|employment|contract)\b',
            'fixed_term':r'\bvisstids\w*\b|\btidsbegränsad\w*\b|\bfixed[- ]term\b',
            'substitute':r'\bvikariat\w*\b'},
        'working_time':{'full_time':r'\bheltid\b|\bfull[- ]time\b','part_time':r'\bdeltid\b|\bpart[- ]time\b'},
        'work_arrangement':{'hybrid':r'\bhybrid(?:arbete| working| work| role)?\b',
            'remote':r'\bfully remote\b|\b100\s*%\s*remote\b|\bhelt på distans\b',
            'remote_possible':r'\bmöjlighet (?:att|till) (?:arbeta )?(?:på )?distans\b|\bmöjlighet att arbeta hemifrån\b',
            'onsite':r'\bon[- ]site\b|\bpå plats\b'},
    }
    for unit in units(text):
        for kind,values in employment_patterns.items():
            for value,pattern in values.items():
                for m in re.finditer(pattern,unit['text'],re.I):
                    negated=bool(re.search(r'(?:not|inte|ej)\s*$',unit['text'][:m.start()],re.I))
                    not_work_location=value=='hybrid' and not re.search(r'work|working|role|position|arrangement|remote|office|on-site|distans|arbet|kontor|tjänst|plats|location',unit['text'],re.I)
                    add(kind,value,unit['start']+m.start(),unit['start']+m.end(),unit,
                        'mentioned_only' if not_work_location else 'explicitly_not_required' if negated else 'explicit','employment_statement')
    for field in ['municipality','region','employer','publication_date','number_of_vacancies']:
        value=job.get(field,'') or ''
        if value:add('source_metadata',field,0,len(value),dict(start=0,end=len(value),heading=None),
            'provided_metadata','original_source_field',source_field=field,metadata_value=value)
    # Every record has auditable, exact offsets, even for negated/mentioned entities.
    for r in records:
        source=job.get(r['source_field'],'') or ''
        if source[r['evidence_start']:r['evidence_end']]!=r['evidence_text'] or source[r['match_start']:r['match_end']]!=r['matched_text']:
            raise ValueError('Requirement evidence validation failed')
    unique={r['requirement_id']:r for r in records};records=list(unique.values())
    result=dict(job,requirements_extractor_version=VERSION,requirements_status='extracted' if text.strip() else 'missing_description')
    skills=[r for r in records if r['kind'] in SKILLS]
    for mode in ['required','preferred','alternative_required','mentioned_only']:
        result['skills_'+mode]=sorted({r['value'] for r in skills if r['modality']==mode})
    for language in ['Swedish','English']:
        found=[r for r in records if r['kind']=='language' and r['value']==language and r['modality'] in QUALIFIED]
        modes={r['modality'] for r in found};review=len(modes)>1 or 'alternative_required' in modes or any(r['kind']=='language' and r['value']==language and r['modality']=='ambiguous' for r in records)
        result[language.lower()+'_requirement']=next(iter(modes)) if len(modes)==1 and not review else 'not_specified'
        result[language.lower()+'_review_required']=review
    education=[r for r in records if r['kind']=='education' and r['modality'] in {'required','preferred','alternative_required'}]
    levels=sorted({r['value'] for r in education})
    result['education_levels']=levels or ['not_specified']
    result['education_level']=levels[0] if len(levels)==1 else 'multiple_levels' if levels else 'not_specified'
    result['education_modalities']=sorted({r['modality'] for r in education}) or ['not_specified']
    experience=[{k:r.get(k) for k in ['stated_years','minimum_years','maximum_years','bound_type','modality','requirement_id']}
        for r in records if r['kind']=='experience' and r['modality'] in {'required','preferred','alternative_required'}]
    result['experience_requirements']=experience
    result['experience_status']='specified' if experience else 'not_specified'
    result['seniority']=seniority[0] if seniority else 'unspecified'
    for kind in employment_patterns:
        result[kind]=sorted({r['value'] for r in records if r['kind']==kind and r['modality']=='explicit'}) or ['not_specified']
    meaningful=[r for r in records if r['kind'] in set(SKILLS)|{'language','education','experience'} and r['modality'] in QUALIFIED]
    result['requirements_data_usable']=bool(meaningful)
    result['requirements_review_required']=any(r['extraction_review_required'] for r in records)
    result['requirements_evidence_count']=len(records)
    return result,records
