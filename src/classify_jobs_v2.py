"""Step 3B: evidence-based relevance, then category. No collection or API calls."""
import json
import re
import unicodedata

import pandas as pd

RULE_VERSION = 'step3b-1.1'
CATEGORIES = [
    'Data Engineer', 'Data Analyst', 'Data Scientist', 'ML Engineer', 'AI Engineer',
    'Analytics Engineer', 'BI Developer / BI Analyst', 'Data Warehouse / ETL Developer',
    'Other relevant Data & AI roles', 'Data Governance / Data Quality', 'Data Architect',
    'Statistician', 'AI / ML Research', 'Data / AI Management',
]
RELEVANCES = ['relevant', 'irrelevant', 'uncertain']


def text(value):
    if value is None or pd.isna(value):
        return ''
    value = unicodedata.normalize('NFKC', str(value)).casefold()
    return re.sub(r'\s+', ' ', ''.join(c if c.isalnum() else ' ' for c in value)).strip()


TITLE_RULES = [
    ('T_DE', 'Data Engineer', r'\b(?:data (?:engineers?|eng|engr)|data ?ingenj(?:ör|or)(?:er|en)?|data platform engineers?|dataplattformsutvecklare)\b'),
    ('T_DA', 'Data Analyst', r'\b(?:data analysts?|data ?analytiker|dataanalytikerna)\b'),
    ('T_DS', 'Data Scientist', r'\b(?:data scientists?|datavetare)\b'),
    ('T_ML', 'ML Engineer', r'\b(?:(?:ml|machine learning|deep learning) (?:engineers?|eng|dev|developers?|utvecklare|ingenjör(?:er)?)|maskininlärnings(?:ingenjör(?:er)?|utvecklare)|(?:utvecklare|ingenjör(?:er)?) inom maskininlärning|mlops engineers?)\b'),
    ('T_AI', 'AI Engineer', r'\b(?:(?:ai|genai|gen ai) (?:engineers?|eng|dev|developers?|utvecklare|ingenjör(?:er)?)|(?:utvecklare|ingenjör(?:er)?) (?:inom|av|för) (?:ai|artificiell intelligens))\b'),
    ('T_AE', 'Analytics Engineer', r'\b(?:analytics engineers?|analysutvecklare inom analytics)\b'),
    ('T_BI', 'BI Developer / BI Analyst', r'\b(?:(?:power )?bi ?(?:developers?|analysts?|utvecklare|analytiker|specialists?|konsult(?:er)?)|business intelligence (?:developers?|analysts?|utvecklare|analytiker|specialists?|consultants?))\b'),
    ('T_DWH', 'Data Warehouse / ETL Developer', r'\b(?:(?:data warehouse|etl|elt|dwh|dw) (?:developers?|dev|utvecklare|engineers?)|datavaruhus ?utvecklare|datalagerutvecklare)\b'),
    ('T_GOV', 'Data Governance / Data Quality', r'\b(?:data (?:governance|quality|stewards?|owners?)|master data management|(?:master data|masterdata|mdm|idq|dq) (?:developers?|utvecklare|specialists?|administrators?|konsult(?:er)?)|datakvalitets(?:specialist|utvecklare)|datastyrning)\b'),
    ('T_ARCH', 'Data Architect', r'\b(?:data (?:architects?|arkitekt(?:er)?)|dataarkitekt(?:er)?|(?:ai (?:och|and) data|ai data|ai) (?:lead |solutions? )?(?:architects?|arkitekt(?:er)?)|ai lösningsarkitekt(?:er)?)\b'),
    ('T_STATS', 'Statistician', r'\b(?:biostatistiker|statistiker|biostatisticians?|statisticians?)\b'),
    ('T_MGMT', 'Data / AI Management', r'\b(?:head of (?:data(?! (?:center|centre|protection|privacy|entry))|ai|analytics|machine learning)|chief (?:data|ai|analytics) officer|(?:data(?: science| engineering| analytics)?|ai|analytics|machine learning) (?:managers?|directors?|chefs?|team leads?)|(?:chef|enhetschef|gruppchef|projektledare|project manager) (?:inom|för|for|of) (?:data|ai|analytics|business intelligence))\b'),
    ('T_OTHER', 'Other relevant Data & AI roles', r'\b(?:(?:data|data science|ai|analytics) (?:specialists?|consultants?|konsult(?:er)?|administrators?)|data (?:och|and) analytics consultants?)\b'),
]
RESEARCH_ROLE = r'\b(?:forsk\w*|postdoktor\w*|doktorand\w*|research\w*|phd|doctoral|postdoc\w*)\b'
RESEARCH_DOMAIN = r'\b(?:ai|ml|machine learning|deep learning|maskininlärning\w*|djupinlärning|artificiell intelligens|data science|computer vision|datoriserad bildbehandling)\b'
TITLE_EXCLUSION = r'\b(?:collaborat\w*|samarbet\w*|working with|work with|hire|recruiting|rekrytera)\b'
NEGATIVE_TITLES = [
    ('N_PHYSICAL', r'\b(?:(?:va|vvs|ils|lsa|rf|pcb|cad|kma|ot|scada|process|drift|bygg|maskin|el|mekanik|optronik) ?ingenjör\w*|ingenjör licensiering|(?:mechanical|electrical|civil|hardware|rf) engineers?|nätplanerare|motorprovare|provcellsingenjör)\b'),
    ('N_COMMERCIAL', r'\b(?:säljare|butikssäljare|sales representatives?|field sales|brand managers?|marknads och försäljningskoordinator)\b'),
    ('N_HR_LEGAL', r'\b(?:rekryterare|recruiters?|affärsjurist|jurist|lawyer|legal counsel|arbetsmarknadskoordinator)\b'),
    ('N_FINANCE', r'\b(?:finance director|ekonomichef|redovisningsekonom|löneadministratör)\b'),
    ('N_CARE_SERVICE', r'\b(?:kock|kassapersonal|servitör|specialpedagog|sjuksköterska|undersköterska|förskollärare|lärare|städare)\b'),
]

# Matched within the SAME clause as a candidate duty/required-competence context.
# Specific actions and objects, not isolated skill words or other people's titles.
ACTION = r'(?:build\w*|develop\w*|design\w*|implement\w*|maintain\w*|optimi[sz]\w*|construct\w*|utveckl\w*|bygg\w*|implementer\w*|förvalt\w*|underhåll\w*|design\w*|optimera\w*)'
DESCRIPTION_RULES = [
    ('D_DE_PIPELINE', 'Data Engineer', rf'\b{ACTION}\b.{{0,100}}\b(?:data pipelines?|datapipelines?|data ingestion|dataplattform\w*|data platforms?)\b'),
    ('D_DE_INGEST', 'Data Engineer', r'\b(?:automate data ingestion|integrera data från (?:olika|flera) källor)\b'),
    ('D_DWH', 'Data Warehouse / ETL Developer', rf'\b{ACTION}\b.{{0,100}}\b(?:etl|elt|data warehouse\w*|datavaruhus\w*|datalager)\b'),
    ('D_DA', 'Data Analyst', r'\b(?:sammanställa och analysera statistik|visualisera och presentera data|omvandl\w* data till insikter|analy[sz]\w* (?:customer |business |sales )?data.{0,100}(?:insights|reports|visuali[sz]ations)|analysera data.{0,100}(?:insikter|rapporter|visualiseringar))\b'),
    ('D_BI', 'BI Developer / BI Analyst', rf'\b{ACTION}\b.{{0,90}}\b(?:power bi (?:reports?|dashboards?|rapporter)|bi (?:dashboards?|reports?|rapporter)|business intelligence (?:solutions?|lösningar))\b'),
    ('D_AE', 'Analytics Engineer', rf'\b{ACTION}\b.{{0,80}}\b(?:dbt (?:models?|modeller)|analytics (?:models?|modeller)|analytiska datamodeller)\b'),
    ('D_DS', 'Data Scientist', rf'\b{ACTION}\b.{{0,90}}\b(?:predictive models?|prediktiva modeller|statistical learning models?)\b'),
    ('D_ML', 'ML Engineer', r'\b(?:train\w*|deploy\w*|develop\w*|build\w*|träna\w*|utveckl\w*|produktionssätt\w*)\b.{0,90}\b(?:machine learning models?|ml models?|maskininlärningsmodeller|djupinlärningsmodeller|maskininlärningsalgoritmer|neural networks|neurala nätverk)\b'),
    ('D_RESEARCH_ML', 'AI / ML Research', r'\b(?:testa|utveckla|tillämpa|utveckling av)\b.{0,100}\b(?:maskininlärningsmetoder|metoder för maskininlärning|klassificerare)\b'),
    ('D_AI', 'AI Engineer', rf'\b{ACTION}\b.{{0,90}}\b(?:ai agents?|ai agenter|ai systems?|ai system|llm (?:applications?|applikationer)|rag (?:systems?|pipelines?))\b'),
    ('D_GOV', 'Data Governance / Data Quality', rf'\b{ACTION}\b.{{0,100}}\b(?:data governance|datastyrning|data quality (?:rules?|frameworks?|management)|datakvalitetsregler|informatica idq|masterdataplattform\w*)\b'),
    ('D_ARCH', 'Data Architect', r'\b(?:design\w*|defin\w*|utforma\w*)\b.{0,90}\b(?:data architecture|enterprise data models?|dataarkitektur|arkitekturen för (?:ai och )?data)\b'),
    ('D_STATS', 'Statistician', r'\b(?:perform\w*|conduct\w*|ansvar\w*|utför\w*|genomför\w*)\b.{0,100}\b(?:statistical (?:analysis|modelling|modeling)|statistisk (?:bearbetning|dataanalys|modellering)|regressionsanalys)\b'),
    ('D_MGMT', 'Data / AI Management', r'\b(?:lead|leads|leading|manage|manages|managing|leda|leder|ansvara för)\b.{0,65}\b(?:data (?:engineering |science )?team\w*|ai team\w*|analytics team\w*|data och ai verksamhet\w*)\b'),
]
NEGATIVE_DUTIES = [
    ('D_NONDATA_SOFTWARE', rf'\b{ACTION}\b.{{0,90}}\b(?:mobile apps?|web applications?|mobile applications?|mobilappar|mobila appar|webbapplikationer|backend services|backend tjänster|java applikationer|applikationer i java)\b'),
    ('D_NONDATA_LAB', r'\b(?:syntetisera och karakterisera|syntetisera karakterisera|utföra laboratoriearbete|stödja den löpande laboratorieverksamheten)\b'),
    ('D_NONDATA_SALES', r'\b(?:sälja in våra erbjudanden|boka och genomföra kundmöten|sell (?:our )?products)\b'),
    ('D_NONDATA_CULTURE', r'\b(?:utvecklare inom destination kultur och föreningsliv|planera.{0,50}kulturevenemang)\b'),
]
DUTY_HEAD = r'\b(?:arbetsuppgifter|ansvarsområden|din roll|om rollen|jobbeskrivning|assignment description|responsibilities|your role|what you will do|the role|i rollen|dina uppgifter)\b'
REQUIRED_HEAD = r'\b(?:requirements|qualifications|technical skills|skills and experience|required competencies|kvalifikationer|kravspecifikation|kompetenskrav|skall krav|din profil|vi söker dig|we are looking for)\b'
OPTIONAL_HEAD = r'\b(?:meriterande|nice to have|preferred qualifications|desirable|bonus skills|ett plus|starkt meriterande)\b'
BACKGROUND_HEAD = r'\b(?:om oss|om företaget|about us|about the company|company description|vårt erbjudande|we offer|vi erbjuder|ansökan|how to apply|application process)\b'
ACTOR = r'\b(?:you (?:will|would|are responsible|build|develop|design|maintain|analy[sz]e)|du (?:kommer|ska|skall|ansvarar|utvecklar|bygger|arbetar|leder)|dina (?:arbetsuppgifter|ansvarsområden)|the role involves|rollen innebär|ansvarar du|ansvara för|responsible for)\b'
COMPETENCE = r'\b(?:experience (?:in|with|of)?\s*(?:building|developing|designing|training|deploying)|erfarenhet av att (?:bygga|utveckla|träna|designa)|hands on experience)\b'
INCIDENTAL = r'\b(?:collaborat\w* with|work(?:ing)? (?:closely )?with|samarbet\w* med|i samarbete med|i samråd med|join a team|support(?:ing)? (?:the )?data (?:scientists?|engineers?|science team)|stött\w* (?:data scientists?|data engineers?)|our services|våra tjänster|vi erbjuder tjänster|recruitment process|rekryteringsprocess\w*|rekryteringsverktyg|recruiting tools|ai assisted|ai assisterad|ai stödd|copilot|chatgpt|coding assistants?|kolleg\w* (?:som|inom)|team (?:includes|consists)|our (?:company|team)|vårt (?:företag|team))\b'
NEGATION = r'\b(?:not required|no experience|not responsible|will not|do not|does not|inte|ingen erfarenhet|utan erfarenhet|inget krav)\b'


def compiled(rules):
    return [(identifier, category, re.compile(pattern)) for identifier, category, pattern in rules]


TITLE_COMPILED = compiled(TITLE_RULES)
DESCRIPTION_COMPILED = compiled(DESCRIPTION_RULES)
NEGATIVE_TITLE_COMPILED = [(code, re.compile(pattern)) for code, pattern in NEGATIVE_TITLES]
NEGATIVE_DUTY_COMPILED = [(code, re.compile(pattern)) for code, pattern in NEGATIVE_DUTIES]


def description_evidence(description, research=False):
    """Track section context and match only candidate duties/required experience.

    Each sentence/bullet is independent: company and optional sections are ignored.
    A new explicit candidate duty can escape a background section. Evidence stores
    the original source clause, rule and normalized match for audit.
    """
    positive, negative, ignored = [], [], []
    state = 'unknown'
    for raw_line in re.split(r'[\r\n\u2028\u2029]+', str(description or '')):
        # Keep heading state across bullet lines, and allow inline headings.
        for raw_clause in re.split(r'[•;]|(?<=[.!?])\s+', raw_line):
            clause = text(raw_clause)
            if not clause:
                continue
            headings = []
            for kind, pattern in [('duties', DUTY_HEAD), ('required', REQUIRED_HEAD),
                                  ('optional', OPTIONAL_HEAD), ('background', BACKGROUND_HEAD)]:
                headings.extend((m.start(), m.end(), kind) for m in re.finditer(pattern, clause))
            if headings:
                # The last heading governs the following text in this clause.
                _, end, state = max(headings)
                body = clause[end:]
            else:
                body = clause
            actor = bool(re.search(ACTOR, body))
            competence = bool(re.search(COMPETENCE, body)) and (
                state == 'required' or bool(re.search(
                    r'\b(?:you (?:have|bring|need)|du (?:har|behöver)|is required|krävs)\b', body)))
            imperative = bool(re.match(rf'^(?:{ACTION}|train\w*|deploy\w*|analy[sz]\w*|analysera|visualisera|sammanställa)\b', body.strip()))
            context = ('required_competence' if competence else
                       'responsibilities' if actor or (state == 'duties' and imperative) or
                       (state == 'unknown' and imperative) else None)
            pos = [(code, cat, m) for code, cat, pattern in DESCRIPTION_COMPILED
                   if (research or code != 'D_RESEARCH_ML') and (m := pattern.search(body))]
            neg = [(code, m) for code, pattern in NEGATIVE_DUTY_COMPILED if (m := pattern.search(body))]
            if not pos and not neg:
                continue
            blocked = ('optional_section' if state == 'optional' else
                       'incidental_or_collaboration' if re.search(INCIDENTAL, body) else
                       'negated_evidence' if re.search(NEGATION, body) else
                       'background_or_unscoped' if not context or (state == 'background' and not actor) else None)
            if blocked:
                ignored.append(dict(reason=blocked, source_text=raw_clause.strip()[:900]))
                continue
            for code, category, match in pos:
                positive.append(dict(rule=code, category=category, context=context,
                                     matched_text=match.group(), source_text=raw_clause.strip()))
            for code, match in neg:
                negative.append(dict(rule=code, context=context, matched_text=match.group(),
                                     source_text=raw_clause.strip()))
    return positive, negative, ignored


def select_categories(categories, research=False):
    """Resolve only documented nested patterns, never score unrelated roles."""
    selected = set(categories)
    if research and selected & {'Data Scientist', 'ML Engineer', 'AI Engineer'}:
        selected -= {'Data Scientist', 'ML Engineer', 'AI Engineer'}
        selected.add('AI / ML Research')
    if {'Data Warehouse / ETL Developer', 'Data Engineer'} <= selected:
        selected.remove('Data Engineer')  # Explicit ETL/warehouse is the narrower rule.
    if {'BI Developer / BI Analyst', 'Data Analyst'} <= selected:
        selected.remove('Data Analyst')  # Explicit BI deliverables narrow generic analysis.
    return [category for category in CATEGORIES if category in selected]


def classify_job(title, description=''):
    title_text = text(title)
    original_description = '' if description is None or pd.isna(description) else str(description)
    title_evidence, negative_title = [], []
    for code, category, pattern in TITLE_COMPILED:
        for match in pattern.finditer(title_text):
            if re.search(TITLE_EXCLUSION, title_text[max(0, match.start()-65):match.start()]):
                continue
            title_evidence.append(dict(rule=code, category=category, matched_text=match.group(), source_text=str(title)))
    research = bool(re.search(RESEARCH_ROLE, title_text))
    if research and re.search(RESEARCH_DOMAIN, title_text):
        title_evidence.append(dict(rule='T_RESEARCH', category='AI / ML Research', source_text=str(title)))
    if re.search(r'\bai (?:(?:and|och) )?ml (?:engineers?|developers?|utvecklare|ingenjör(?:er)?)\b', title_text):
        title_evidence.extend(dict(rule='T_AI_ML_COMBINED', category=category, source_text=str(title))
                              for category in ['AI Engineer', 'ML Engineer'])
    for code, pattern in NEGATIVE_TITLE_COMPILED:
        if match := pattern.search(title_text):
            negative_title.append(dict(rule=code, matched_text=match.group(), source_text=str(title)))
    title_categories = select_categories([e['category'] for e in title_evidence], research)
    duties, negative_duties, ignored = description_evidence(original_description, research)
    duty_categories = select_categories([e['category'] for e in duties], research)
    if duty_categories and re.search(r'\b(?:enhetschef|gruppchef|chef|head|manager|director)\b', title_text):
        duties.append(dict(rule='D_MANAGER_SCOPE', category='Data / AI Management',
                           context='title_and_responsibilities', source_text=str(title),
                           matched_text='Management role with concrete Data/AI duties; not a hands-on developer title'))
        duty_categories = ['Data / AI Management']
    review_reasons = []
    # Stage 1: explicit evidence establishes relevance; no-match is always uncertain.
    if title_categories:
        if negative_title:
            relevance, selected = 'uncertain', []
            review_reasons.append('conflicting_relevant_and_unrelated_title')
        else:
            relevance, selected = 'relevant', title_categories
        method = 'title_rule'
    elif duty_categories:
        if negative_title or negative_duties:
            relevance, selected = 'uncertain', []
            review_reasons.append('conflicting_data_and_nondata_duties_or_title')
        else:
            relevance, selected = 'relevant', duty_categories
        method = 'description_rule'
        if not any(e['context'] == 'responsibilities' for e in duties):
            review_reasons.append('required_competence_only')
    elif negative_title:
        relevance, selected, method = 'irrelevant', [], 'title_rule'
    elif negative_duties:
        relevance, selected, method = 'irrelevant', [], 'description_rule'
    else:
        relevance, selected = 'uncertain', []
        method = 'description_rule' if text(original_description) else 'title_rule'
        review_reasons.append('insufficient_role_specific_evidence')
    # Stage 2: incompatible relevant categories retain relevant status but abstain.
    if len(selected) > 1:
        primary_category = None
        review_reasons.append('conflicting_primary_categories')
    else:
        primary_category = selected[0] if selected else None
    if not title_text:
        review_reasons.append('missing_title')
    if not text(original_description):
        review_reasons.append('missing_description')
    if re.search(r'\b(?:datavetare|data ?ingenjör|data ?ingenjor|data administrators?|data specialists?)\b', title_text):
        review_reasons.append('broad_title_alias')
    # An explicit specialist title can still have incompatible stated core duties.
    if title_categories and not negative_title and negative_duties and not duty_categories:
        relevance, primary_category = 'uncertain', None
        review_reasons.append('title_description_conflict')
    evidence = dict(title_matches=title_evidence, description_matches=duties,
                    unrelated_title_matches=negative_title, unrelated_duty_matches=negative_duties,
                    ignored_description_matches=ignored,
                    decision='explicit_evidence' if relevance != 'uncertain' else 'abstained',
                    category_policy='Explicit warehouse/ETL narrows pipeline rules; BI narrows generic analysis; research titles scope AI/ML duties to research. Other multi-category cases abstain.')
    candidates = list(dict.fromkeys(title_categories + duty_categories))
    return dict(relevance=relevance, primary_category=primary_category,
                classification_method=method, evidence=evidence,
                review_required=bool(review_reasons) or relevance == 'uncertain',
                candidate_categories=candidates, review_reasons=review_reasons,
                rule_version=RULE_VERSION)


def classify_frame(frame):
    if any(col in frame for col in ['relevance', 'primary_category', 'step3_classification_method']):
        raise ValueError('Expected original Step 3 input, not already-classified Step 3B data')
    legacy = ['primary_job_category', 'classification_method', 'classification_evidence', 'review_status']
    out = frame.rename(columns={col:'step3_'+col for col in legacy}).copy(deep=True)
    labels = [classify_job(row.job_title, row.description_text) for row in frame.itertuples()]
    for label, row in zip(labels, frame.to_dict('records')):
        for col in ['missing_id', 'missing_title', 'missing_description']:
            if str(row.get(col, '')).casefold() == 'true':
                label['review_reasons'].append(col)
        if 'date_status' in row and row['date_status'] != 'valid':
            label['review_reasons'].append('invalid_or_out_of_window_date')
        label['review_reasons'] = list(dict.fromkeys(label['review_reasons']))
        label['review_required'] = label['review_required'] or bool(label['review_reasons'])
    for col in labels[0] if labels else []:
        out[col] = [label[col] for label in labels]
    return out
