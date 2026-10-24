"""Step 3C reporting scopes; never changes the Step 3B predictions."""
import json
import re

from src.classify_jobs_v2 import text

CORE_CATEGORIES = [
    'Data Engineer', 'Data Analyst', 'Data Scientist', 'ML Engineer', 'AI Engineer',
    'Analytics Engineer', 'BI Developer / BI Analyst', 'Data Warehouse / ETL Developer',
    'Data Architect', 'Data Governance / Data Quality',
]
EXTENDED_SCOPES = {'AI / ML Research':'ai_ml_research', 'Statistician':'statistician',
                   'Data / AI Management':'data_ai_management'}
POSITION_RULES = {
    'doctoral': r'\b(?:doktorand\w*|doctoral (?:position|student|candidate|fellow)\w*|phd (?:student|candidate|position|fellow)\w*)\b',
    'internship': r'\b(?:intern|interns|internship\w*|praktikant\w*|praktik|lia)\b',
    'thesis': r'\b(?:thesis|examensarbete|exjobb)\b',
    'postdoctoral': r'\b(?:postdoktor\w*|postdoc\w*|post doctoral)\b',
    'student_work': r'\b(?:studentarbetare|studentmedarbetare|working student|student assistant)\b',
}


def policy_flags(title, relevance, category):
    normalized = text(title)
    flags = {'is_'+name:bool(re.search(pattern, normalized)) for name,pattern in POSITION_RULES.items()}
    if relevance != 'relevant':
        scope = relevance
    elif category in CORE_CATEGORIES:
        scope = 'core_industry'
    else:
        scope = EXTENDED_SCOPES.get(category, 'other_relevant' if category else 'relevant_unresolved_category')
    return dict(role_scope=scope, **flags,
                core_dashboard_candidate=scope == 'core_industry' and not any(flags.values()),
                policy_flag_evidence=json.dumps([name for name,pattern in POSITION_RULES.items()
                                                if re.search(pattern,normalized)],ensure_ascii=False))


def add_policy_flags(frame):
    out = frame.copy(deep=True)
    flags = [policy_flags(row.job_title,row.relevance,row.primary_category) for row in frame.itertuples()]
    for col in flags[0] if flags else []:
        out[col] = [value[col] for value in flags]
    return out
