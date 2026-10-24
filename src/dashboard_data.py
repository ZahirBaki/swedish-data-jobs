"""Read-only analytics with distinct-ID denominators for dashboard publications."""
import json
from pathlib import Path

import pandas as pd

REQUESTED = ('required', 'preferred', 'alternative_required')
SKILLS = ('programming_language', 'cloud_platform', 'data_tool')
PERIOD = '21 April – 21 October 2026'


def load_dataset(directory):
    directory = Path(directory)
    jobs = pd.read_json(directory / 'jobs_requirements.jsonl', lines=True, dtype=False)
    jobs['advertisement_id'] = jobs.advertisement_id.astype(str)
    facts = pd.read_csv(directory / 'requirements_evidence.csv', dtype=str,
                        keep_default_na=False, encoding='utf-8-sig')
    validate(jobs, facts)
    return jobs, facts


def validate(jobs, facts):
    if jobs.advertisement_id.duplicated().any():
        raise ValueError('Dataset must contain unique advertisement IDs')
    if jobs.advertisement_id.isna().any() or jobs.advertisement_id.eq('').any():
        raise ValueError('Missing advertisement ID')
    if not set(facts.advertisement_id).issubset(set(jobs.advertisement_id)):
        raise ValueError('Evidence refers to an unknown advertisement')
    if not jobs.ai_relevance.eq('relevant').all() or not jobs.classification_source_valid.eq(True).all():
        raise ValueError('Dashboard accepts only source-validated relevant advertisements')


def filter_data(jobs, facts, **filters):
    """None means no filter; an empty list deliberately selects zero jobs."""
    selected = jobs.copy()
    for column, values in filters.items():
        if values is not None and column == 'analysis_opportunity_types':
            selected = selected[selected[column].map(lambda types: bool(set(types or ['not_flagged_opportunity']).intersection(values)))]
        elif values is not None:
            selected = selected[selected[column].fillna('not_specified').replace('', 'not_specified').isin(values)]
    return selected, facts[facts.advertisement_id.isin(selected.advertisement_id)].copy()


def count_ads(frame, columns, denominator):
    columns = [columns] if isinstance(columns, str) else list(columns)
    result = frame.groupby(columns, dropna=False).advertisement_id.nunique().reset_index(name='job_count')
    result['denominator_jobs'] = denominator
    result['percent'] = result.job_count / denominator * 100 if denominator else 0.0
    return result.sort_values('job_count', ascending=False).reset_index(drop=True)


def skill_counts(jobs, facts, kinds=SKILLS, modalities=REQUESTED):
    selected = facts[facts.kind.isin(kinds) & facts.modality.isin(modalities)]
    # A job mentioning a skill several times, or under two modalities, counts once.
    return count_ads(selected, ['kind', 'value'], jobs.advertisement_id.nunique())


def coverage(jobs, facts):
    total = jobs.advertisement_id.nunique()
    qualified = facts[facts.modality.isin((*REQUESTED, 'explicitly_not_required'))]
    groups = dict(programming_languages=['programming_language'],
                  cloud_platforms=['cloud_platform'], data_tools=['data_tool'],
                  swedish=['language'], english=['language'], education=['education'],
                  professional_experience=['experience'])
    result = []
    for name, kinds in groups.items():
        subset = qualified[qualified.kind.isin(kinds)]
        if name in ('swedish', 'english'):
            # Job-level status resolves conflicts conservatively to missing evidence.
            count = jobs.loc[jobs[name + '_requirement'].ne('not_specified'), 'advertisement_id'].nunique()
        else:
            count = subset.advertisement_id.nunique()
        result.append(dict(requirement_type=name, job_count=int(count), denominator_jobs=int(total),
                           percent=100 * count / total if total else 0.0))
    for name, column, missing in [('seniority', 'seniority', 'unspecified'),
                                  ('region', 'region', ''), ('municipality', 'municipality', '')]:
        count = jobs.loc[jobs[column].fillna('').ne(missing) & jobs[column].notna(), 'advertisement_id'].nunique()
        result.append(dict(requirement_type=name, job_count=int(count), denominator_jobs=int(total),
                           percent=100 * count / total if total else 0.0))
    for name in ('employment_type', 'working_time', 'work_arrangement'):
        count = facts.loc[facts.kind.eq(name) & facts.modality.eq('explicit'), 'advertisement_id'].nunique()
        result.append(dict(requirement_type=name, job_count=int(count), denominator_jobs=int(total),
                           percent=100 * count / total if total else 0.0))
    return pd.DataFrame(result)


def language_counts(jobs, language):
    column = language + '_requirement'
    result = count_ads(jobs, column, jobs.advertisement_id.nunique())
    result['denominator_specified_jobs'] = jobs.loc[jobs[column].ne('not_specified'), 'advertisement_id'].nunique()
    return result


def summary(jobs, facts):
    return dict(analyzed_jobs=int(jobs.advertisement_id.nunique()), collection_period=PERIOD,
                usable_requirements_jobs=int(jobs.loc[jobs.requirements_data_usable.eq(True), 'advertisement_id'].nunique()),
                coverage=coverage(jobs, facts).to_dict('records'),
                categories=count_ads(jobs, 'ai_primary_category', len(jobs)).to_dict('records'),
                scopes=count_ads(jobs, 'reporting_scope', len(jobs)).to_dict('records'),
                requested_skills=skill_counts(jobs, facts).to_dict('records'),
                api_requests=0, api_cost_usd=0,
                limitations='Selected AI-classified, not independently human-verified subset. Not representative of Swedish market demand. Missing evidence is not absence of a requirement. Distinct advertisement IDs may still describe repeated postings.')


def _completed_publications(root, marker_name, stage):
    """Ignore incomplete publications without disturbing valid historical data."""
    candidates = []
    for marker in root.glob(f'*/{marker_name}'):
        directory = marker.parent
        manifest = directory / 'manifest.json'
        try:
            metadata = json.loads(manifest.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        required = ('jobs_requirements.jsonl', 'requirements_evidence.csv')
        if metadata.get('stage') == stage and all((directory / name).is_file() for name in required):
            candidates.append(directory)
    return candidates


def latest_dataset(root):
    """Prefer completed Step 7 publications; resolve afresh on each app rerun."""
    root = Path(root)
    candidates = _completed_publications(root, 'step7_summary.json', 'step7-ai-relevant')
    if not candidates:
        candidates = _completed_publications(root, 'step6_summary.json', 'step6-combined')
    if not candidates:
        return root / 'step5-available108-20261023'
    return max(candidates, key=lambda path: (path.joinpath('manifest.json').stat().st_mtime_ns, str(path)))


def dashboard_dataset(root, override=None):
    """Prefer an explicit path, then local research, then the fictional demo.

    An invalid explicit path is returned unchanged so the UI can explain the
    configuration error rather than silently showing a different dataset.
    """
    if override:
        return Path(override).expanduser()
    latest = latest_dataset(root)
    if all((latest / filename).is_file() for filename in
           ('jobs_requirements.jsonl', 'requirements_evidence.csv')):
        return latest
    return Path(__file__).resolve().parents[1] / 'examples' / 'demo_dataset'
