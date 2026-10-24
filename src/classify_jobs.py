"""Explainable, conservative role rules and reproducible review artifacts.

Run with python -m src.classify_jobs. No collection/API dependencies.
"""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.clean_jobs import ROOT, clean_jobs, latest_collection, load_jobs, normalize_text

UNCERTAIN = 'Uncertain / Needs Review'
IRRELEVANT = 'Irrelevant'
OTHER = 'Other relevant Data & AI roles'
# Boundaries prevent AI in ordinary words and BI in unrelated abbreviations.
RULES = [
    ('Data Engineer', r'\b(?:data[ -]engineers?|dataingenj[oö]r|dataplattformsutvecklare|data platform engineer)\b'),
    ('Data Analyst', r'\b(?:data[ -]analysts?|dataanalytiker)\b'),
    ('Data Scientist', r'\b(?:data scientists?|datavetare)\b'),
    ('ML Engineer', r'\b(?:(?:machine learning|ml)[ -](?:engineer|developer|utvecklare)|maskininlärnings(?:ingenjör|utvecklare))\b'),
    ('AI Engineer', r'\b(?:ai[ -](?:engineer|developer|utvecklare|ingenjör)|ingenjör inom artificiell intelligens)\b'),
    ('Analytics Engineer', r'\banalytics[ -]engineer\b'),
    ('BI Developer / BI Analyst', r'\b(?:(?:power )?bi[ -]?(?:developer|analyst|utvecklare|analytiker)|business intelligence (?:developer|analyst|utvecklare|analytiker))\b'),
    ('Data Warehouse / ETL Developer', r'\b(?:(?:data warehouse|etl)[ -]?(?:developer|utvecklare)|datavaruhusutvecklare)\b'),
    (OTHER, r'\b(?:data (?:architect|arkitekt|steward|scientific researcher|specialist)|dataarkitekt|(?:ai|ml|data science)[ -](?:researcher|forskare|konsult|specialist|architect)|mlops engineer)\b'),
]
CATEGORIES = [category for category, _ in RULES] + [IRRELEVANT, UNCERTAIN]
DOMAIN = r'\b(?:data|analytics|analys|bi|business intelligence|etl|datavaruhus|ai|ml|machine learning|maskininlärning|artificiell intelligens)\b'
AMBIGUOUS = r'\b(?:engineer|ingenjör|utvecklare|developer|analyst|analytiker|forskare|researcher|doktorand|postdoktor|konsult|specialist)\b'
# Clear unrelated occupations can be excluded despite incidental AI mentions.
UNRELATED = r'\b(?:säljare|kock|specialpedagog|sjuksköterska|undersköterska|kassapersonal|servitör|förskollärare|lärare|städare|butikssäljare)\b'
RULE_VERSION = '1.0'


def classify_job(title, description=''):
    title, description = normalize_text(title), normalize_text(description)
    def result(category, method, evidence, review):
        return dict(primary_job_category=category, classification_method=method,
                    classification_evidence=evidence, review_status=review)
    if not title:
        return result(UNCERTAIN, 'missing_title', 'No usable job title', 'needs_review')
    hits = [(category, match.group()) for category, pattern in RULES
            if (match := re.search(pattern, title))]
    if hits and re.search(UNRELATED, title):
        return result(UNCERTAIN, 'conflicting_title', f'Role and unrelated occupation in title: {hits}', 'needs_review')
    if len(hits) == 1:
        category, evidence = hits[0]
        return result(category, 'title_rule', f'{category}: title matched {evidence!r}', 'pending_validation')
    if len(hits) > 1:
        return result(UNCERTAIN, 'conflicting_title', f'Multiple title categories: {hits}', 'needs_review')
    if re.search(UNRELATED, title):
        return result(IRRELEVANT, 'unrelated_title', 'Explicit unrelated occupation in title', 'pending_validation')
    if re.search(DOMAIN, title):
        return result(UNCERTAIN, 'ambiguous_title', 'Data/AI domain in title without a clear role rule', 'needs_review')
    if re.search(AMBIGUOUS, title):
        evidence = [(cat, m.group()) for cat, pattern in RULES if (m := re.search(pattern, description))]
        return result(UNCERTAIN, 'description_support' if evidence else 'ambiguous_title',
                      f'Ambiguous title; description role mentions: {evidence}; mentions do not establish duties', 'needs_review')
    if not description or re.search(DOMAIN, description):
        return result(UNCERTAIN, 'insufficient_evidence', 'Unrecognized title; missing description or domain mention requires review', 'needs_review')
    return result(IRRELEVANT, 'no_role_evidence', 'No title rule or Data/AI domain evidence', 'pending_validation')


def classify_jobs(frame):
    out = frame.copy(deep=True)
    labels = pd.DataFrame([classify_job(t, d) for t, d in zip(out.job_title, out.description_text)], index=out.index)
    for col in labels:
        out[col] = labels[col]
    quality = out.missing_title | out.missing_description | out.missing_id | ~out.in_analysis_window
    out.loc[quality, 'review_status'] = 'needs_review'
    return out


def review_sample(frame, size=150, seed=42):
    """Random balanced strata; small strata are censused, remaining quota redistributed."""
    if size < 100:
        raise ValueError('Request at least 100 review rows')
    target = min(size, len(frame))
    shuffled = frame.sample(frac=1, random_state=seed)
    groups = [group for _, group in shuffled.groupby('primary_job_category', sort=True)]
    indices = []
    depth = 0
    while len(indices) < target:
        for group in groups:
            if depth < len(group) and len(indices) < target:
                indices.append(group.index[depth])
        depth += 1
    sample = frame.loc[indices].copy()
    sample['category_population_size'] = sample.primary_job_category.map(frame.primary_job_category.value_counts())
    sample['category_sample_size'] = sample.primary_job_category.map(sample.primary_job_category.value_counts())
    sample['sampling_seed'] = seed
    sample['correct_category'] = ''
    sample['review_notes'] = ''
    return sample


def evaluate_review(review):
    """Sample metrics only; incomplete labels are excluded, unknown labels rejected."""
    labels = review.correct_category.fillna('').str.strip()
    if invalid := set(labels) - set(CATEGORIES) - {''}:
        raise ValueError(f'Unknown manual categories: {sorted(invalid)}')
    reviewed = review.loc[labels.ne('')].copy()
    reviewed['correct_category'] = labels.loc[reviewed.index]
    if reviewed.empty:
        return {'reviewed': 0, 'accuracy': None, 'note': 'No human labels supplied'}, pd.DataFrame()
    confusion = pd.crosstab(reviewed.correct_category, reviewed.primary_job_category,
                            rownames=['Human label'], colnames=['Predicted']).reindex(index=CATEGORIES, columns=CATEGORIES, fill_value=0)
    return {'reviewed': len(reviewed), 'sample_size': len(review),
            'accuracy': float(reviewed.correct_category.eq(reviewed.primary_job_category).mean()),
            'note': 'Accuracy on labeled review sample only; balanced stratification is not population representative'}, confusion


def run_pipeline(collection=None, output=None, sample_size=150, seed=42):
    collection = Path(collection) if collection else latest_collection()
    source = collection / 'jobs.csv'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    config = json.loads((collection / 'configuration.json').read_text(encoding='utf-8'))
    raw = load_jobs(source)
    cleaned = clean_jobs(raw, config['start'], config['end'])
    classified = classify_jobs(cleaned)
    eligible = classified.loc[classified.in_analysis_window]
    counts = eligible.primary_job_category.value_counts().reindex(CATEGORIES, fill_value=0)
    summary = dict(total_input_advertisements=len(raw), total_unique_advertisements=len(classified),
                   duplicate_ids_removed=len(raw)-len(classified), analysis_advertisements=len(eligible),
                   excluded_date_rows=len(classified)-len(eligible),
                   relevant=int(counts.drop([IRRELEVANT, UNCERTAIN]).sum()),
                   irrelevant=int(counts[IRRELEVANT]), uncertain=int(counts[UNCERTAIN]),
                   count_per_job_category={k:int(v) for k,v in counts.items()},
                   missing_data_statistics={col:int(raw[col].map(normalize_text).eq('').sum()) for col in raw},
                   date_status_counts={k:int(v) for k,v in classified.date_status.value_counts().items()},
                   source=str(source.resolve()), source_sha256=digest,
                   start=config['start'], end_inclusive=config['end'], rule_version=RULE_VERSION,
                   sample_seed=seed, sample_requested=sample_size, sample_actual=min(sample_size,len(classified)),
                   accuracy=None, validation_status='Awaiting human review', market_coverage_complete=False,
                   coverage_note='Keyword-retrieved candidates only. Counts are provisional classifications, not complete Swedish job-market demand. Category totals exclude invalid/out-of-window dates; all rows remain in output.')
    output = Path(output) if output else ROOT / 'data' / 'processed' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    classified.to_csv(output / 'jobs_classified.csv', index=False, encoding='utf-8-sig')
    review_sample(classified, sample_size, seed).to_csv(output / 'manual_review.csv', index=False, encoding='utf-8-sig')
    (output / 'classification_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, 'Source changed during pipeline'
    return output, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sample-size', type=int, default=150)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    output, summary = run_pipeline(args.collection, args.output, args.sample_size, args.seed)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f'Output: {output.resolve()}')
