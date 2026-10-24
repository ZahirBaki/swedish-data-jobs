"""Reproducible offline Step 3B outputs, comparison, and separate review samples."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.classify_jobs_v2 import CATEGORIES, RELEVANCES, RULE_VERSION, classify_frame

ROOT = Path(__file__).resolve().parents[1]
LEGACY_FIELDS = ['primary_job_category', 'classification_method', 'classification_evidence', 'review_status']
STRUCTURED_FIELDS = ['evidence', 'candidate_categories', 'review_reasons']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def latest_step3(root=ROOT / 'data' / 'processed'):
    # v2 writes step3b_summary.json, deliberately not classification_summary.json.
    candidates = [p.parent for p in Path(root).glob('*/classification_summary.json')
                  if (p.parent / 'jobs_classified.csv').is_file()]
    if not candidates:
        raise FileNotFoundError('No original Step 3 classified output found')
    return max(candidates, key=lambda p:p.name)


def original_relevance(category):
    return {'Irrelevant':'irrelevant', 'Uncertain / Needs Review':'uncertain'}.get(category, 'relevant')


def make_samples(frame, representative_size=200, targeted_size=150, seed=42):
    if representative_size < 1 or targeted_size < 1:
        raise ValueError('Review sample sizes must be positive')
    random = frame.sample(n=min(representative_size, len(frame)), random_state=seed).copy()
    random['sampling_design'] = 'simple_random_without_replacement'
    random['inclusion_probability'] = len(random) / len(frame) if len(frame) else 0
    # Targeted cases are disjoint from the representative sample; no metrics pooling.
    remaining = frame.drop(index=random.index)
    pool = remaining.loc[remaining.relevance.eq('uncertain') |
                         remaining.review_reasons.map(lambda rs:any('conflict' in r for r in rs))].copy()
    pool['target_reason'] = pool.review_reasons.map(
        lambda rs: next((r for r in rs if 'conflict' in r), 'uncertain_insufficient_evidence'))
    strata = [group for _,group in pool.sample(frac=1, random_state=seed+1).groupby('target_reason', sort=True)]
    selected, depth, target = [], 0, min(targeted_size, len(pool))
    while len(selected) < target:
        for group in strata:
            if depth < len(group) and len(selected) < target:
                selected.append(group.index[depth])
        depth += 1
    targeted = pool.loc[selected].copy()
    targeted['sampling_design'] = 'targeted_balanced_by_review_reason_excluding_representative_sample'
    targeted['inclusion_probability'] = None
    for sample, sample_seed in [(random, seed), (targeted, seed+1)]:
        sample['sampling_seed'] = sample_seed
        sample['sample_order'] = range(1, len(sample)+1)
        for col in ['human_relevance', 'human_primary_category', 'human_verified', 'reviewer', 'review_notes']:
            sample[col] = ''
    return random, targeted


def serialize_csv(frame, path):
    serial = frame.copy(deep=True)
    for col in STRUCTURED_FIELDS:
        if col in serial:
            serial[col] = serial[col].map(lambda value:json.dumps(value, ensure_ascii=False))
    serial.to_csv(path, index=False, encoding='utf-8-sig')


def evaluate_verified_review(review):
    """Only explicitly verified human labels can contribute to any accuracy."""
    required = {'human_verified', 'human_relevance', 'human_primary_category', 'reviewer',
                'relevance', 'primary_category', 'sampling_design'}
    if missing := required - set(review):
        raise ValueError(f'Missing review columns: {sorted(missing)}')
    verified = review.human_verified.fillna('').str.strip().str.casefold()
    if set(verified) - {'', 'true', 'false'}:
        raise ValueError('human_verified must be blank, true, or false')
    rows = review.loc[verified.eq('true')].copy().fillna('')
    if rows.empty:
        return {'verified_rows':0, 'validation_status':'No verified human labels; accuracy unavailable'}
    for col in ['human_relevance', 'human_primary_category', 'reviewer']:
        rows[col] = rows[col].str.strip()
    if not set(rows.human_relevance) <= set(RELEVANCES):
        raise ValueError('Verified rows require a defined human_relevance')
    if not set(rows.human_primary_category) <= set(CATEGORIES) | {''}:
        raise ValueError('Unknown human_primary_category')
    if rows.reviewer.eq('').any():
        raise ValueError('Verified rows require a human reviewer identifier')
    if (rows.human_relevance.ne('relevant') & rows.human_primary_category.ne('')).any():
        raise ValueError('Human category must be empty for irrelevant/uncertain rows')
    if rows.sampling_design.nunique() != 1:
        raise ValueError('Do not pool representative and targeted samples')
    confusion = pd.crosstab(rows.human_relevance, rows.relevance).reindex(
        index=RELEVANCES, columns=RELEVANCES, fill_value=0)
    return dict(verified_rows=len(rows), sample_rows=len(review),
                relevance_sample_accuracy=float(rows.human_relevance.eq(rows.relevance).mean()),
                category_and_relevance_sample_accuracy=float((rows.human_relevance.eq(rows.relevance) &
                    rows.human_primary_category.eq(rows.primary_category)).mean()),
                relevance_confusion_matrix=confusion.to_dict(orient='index'),
                scope='Verified rows of this sample only; partial review and targeted selection limit generalization',
                sampling_design=rows.sampling_design.iloc[0])


def run(processed=None, output=None, seed=42, representative_size=200, targeted_size=150):
    if representative_size < 1 or targeted_size < 1:
        raise ValueError('Review sample sizes must be positive')
    processed = Path(processed) if processed else latest_step3()
    source = processed / 'jobs_classified.csv'
    # Protect original Step 3 and investigation artifacts, collector and old code/notebooks.
    protected = [p for p in (ROOT / 'data' / 'processed').rglob('*') if p.is_file()]
    protected += [p for p in (ROOT / 'data' / 'collections').glob('*/jobs.csv')]
    protected += [ROOT / 'src' / name for name in ['collect_jobs.py', 'classify_jobs.py', 'clean_jobs.py', 'investigate_uncertain.py']]
    protected += list((ROOT / 'notebooks').glob('*.ipynb'))
    protected += [source]
    before_hashes = {str(p.resolve()):digest(p) for p in protected}
    baseline = pd.read_csv(source, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    required = set(LEGACY_FIELDS) | {'job_title', 'description_text', 'advertisement_id'}
    if missing := required - set(baseline):
        raise ValueError(f'Missing Step 3 fields: {sorted(missing)}')
    if baseline.empty or baseline.advertisement_id.eq('').any() or baseline.advertisement_id.duplicated().any():
        raise ValueError('Step 3 input must have unique nonempty advertisement IDs')
    old_categories = set(baseline.primary_job_category)
    if old_categories - (set(CATEGORIES) | {'Irrelevant', 'Uncertain / Needs Review'}):
        raise ValueError('Unrecognized original Step 3 categories')
    classified = classify_frame(baseline)
    classified['step3_relevance'] = baseline.primary_job_category.map(original_relevance)
    # Classify every row, and retain Step 3 date eligibility for comparable totals.
    eligible = (baseline.in_analysis_window.str.casefold().eq('true')
                if 'in_analysis_window' in baseline else pd.Series(True,index=baseline.index))
    analysis = classified.loc[eligible]
    old_rel = analysis.step3_relevance.value_counts().reindex(RELEVANCES, fill_value=0)
    new_rel = analysis.relevance.value_counts().reindex(RELEVANCES, fill_value=0)
    relevance_comparison = pd.DataFrame({'relevance':RELEVANCES, 'before':old_rel.values, 'after':new_rel.values})
    relevance_comparison['change'] = relevance_comparison.after - relevance_comparison.before
    category_labels = CATEGORIES + ['(null)']
    old_primary = baseline.loc[eligible, 'primary_job_category'].where(
        baseline.loc[eligible, 'primary_job_category'].isin(CATEGORIES), '(null)')
    old_counts = old_primary.value_counts().reindex(category_labels, fill_value=0)
    new_counts = analysis.primary_category.fillna('(null)').value_counts().reindex(category_labels, fill_value=0)
    category_comparison = pd.DataFrame({'primary_category':category_labels, 'before':old_counts.values, 'after':new_counts.values})
    category_comparison['change'] = category_comparison.after - category_comparison.before
    transitions = pd.crosstab(analysis.step3_relevance, analysis.relevance).reindex(
        index=RELEVANCES, columns=RELEVANCES, fill_value=0)
    random, targeted = make_samples(classified, representative_size, targeted_size, seed)
    output = Path(output) if output else ROOT / 'data' / 'processed' / (
        'step3b-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    serialize_csv(classified, output / 'jobs_classified_v2.csv')
    # JSONL provides explicit nulls and booleans without CSV's empty-cell conventions.
    with (output / 'classification_comparison.jsonl').open('w',encoding='utf-8') as file:
        for row in classified.itertuples():
            file.write(json.dumps(dict(advertisement_id=row.advertisement_id,
                before=dict(primary_job_category=row.step3_primary_job_category,
                            relevance=row.step3_relevance, method=row.step3_classification_method,
                            evidence=row.step3_classification_evidence, review_status=row.step3_review_status),
                after=dict(relevance=row.relevance, primary_category=None if pd.isna(row.primary_category) else row.primary_category,
                           classification_method=row.classification_method, evidence=row.evidence,
                           review_required=row.review_required, review_reasons=row.review_reasons,
                           candidate_categories=row.candidate_categories)), ensure_ascii=False, allow_nan=False)+'\n')
    serialize_csv(random, output / 'manual_review_representative.csv')
    serialize_csv(targeted, output / 'manual_review_targeted.csv')
    relevance_comparison.to_csv(output / 'relevance_comparison.csv', index=False)
    category_comparison.to_csv(output / 'category_comparison.csv', index=False)
    transitions.to_csv(output / 'relevance_transitions.csv')
    summary = dict(rule_version=RULE_VERSION, source=str(source.resolve()), source_sha256=digest(source),
                   implementation_sha256={name:digest(ROOT / 'src' / name)
                                          for name in ['classify_jobs_v2.py', 'run_step3b.py']},
                   total_input_advertisements=len(baseline), total_output_advertisements=len(classified),
                   analysis_advertisements=len(analysis), date_excluded_from_counts=int((~eligible).sum()),
                   relevance_before={k:int(v) for k,v in old_rel.items()},
                   relevance_after={k:int(v) for k,v in new_rel.items()},
                   category_comparison=category_comparison.to_dict('records'),
                   relevance_transitions=transitions.to_dict(orient='index'),
                   relevant_without_primary_category=int((analysis.relevance.eq('relevant') & analysis.primary_category.isna()).sum()),
                   review_required=int(classified.review_required.sum()),
                   method_counts={k:int(v) for k,v in classified.classification_method.value_counts().items()},
                   representative_sample_size=len(random), representative_seed=seed,
                   targeted_sample_size=len(targeted), targeted_seed=seed+1,
                   representative_sampling='Uniform random without replacement from all source rows',
                   targeted_sampling='Balanced across uncertain/conflicting review reasons, excluding representative rows',
                   samples_disjoint=True, manual_labels_applied=0,
                   validation_status='No verified human labels; no accuracy calculated',
                   market_coverage_complete=False,
                   coverage_note='Provisional classifications of keyword-retrieved candidates, not complete Swedish job-market demand.',
                   preserved_input_sha256=before_hashes)
    for name in ['classify_jobs_v2.py', 'run_step3b.py']:
        (output / name).write_bytes((ROOT / 'src' / name).read_bytes())
    (output / 'step3b_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def table(df):
        return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join(
            '| '+' | '.join(str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
    report = f'''# Step 3B classification comparison

Source: `{source}`. Rule version: `{RULE_VERSION}`. {len(classified):,} advertisements classified; {len(analysis):,} date-eligible rows contribute to comparisons.

## Relevance before and after

{table(relevance_comparison)}

## Primary categories before and after

{table(category_comparison)}

Old Irrelevant and Uncertain / Needs Review labels are mapped to separate relevance values, with a null primary category. The nine substantive original categories remain available, plus five new ones. Original labels, methods, evidence and review status are preserved in `step3_` columns and `classification_comparison.jsonl`. CSV null categories are empty cells; JSONL uses literal null and boolean values.

{summary['relevant_without_primary_category']} relevant advertisements have no primary category because multiple category rules conflict. Relevance and category are separate decisions. {summary['review_required']} rows require review (including uncertain, conflicting, broad aliases, missing data or competence-only evidence). A false review_required flag means no rule-level review trigger, not human verification.

## Review and limitations

`manual_review_representative.csv` contains {len(random)} uniformly sampled advertisements (seed {seed}); `manual_review_targeted.csv` contains {len(targeted)} separate uncertain/conflicting cases (seed {seed+1}). The sets are disjoint. Both preserve full source descriptions and predictions. Fill human_relevance, human_primary_category (empty if irrelevant/uncertain), reviewer and review_notes. Set human_verified=true only after a human has read and checked the label. Relevant cases may retain an empty category if the role cannot be resolved. Do not pool targeted labels with the representative sample to claim population accuracy. Partial human review can bias evaluation.

Run `python -m src.run_step3b --evaluate-review PATH_TO_REVIEW_CSV` to evaluate explicitly verified labels. With none present it reports that accuracy is unavailable. No human labels were supplied or applied in this run; no accuracy was calculated.

Title rules match explicit roles, English/Swedish variants, plural forms and abbreviations; punctuation and seniority surrounding a role do not prevent matching. Ambiguous abbreviations such as DE, DA and DS alone do not establish a role. Description rules require a specific action/object pair in a candidate-duty or required-experience context. Optional, company-background, collaboration, recruitment-tool and negated contexts are rejected. Explicit non-Data occupations or specific non-Data duties may establish irrelevance; a no-match never does. Nested ETL/pipeline and BI/analysis patterns have documented specificity priorities; other category conflicts abstain. No paid services or extraction of skill/education/language fields are involved.

This is transparent heuristic classification, not robust natural-language understanding. Sentence/section parsing can miss unstructured, mixed-language or oddly formatted responsibilities. Titles and required competencies can overstate actual duties. Broad Swedish dataingenjör/datavetare and generic Data Administrator/Specialist aliases require review. Academic AI/ML posts and explicit Data/AI team/programme management are included, but generic academic, GIS, security and hybrid engineering titles require duties evidence. Thesis/student titles alone are not sufficient. Research scope and overlapping responsibilities need human validation. Counts do not establish complete market demand or accuracy, and fewer uncertain rows are not proof of improvement.
'''
    (output / 'comparison_report.md').write_text(report,encoding='utf-8')
    for path, previous in before_hashes.items():
        if digest(path) != previous:
            raise AssertionError(f'Protected source changed: {path}')
    saved = pd.read_csv(output / 'jobs_classified_v2.csv', dtype=str, keep_default_na=False)
    for column in baseline:
        target = 'step3_'+column if column in LEGACY_FIELDS else column
        pd.testing.assert_series_equal(saved[target],baseline[column],check_names=False)
    assert set(random.advertisement_id).isdisjoint(targeted.advertisement_id)
    assert int(new_rel.sum()) == len(analysis) == int(new_counts.sum())
    assert len(classified) == len(baseline)
    assert classified.loc[classified.relevance.ne('relevant'), 'primary_category'].isna().all()
    (output / 'verification.json').write_text(json.dumps(dict(input_rows=len(baseline), output_rows=len(saved),
        source_columns_preserved=True, protected_files_unchanged=len(before_hashes),
        review_samples_disjoint=True, summary_counts_reconciled=True), indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k != 'preserved_input_sha256'},ensure_ascii=True,indent=2))
    print(f'Output: {output.resolve()}')
    return output, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--representative-size',type=int,default=200)
    parser.add_argument('--targeted-size',type=int,default=150)
    parser.add_argument('--evaluate-review',type=Path)
    args=parser.parse_args()
    if args.evaluate_review:
        review=pd.read_csv(args.evaluate_review,dtype=str,keep_default_na=False)
        print(json.dumps(evaluate_verified_review(review),ensure_ascii=True,indent=2))
    else:
        run(args.processed,args.output,args.seed,args.representative_size,args.targeted_size)
