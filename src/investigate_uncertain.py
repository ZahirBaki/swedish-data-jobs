"""Read-only diagnostics of saved uncertain predictions, never a relabeling step.

Run: python -m src.investigate_uncertain --seed 42
Uses pandas and the standard library only. Screening buckets are unvalidated
title evidence for investigation, not new primary categories or market counts.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'src').is_dir() and (p / 'data').is_dir())
UNCERTAIN = 'Uncertain / Needs Review'
SENIORITY = ('senior', 'seniora', 'junior', 'juniora', 'sr', 'jr',
             'erfaren', 'erfarna', 'experienced', 'principal', 'staff',
             'nyexaminerad', 'nyutexaminerad', 'midlevel')


def punctuation_words(title):
    text = unicodedata.normalize('NFKC', title).casefold()
    text = ''.join(c if c.isalnum() else ' ' for c in text)
    return re.sub(r'\s+', ' ', text).strip()


def title_group(title):
    words = re.sub(r'\bmid level\b', ' ', punctuation_words(title))
    words = re.sub(r'\b(?:' + '|'.join(SENIORITY) + r')\b', ' ', words)
    # Compact key makes e.g. GIS-ingenjör, GIS ingenjör, and GISingenjör agree.
    return ''.join(words.split()) or '(empty after normalization)'


# These are topic-screening probes for this investigation, NOT classifier rules.
# Positive evidence requires a recognizable domain/role combination, not bare AI.
POTENTIAL_PROBES = [
    ('core_role_spelling_or_plural', r'\b(?:data (?:engineers?|analysts?|scientists?)|ai (?:engineers?|developers?|utvecklare|ingenjörer?)|ml (?:engineers?|developers?|utvecklare)|machine learning (?:engineers?|developers?)|analytics engineers?|bi (?:developers?|analysts?|utvecklare|analytiker))\b'),
    ('data_governance_management_architecture', r'\b(?:data (?:governance|stewards?|owners?|administrat\w*|management|quality|modellers?|modelers?|architect\w*|arkitekt\w*|lead|process)|masterdata\w*|datakvalitet\w*|datamodell\w*|datasamordnare|datamigrering|chief data)\b'),
    ('analytics_bi_warehouse_role', r'\b(?:analytics (?:consultant\w*|specialist\w*|lead|manager\w*)|data (?:och|and) analytics|business intelligence|bi (?:specialist\w*|konsult\w*|architect\w*|arkitekt\w*|ansvarig)|datavaruhus\w*|data warehouse|snowflake developer\w*|idq utvecklare|informatica utvecklare)\b'),
    ('ai_ml_architecture_operations', r'\b(?:ai (?:ops|operations|architect\w*|arkitekt\w*|lead|specialist\w*|konsult\w*|och data|and data)|mlops|aiops)\b'),
    ('ai_ml_research_title', r'\b(?:forsk\w*|doktorand\w*|postdoktor\w*|research\w*|phd)\b.{0,140}\b(?:ai|ml|maskininlärning\w*|artificiell intelligens|machine learning|deep learning|computer vision|datoriserad bildbehandling)\b'),
    ('ai_development_or_student_role', r'\b(?:utvecklar\w*|studentarbetare|engineer\w*)\s+(?:inom|in|med fokus på)\s+ai\b'),
    ('statistics_or_domain_analytics', r'\b(?:statistiker|statistician|biostatistiker|biostatistician|marknadsanalytiker|marketing analyst|produktanalytiker|product analyst)\b'),
]
UNRELATED_PROBES = [
    ('physical_engineering', r'\b(?:(?:va|vvs|ils|lsa|rf|pcb|cad|kma|ot|scada|process|drift|bygg|maskin|el|mekanik|mekanisk|optronik)[ ]?ingenjör\w*|ingenjör licensiering|(?:mechanical|electrical|civil|hardware|rf) engineer\w*|nätplanerare|motorprovare|provcellsingenjör)\b'),
    ('commercial_legal_hr_administration', r'\b(?:rekryterare|recruiter|affärsjurist|jurist|lawyer|legal counsel|finance director|ekonomichef|redovisningsekonom|löneadministratör|arbetsmarknadskoordinator|logistikadministratör|marknads och försäljningskoordinator|brand manager|field sales representative|säljare|butikssäljare|kock|specialpedagog|sjuksköterska|undersköterska)\b'),
]
DOMAIN_TITLE = r'\b(?:ai|ml|data|analytics|bi|machine learning|maskininlärning\w*|dataanalys\w*|artificiell intelligens)\b'
BODY_SUPPORT = r'\b(?:data pipelines?|datapipelines?|datavaruhus\w*|etl|machine learning|maskininlärning\w*|data governance|masterdata\w*|statistisk dataanalys)\b'
RECRUITMENT_AI = r'(?:ai.{0,100}(?:rekryteringsprocess|recruitment process)|(?:rekryteringsprocess|recruitment process).{0,100}ai)'


def screen_title(title):
    text = punctuation_words(title)
    positive = [name for name, pattern in POTENTIAL_PROBES if re.search(pattern, text)]
    unrelated = [name for name, pattern in UNRELATED_PROBES if re.search(pattern, text)]
    domain = bool(re.search(DOMAIN_TITLE, text))
    if positive and not unrelated:
        bucket = 'Potentially relevant title evidence'
    elif unrelated and not positive and not domain:
        bucket = 'Apparently unrelated title evidence'
    else:
        bucket = 'Unresolved from title screening'
    return bucket, '; '.join(positive), '; '.join(unrelated), domain


def reason(row):
    method, evidence = row.classification_method, row.classification_evidence
    if method == 'ambiguous_title':
        return ('Domain title lacks a specific role rule' if evidence.startswith('Data/AI domain')
                else 'Generic role title; no existing description role match')
    return {'insufficient_evidence': 'Unrecognized title; description missing or has a broad domain mention',
            'description_support': 'Description mentions named roles but does not establish advertised duties',
            'conflicting_title': 'Multiple title roles or conflicting occupation cues',
            'missing_title': 'Missing usable title'}.get(method, 'Other saved method: ' + method)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def markdown_table(frame):
    # Avoid pandas.to_markdown: tabulate is not a project dependency.
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    rows = ['| ' + ' | '.join(map(cell, frame.columns)) + ' |',
            '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    rows.extend('| ' + ' | '.join(map(cell, row)) + ' |' for row in frame.itertuples(index=False, name=None))
    return '\n'.join(rows)


def investigate(processed=None, output=None, seed=42):
    if processed is None:
        candidates = [p.parent for p in (ROOT / 'data' / 'processed').glob('*/classification_summary.json')
                      if (p.parent / 'jobs_classified.csv').is_file()]
        if not candidates:
            raise FileNotFoundError('No classified dataset with a summary found')
        processed = max(candidates, key=lambda p: p.name)
    processed = Path(processed)
    protected = list(processed.glob('*')) + [ROOT / 'src' / 'classify_jobs.py', ROOT / 'src' / 'clean_jobs.py']
    protected += list((ROOT / 'data' / 'collections').glob('*/jobs.csv'))
    hashes = {str(p.resolve()): sha256(p) for p in protected if p.is_file()}
    source = processed / 'jobs_classified.csv'
    frame = pd.read_csv(source, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    required = {'advertisement_id', 'job_title', 'description_text', 'primary_job_category',
                'classification_method', 'classification_evidence'}
    if missing := required - set(frame):
        raise ValueError(f'Missing required fields: {sorted(missing)}')
    uncertain = frame.loc[frame.primary_job_category.eq(UNCERTAIN)].copy()
    if len(uncertain) < 50:
        raise ValueError('Need at least 50 uncertain rows for this investigation')
    output = Path(output) if output else ROOT / 'data' / 'processed' / (
        'uncertain-investigation-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    titles = uncertain.groupby('job_title', dropna=False).size().reset_index(name='advertisement_count')
    titles = titles.sort_values(['advertisement_count', 'job_title'], ascending=[False, True]).reset_index(drop=True)
    titles.head(100).to_csv(output / 'top_100_uncertain_titles.csv', index=False, encoding='utf-8-sig')
    uncertain['investigation_title_group'] = uncertain.job_title.map(title_group)
    variants = uncertain.groupby(['investigation_title_group', 'job_title']).size().reset_index(name='advertisement_count')
    grouped = uncertain.groupby('investigation_title_group').agg(
        advertisement_count=('job_title', 'size'), distinct_original_titles=('job_title', 'nunique')).reset_index()
    examples = {}
    for key, rows in variants.groupby('investigation_title_group'):
        rows = rows.sort_values(['advertisement_count', 'job_title'], ascending=[False, True])
        examples[key] = json.dumps([dict(title=r.job_title, count=int(r.advertisement_count))
                                    for r in rows.itertuples()], ensure_ascii=False)
    grouped['original_title_variants_json'] = grouped.investigation_title_group.map(examples)
    grouped = grouped.sort_values(['advertisement_count', 'investigation_title_group'], ascending=[False, True])
    grouped.to_csv(output / 'grouped_uncertain_titles.csv', index=False, encoding='utf-8-sig')
    grouped.head(100).to_csv(output / 'top_100_grouped_titles.csv', index=False, encoding='utf-8-sig')
    variants.to_csv(output / 'title_group_membership.csv', index=False, encoding='utf-8-sig')
    uncertain['investigation_reason'] = uncertain.apply(reason, axis=1)
    screening = pd.DataFrame([screen_title(t) for t in uncertain.job_title], index=uncertain.index,
                             columns=['investigation_screen', 'positive_title_probes', 'unrelated_title_probes', 'domain_in_title'])
    uncertain = uncertain.join(screening)
    uncertain['description_support_probe'] = uncertain.description_text.map(
        lambda text: bool(re.search(BODY_SUPPORT, text.casefold())))
    uncertain['recruitment_ai_probe'] = uncertain.description_text.map(
        lambda text: bool(re.search(RECRUITMENT_AI, re.sub(r'\s+', ' ', text.casefold()))))
    # Preserve physical CSV row identity for reruns even if an ID is unexpectedly repeated.
    uncertain['source_csv_data_row'] = uncertain.index + 1
    sample = uncertain.sample(n=50, random_state=seed).copy()
    sample['sample_order'] = range(1, 51)
    sample['sampling_seed'] = seed
    sample.to_csv(output / 'uncertain_sample_50.csv', index=False, encoding='utf-8-sig')
    audit_columns = ['source_csv_data_row', 'advertisement_id', 'job_title', 'classification_method',
                     'classification_evidence', 'investigation_reason', 'investigation_title_group',
                     'investigation_screen', 'positive_title_probes', 'unrelated_title_probes',
                     'domain_in_title', 'description_support_probe', 'recruitment_ai_probe']
    uncertain[audit_columns].to_csv(output / 'uncertain_diagnostics.csv', index=False, encoding='utf-8-sig')
    counts = uncertain.investigation_screen.value_counts()
    reasons = uncertain.investigation_reason.value_counts()
    methods = uncertain.classification_method.value_counts()
    summary = dict(source=str(source.resolve()), source_sha256=hashes[str(source.resolve())],
                   input_rows=len(frame), uncertain_rows=len(uncertain),
                   distinct_original_titles=len(titles), normalized_title_groups=len(grouped),
                   multi_variant_groups=int(grouped.distinct_original_titles.gt(1).sum()),
                   advertisements_in_multi_variant_groups=int(grouped.loc[grouped.distinct_original_titles.gt(1), 'advertisement_count'].sum()),
                   title_screening_counts={k:int(v) for k,v in counts.items()},
                   saved_classification_method_counts={k:int(v) for k,v in methods.items()},
                   investigation_reason_counts={k:int(v) for k,v in reasons.items()},
                   body_support_probe_count=int(uncertain.description_support_probe.sum()),
                   recruitment_ai_probe_count=int(uncertain.recruitment_ai_probe.sum()),
                   sample_size=50, sampling_seed=seed, sampling_design='Simple random sample without replacement from source row order; pandas.sample',
                   sample_ids_in_order=sample.advertisement_id.tolist(), pandas_version=pd.__version__,
                   source_files_preserved_sha256=hashes,
                   validation_status='Unvalidated investigation; no human labels or accuracy',
                   caveat='Screening counts describe title evidence, not final job categories. Unresolved rows must not be counted as unrelated. No cohort-wide desk review or market-demand inference.')
    (output / 'investigation_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output / 'investigation_probes.json').write_text(json.dumps(
        dict(potential_title_probes=POTENTIAL_PROBES, unrelated_title_probes=UNRELATED_PROBES,
             domain_title_probe=DOMAIN_TITLE, body_support_probe=BODY_SUPPORT,
             recruitment_ai_probe=RECRUITMENT_AI, seniority_terms=SENIORITY), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output / 'investigate_uncertain.py').write_bytes(Path(__file__).read_bytes())
    report = f'''# Investigation of uncertain advertisements

Source: `{source.relative_to(ROOT) if source.is_relative_to(ROOT) else source}`.
SHA-256: `{summary['source_sha256']}`. The latest processed run with a classification summary was selected; investigation folders do not qualify as classified inputs.

There are **{len(uncertain):,} uncertain advertisements out of {len(frame):,} candidates** ({len(uncertain)/len(frame):.1%}). This is an investigation of saved predictions, not a change to classification. Original titles, descriptions, predictions and files remain unchanged. Counts do not establish complete Swedish job-market demand or classifier accuracy.

## Why the current classifier abstained

These counts describe the saved classifier execution paths and are exact, not inferred from the sample.

{markdown_table(reasons.rename_axis('Reason').reset_index(name='Advertisements'))}

`ambiguous_title` combines two very different paths: domain-specific titles unsupported by the narrow role patterns, and generic occupation words. The latter path returns uncertain even if there is no Data/AI evidence in either the title or description. `insufficient_evidence` means no title rule and either a missing description or a broad domain mention; it does not mean relevance was demonstrated. A coworker role, employer service list or recruitment-process AI disclaimer can produce domain evidence unrelated to advertised duties. Description role matches deliberately do not establish duties. Multiple relevant title matches can also cause abstention.

## Cohort screening: apparent unrelated versus potential relevance

{markdown_table(counts.rename_axis('Investigation screen').reset_index(name='Advertisements'))}

These are **unvalidated title-only screening counts**, not final labels and not estimates from the 50-row sample. Every uncertain row is accounted for once in this table. Positive probes look for named Data/AI, analytics, statistics, governance, architecture or research roles. Negative probes look for explicit physical engineering, commercial, legal, HR or administrative occupations. A negative probe counts as apparently unrelated only without a positive probe or broad Data/AI token in the title. Conflicting, generic software/IT, unrecognized and interdisciplinary titles remain unresolved. This conservative screen misses some relevant roles whose duties appear only in the body and may misread domain terminology. Neither apparently unrelated nor potential means verified. No row is assigned a new job category.

The exact probes are in `investigation_probes.json`; per-row evidence is in `uncertain_diagnostics.csv`. The existing classifier is not imported or rerun. Body probes are separate nonexclusive diagnostics and do not determine the title-screen buckets. A body support phrase appears in {summary['body_support_probe_count']:,} uncertain descriptions; the bounded AI/recruitment-process probe appears in {summary['recruitment_ai_probe_count']:,}. These probes count mentions, not causation, job duties or validated relevance; they may miss variants or match incidental text.

## Grouping similar titles

The {len(titles):,} distinct original titles become {len(grouped):,} comparison groups. {summary['multi_variant_groups']:,} groups contain multiple distinct original titles, covering {summary['advertisements_in_multi_variant_groups']:,} advertisements. Grouping uses Unicode NFKC, case folding, punctuation/symbol separation, an explicit seniority-term list and whitespace-insensitive keys. No employer, location, technology or other substantive words are removed. Lead, manager, trainee, researcher and project titles retain those words because they can change role scope. Inflections and translations are not silently merged. Grouping is for frequency inspection only: punctuation can be meaningful (e.g. C# versus C++), so inspect original variants before treating a group as one occupation. No advertisements are deduplicated by title.

`grouped_uncertain_titles.csv` contains every group and original variants with counts. `title_group_membership.csv` is the long-form membership audit. The 100 most common original and grouped titles are also separate CSV files; ties sort alphabetically for reproducibility.

## Seeded inspection sample

`uncertain_sample_50.csv` contains **50 distinct source rows**, sampled without replacement with seed **{seed}**, from the full uncertain subset in saved CSV row order. All original columns, full descriptions and titles are preserved, alongside diagnostic fields, row positions and sample order. No excerpt replaces the source description. This is a separate simple random sample, not the earlier balanced classification review sample. Seed, IDs, pandas version and hashes are recorded in `investigation_summary.json`. Any desk-inspection observations are assistant assessments, not human validation or a measured accuracy.

## Recommendations before changing classification

1. Normalize Unicode dashes and title punctuation for matching and add bounded singular/plural variants. Inspect `Senior AI Engineers` and compound Swedish role forms. Keep original titles and evidence.
2. Review named gaps such as Data Governance Lead, Data Administrator, Data Stewards, BI Specialist, Analytics Consultant, Snowflake Developer and IDQ-utvecklare. Require duties-based corroboration for vendor titles and define the boundary of Other relevant roles.
3. Add contextual research handling: explicitly named ML/AI research posts are plausible Other relevant roles; generic researchers and wet-lab jobs need actual computational duties. Decide how PhD, postdoc, thesis and student work should be represented before counting them alongside regular jobs.
4. Separate generic software/IT and physical engineering from data-specific duties. Expand explicit unrelated-occupation handling only after checking descriptions and counterexamples; do not blanket-exclude engineer/developer/researcher titles.
5. Recognize whether a description states the advertised responsibilities, names another team member, lists company offerings, or describes recruitment-tool use. A bare AI mention or generic data analysis is not AI Engineer evidence.
6. For conflicting relevant title matches, retain all candidate categories and review a documented primary-role tie-break policy. Do not discard the advertisement's apparent relevance merely because its primary category is ambiguous.
7. Human-review this 50-row sample and the existing stratified sample, retaining an untouched evaluation subset. Evaluate proposed changes on reviewed labels and report category errors as well as sample accuracy. No rule changes are made by this investigation.

## Top 100 original uncertain titles

{markdown_table(titles.head(100))}

## Top 100 grouped titles

{markdown_table(grouped.head(100))}
'''
    (output / 'report.md').write_text(report, encoding='utf-8')
    # Check saved artifacts against the input, including full description preservation.
    saved = pd.read_csv(output / 'uncertain_sample_50.csv', dtype=str, keep_default_na=False, encoding='utf-8-sig')
    pd.testing.assert_frame_equal(saved[frame.columns].reset_index(drop=True), sample[frame.columns].reset_index(drop=True))
    assert saved.source_csv_data_row.nunique() == 50
    assert int(counts.sum()) == len(uncertain) == int(grouped.advertisement_count.sum())
    assert int(variants.advertisement_count.sum()) == len(uncertain)
    assert sample.index.tolist() == uncertain.sample(n=50, random_state=seed).index.tolist()
    for path, before in hashes.items():
        assert sha256(path) == before, f'Protected input changed: {path}'
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f'Output: {output.resolve()}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    investigate(args.processed, args.output, args.seed)
