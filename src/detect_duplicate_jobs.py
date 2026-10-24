"""Step 3C: flag exact/near duplicate candidates without deleting any records.

Run: python -m src.detect_duplicate_jobs. Only pandas and the standard library.
"""
import argparse
import hashlib
import itertools
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict,dataclass
from datetime import datetime,timezone
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from src.clean_jobs import publication_day
from src.classification_policy import add_policy_flags,CORE_CATEGORIES,EXTENDED_SCOPES,POSITION_RULES
from src.validate_classification import ROOT,latest_step3b,prepare_review_files,read_csv,sha256

VERSION = 'step3c-1.0'
PAIR_COLUMNS = ['left_id','right_id','match_type','description_similarity','title_similarity',
                'publication_day_gap','same_known_location','left_vacancies','right_vacancies',
                'separate_vacancy_signals','metadata_missing','limited_description_context',
                'duplicate_review_status','duplicate_group_id']


@dataclass(frozen=True)
class Settings:
    near_threshold: float = .85
    title_threshold: float = .65
    shingle_size: int = 5
    min_description_tokens: int = 40

    def __post_init__(self):
        if not 0 < self.near_threshold <= 1 or not 0 <= self.title_threshold <= 1:
            raise ValueError('Similarity thresholds must be within (0,1] and [0,1]')
        if self.shingle_size < 1 or self.min_description_tokens < self.shingle_size:
            raise ValueError('Invalid shingle/token settings')


def normalize(value):
    """Conservative exact-text normalization; preserve punctuation and numbers.

    This avoids treating C# and C++ or distinct employers as exact aliases.
    NFKC, casefold, Unicode separator/format whitespace and dash variants only.
    """
    if value is None or pd.isna(value):
        return ''
    value = unicodedata.normalize('NFKC',str(value)).casefold()
    value = ''.join(' ' if unicodedata.category(c).startswith('Z') or unicodedata.category(c)=='Cf'
                    else '-' if unicodedata.category(c)=='Pd' else c for c in value)
    return re.sub(r'\s+',' ',value).strip()


def tokens(value):
    return re.findall(r'\w+',value,flags=re.UNICODE)


def shingles(words,size):
    return frozenset(tuple(words[i:i+size]) for i in range(len(words)-size+1))


def vacancies(value):
    normalized = str(value).strip()
    return int(normalized) if normalized.isdigit() and int(normalized)>0 else None


def pair_metadata(left,right,match_type,token_counts,settings):
    left_day,right_day = publication_day(left.get('publication_date')),publication_day(right.get('publication_date'))
    gap = abs((left_day-right_day).days) if left_day and right_day else None
    left_loc = tuple(normalize(left.get(c,'')) for c in ['municipality','region'])
    right_loc = tuple(normalize(right.get(c,'')) for c in ['municipality','region'])
    known_location = all(left_loc) and all(right_loc)
    same_location = left_loc==right_loc if known_location else None
    lv,rv = vacancies(left.get('number_of_vacancies')),vacancies(right.get('number_of_vacancies'))
    signals = []
    if gap is not None and gap>0:
        signals.append('different_publication_days_possible_repost_or_separate_requisition')
    if same_location is False:
        signals.append('different_locations_possible_local_vacancies_or_multilocation_advertising')
    if lv is not None and rv is not None and lv!=rv:
        signals.append('different_vacancy_counts')
    if (lv or 0)>1 or (rv or 0)>1:
        signals.append('multiple_vacancies_declared')
    missing = gap is None or not known_location or lv is None or rv is None
    placeholder = any(re.search(r'för att se annonsen i sin helhet|to see the advertisement|to see the full advertisement',
                                normalize(row['description_text'])) for row in [left,right])
    limited = min(token_counts)<settings.min_description_tokens or placeholder
    if limited:
        status = 'limited_context_needs_review'
    elif signals:
        status = 'possible_separate_vacancies_or_repost_needs_review'
    elif missing:
        status = 'metadata_incomplete_needs_review'
    elif match_type=='exact':
        status = 'exact_repeat_candidate_needs_review'
    else:
        status = 'near_duplicate_needs_review'
    return dict(publication_day_gap=gap,same_known_location=same_location,left_vacancies=lv,
                right_vacancies=rv,separate_vacancy_signals=json.dumps(signals),metadata_missing=missing,
                limited_description_context=limited,duplicate_review_status=status)


def detect_duplicates(frame,settings=None):
    settings = settings or Settings()
    required = {'advertisement_id','employer','job_title','description_text','publication_date',
                'municipality','region','number_of_vacancies'}
    if missing := required-set(frame):
        raise ValueError(f'Missing fields: {sorted(missing)}')
    if frame.advertisement_id.map(normalize).eq('').any() or frame.advertisement_id.duplicated().any():
        raise ValueError('Expected Step 3B unique nonempty IDs; do not silently discard records')
    if any(c in frame for c in ['duplicate_group_id','exact_duplicate_group_id','duplicate_review_status']):
        raise ValueError('Input is already annotated; select the original Step 3B dataset')
    rows = frame.fillna('').to_dict('records')
    normalized = [(normalize(r['employer']),normalize(r['job_title']),normalize(r['description_text'])) for r in rows]
    exact_blocks,employer_blocks = defaultdict(list),defaultdict(list)
    for i,key in enumerate(normalized):
        if all(key):
            # Full tuple keys avoid hash collisions in the exact-equality check.
            exact_blocks[key].append(i)
        if key[0] and key[1] and key[2]:
            employer_blocks[key[0]].append(i)
    exact_ids = ['']*len(rows)
    pairs = []
    def add_pair(i,j,kind,description_score,title_score):
        if rows[i]['advertisement_id'] > rows[j]['advertisement_id']:
            i,j = j,i
        pairs.append(dict(left_id=rows[i]['advertisement_id'],right_id=rows[j]['advertisement_id'],
            match_type=kind,description_similarity=description_score,title_similarity=title_score,
            **pair_metadata(rows[i],rows[j],kind,
                [len(tokens(normalized[i][2])),len(tokens(normalized[j][2]))],settings)))
    for key,indices in exact_blocks.items():
        if len(indices)<2:
            continue
        identifier = 'exact-'+hashlib.sha256(json.dumps(key,ensure_ascii=False).encode()).hexdigest()[:16]
        for i in indices:
            exact_ids[i] = identifier
        for i,j in itertools.combinations(indices,2):
            add_pair(i,j,'exact',1.0,1.0)
    comparison_count = 0
    # Near search is blocked by exact normalized employer identity. No agency/client
    # identity guesses, cross-employer entity resolution or global N^2 text scan.
    for indices in employer_blocks.values():
        if len(indices)<2:
            continue
        word_lists = {i:tokens(normalized[i][2]) for i in indices}
        sets = {i:shingles(words,settings.shingle_size) for i,words in word_lists.items()
                if len(words)>=settings.min_description_tokens}
        for i,j in itertools.combinations(sorted(sets),2):
            if exact_ids[i] and exact_ids[i]==exact_ids[j]:
                continue
            a,b = sets[i],sets[j]
            if min(len(a),len(b))/max(len(a),len(b)) < settings.near_threshold:
                continue
            title_score = SequenceMatcher(None,normalized[i][1],normalized[j][1],autojunk=False).ratio()
            if title_score < settings.title_threshold:
                continue
            comparison_count += 1
            intersection = len(a & b)
            score = intersection/(len(a)+len(b)-intersection)
            if score>=settings.near_threshold:
                add_pair(i,j,'near',score,title_score)
    # Connected components are review clusters, NOT a claim of transitive equality.
    parent = {r['advertisement_id']:r['advertisement_id'] for r in rows}
    def find(identifier):
        while parent[identifier]!=identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier
    for pair in pairs:
        left,right = find(pair['left_id']),find(pair['right_id'])
        parent[max(left,right)] = min(left,right)
    components = defaultdict(list)
    for identifier in parent:
        components[find(identifier)].append(identifier)
    group_by_id = {}
    for ids in components.values():
        if len(ids)>1:
            group_id = 'dup-'+hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest()[:16]
            group_by_id.update({identifier:group_id for identifier in ids})
    statuses = defaultdict(set)
    for pair in pairs:
        pair['duplicate_group_id'] = group_by_id[pair['left_id']]
        for identifier in [pair['left_id'],pair['right_id']]:
            statuses[identifier].add(pair['duplicate_review_status'])
    status_order = ['limited_context_needs_review','possible_separate_vacancies_or_repost_needs_review',
                    'metadata_incomplete_needs_review','near_duplicate_needs_review','exact_repeat_candidate_needs_review']
    out = frame.copy(deep=True)
    out['exact_duplicate_group_id'] = exact_ids
    out['duplicate_group_id'] = out.advertisement_id.map(group_by_id).fillna('')
    out['duplicate_review_status'] = out.advertisement_id.map(
        lambda identifier:next((s for s in status_order if s in statuses[identifier]),'not_flagged_by_current_rules'))
    out['duplicate_review_required'] = out.duplicate_group_id.ne('')
    out['duplicate_status_evidence'] = out.advertisement_id.map(lambda identifier:json.dumps(sorted(statuses[identifier])))
    pair_frame = pd.DataFrame(pairs,columns=PAIR_COLUMNS).sort_values(['left_id','right_id']).reset_index(drop=True)
    group_rows = []
    for group_id,group in out.loc[out.duplicate_review_required].groupby('duplicate_group_id',sort=True):
        links = pair_frame.loc[pair_frame.duplicate_group_id.eq(group_id)]
        days = group.publication_date.map(publication_day).dropna()
        group_rows.append(dict(duplicate_group_id=group_id,advertisement_count=len(group),
            advertisement_ids=json.dumps(sorted(group.advertisement_id)),employers=json.dumps(sorted(set(group.employer)),ensure_ascii=False),
            original_titles=json.dumps(sorted(set(group.job_title)),ensure_ascii=False),
            exact_text_subgroups=group.exact_duplicate_group_id.replace('',pd.NA).nunique(),
            exact_pair_count=int(links.match_type.eq('exact').sum()),near_pair_count=int(links.match_type.eq('near').sum()),
            publication_min=str(min(days)) if len(days) else '',publication_max=str(max(days)) if len(days) else '',
            municipalities=json.dumps(sorted(set(group.municipality.fillna('').astype(str))),ensure_ascii=False),
            vacancy_counts=json.dumps(sorted(set(group.number_of_vacancies.fillna('').astype(str)))),
            review_statuses=json.dumps(sorted(set(links.duplicate_review_status))),
            separate_vacancy_signal_pairs=int(links.separate_vacancy_signals.ne('[]').sum())))
    groups = pd.DataFrame(group_rows,columns=['duplicate_group_id','advertisement_count','advertisement_ids',
        'employers','original_titles','exact_text_subgroups','exact_pair_count','near_pair_count',
        'publication_min','publication_max','municipalities','vacancy_counts','review_statuses','separate_vacancy_signal_pairs'])
    if len(groups):
        groups = groups.sort_values(['advertisement_count','duplicate_group_id'],ascending=[False,True])
    summary = dict(input_rows=len(frame),output_rows=len(out),exact_text_groups=len(set(exact_ids)-{''}),
        advertisements_in_exact_text_groups=int(out.exact_duplicate_group_id.ne('').sum()),
        exact_pairs=int(pair_frame.match_type.eq('exact').sum()),near_pairs=int(pair_frame.match_type.eq('near').sum()),
        duplicate_review_groups=len(groups),flagged_advertisements=int(out.duplicate_review_required.sum()),
        pair_status_counts={k:int(v) for k,v in pair_frame.duplicate_review_status.value_counts().items()},
        row_status_counts={k:int(v) for k,v in out.duplicate_review_status.value_counts().items()},
        near_text_comparisons=comparison_count,rows_missing_exact_identity_fields=sum(not all(key) for key in normalized),
        settings=asdict(settings),records_deleted=0,records_merged=0)
    return out,pair_frame,groups,summary


def run(processed=None,output=None,settings=None):
    processed = Path(processed) if processed else latest_step3b()
    source = processed/'jobs_classified_v2.csv'
    # Existing data/code/notebooks are read-only inputs; new artifacts go elsewhere.
    protected = [p for p in (ROOT/'data'/'processed').rglob('*') if p.is_file()]
    protected += list((ROOT/'data'/'collections').glob('*/jobs.csv'))
    protected += list((ROOT/'notebooks').glob('*.ipynb'))
    protected += [ROOT/'src'/name for name in ['collect_jobs.py','classify_jobs.py','classify_jobs_v2.py','run_step3b.py','clean_jobs.py']]
    protected += [p for p in processed.iterdir() if p.is_file()]
    hashes = {str(p.resolve()):sha256(p) for p in protected}
    original = read_csv(source)
    annotated,pairs,groups,summary = detect_duplicates(original,settings)
    annotated = add_policy_flags(annotated)
    output = Path(output) if output else ROOT/'data'/'processed'/('step3c-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    output.mkdir(parents=True,exist_ok=False)
    annotated.to_csv(output/'jobs_duplicate_review.csv',index=False,encoding='utf-8-sig')
    pairs.to_csv(output/'duplicate_pairs.csv',index=False,encoding='utf-8-sig')
    groups.to_csv(output/'duplicate_groups.csv',index=False,encoding='utf-8-sig')
    review = prepare_review_files(processed,output,annotated)
    summary.update(version=VERSION,source=str(source.resolve()),source_sha256=sha256(source),
        existing_classification_counts={k:int(v) for k,v in original.relevance.value_counts().items()},
        review_preparation=review,role_scope_counts={k:int(v) for k,v in annotated.role_scope.value_counts().items()},
        position_flag_counts={col:int(annotated[col].sum()) for col in annotated if col.startswith('is_')},
        core_dashboard_candidates=int(annotated.core_dashboard_candidate.sum()),
        caveat='Duplicate candidates and title-derived scope flags are unverified. No unique-vacancy count, deletion, merging, classifier accuracy or complete market-demand claim.',
        protected_input_sha256=hashes,
        implementation_sha256={name:sha256(ROOT/'src'/name) for name in [
            'detect_duplicate_jobs.py','validate_classification.py','classification_policy.py']})
    (output/'step3c_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'policy_snapshot.json').write_text(json.dumps(dict(core_categories=CORE_CATEGORIES,
        extended_scopes=EXTENDED_SCOPES,position_rules=POSITION_RULES),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name in ['detect_duplicate_jobs.py','validate_classification.py','classification_policy.py']:
        (output/name).write_bytes((ROOT/'src'/name).read_bytes())
    examples = []
    for group in groups.head(10).itertuples():
        members = annotated.loc[annotated.duplicate_group_id.eq(group.duplicate_group_id)]
        lines = [f'### {group.duplicate_group_id} ({len(members)} advertisements)']
        for row in members.head(5).itertuples():
            lines.append(f'- {row.advertisement_id}: {row.job_title}; {row.publication_date}; {row.municipality}; vacancies={row.number_of_vacancies}; {row.duplicate_review_status}')
        lines.append('\nDescription excerpt: '+members.description_text.iloc[0][:550].replace('\n',' '))
        examples.append('\n'.join(lines))
    report = f'''# Step 3C duplicate investigation

Source: `{source}`. Version: {VERSION}. All {len(original):,} source advertisements and original prediction fields are preserved.

## Actual findings

- Exact normalized-text groups: {summary['exact_text_groups']:,}, containing {summary['advertisements_in_exact_text_groups']:,} advertisements.
- Exact matching pairs: {summary['exact_pairs']:,}; additional near-matching pairs: {summary['near_pairs']:,}.
- Connected duplicate-review groups: {summary['duplicate_review_groups']:,}; advertisements flagged: {summary['flagged_advertisements']:,}.
- Records deleted: 0. Records merged: 0. Original relevance predictions are unchanged: {json.dumps(summary['existing_classification_counts'])}.

Pair review statuses (not mutually exclusive group counts):

{json.dumps(summary['pair_status_counts'],indent=2)}

## Method and interpretation

Exact identity requires nonempty normalized employer, title and full description equality. Normalization uses Unicode NFKC, case folding, whitespace/separator normalization and equivalent dashes; punctuation, numbers and substantive text otherwise remain meaningful. Missing fields are not assumed equal. Case/whitespace-only changes count as normalized exact, not raw byte equality. Legal employer aliases are not inferred.

Near candidates use word-{summary['settings']['shingle_size']} shingle Jaccard similarity at least {summary['settings']['near_threshold']}, title SequenceMatcher similarity at least {summary['settings']['title_threshold']}, and at least {summary['settings']['min_description_tokens']} description tokens. They are searched within the same exact normalized employer block. Both criteria are required. No all-market semantic equivalence is claimed: employer aliases, agency/client differences, short texts, translations and large title rewrites can be missed. Shared employer boilerplate can create false positives. Exact and near pairs are distinct; counts of their connected components must not be added as if disjoint populations.

Publication days use Europe/Stockholm. Pair evidence records dates/day gap, municipality/region equality and vacancy counts. Different days can mean a repost or separate requisition; different locations can mean local vacancies or multi-location advertising. Different or multiple declared vacancies require review. Identical content/date/location with one declared vacancy on both records is only an exact-repeat *candidate*, never a verified duplicate. Missing metadata, short descriptions and known university link-out boilerplate receive caution flags. Vacancy counts do not establish identity, and summing counts across repeated ads can double-count the same openings.

`duplicate_group_id` is a stable hash of sorted member IDs in a connected exact/near graph. A-B and B-C links do not imply A-C similarity. `exact_duplicate_group_id` independently identifies exact-text subgroups. Group IDs may change if members change. Pair-level statuses/evidence are authoritative; a row's single status shows its highest-priority review caution. Unflagged means no candidate under these rules, not proven uniqueness. No representative record is selected, and no deduplicated demand estimate is produced.

## Files and human validation

`jobs_duplicate_review.csv` preserves every original column and adds duplicate/scope flags. `duplicate_pairs.csv` records every qualifying link; `duplicate_groups.csv` lists group membership and context. Full descriptions remain in the annotated dataset and both review CSVs; excerpts here are examples only.

The existing {review['representative_size']}-row representative sample was {'reused without resampling' if review['representative_reused'] else 'generated using recorded proportional strata'}. It preserves existing human fields, if any, without verifying new labels. `manual_review_diagnostic.csv` contains {review['diagnostic_size']} separate FP/FN/conflict audit cases (seed 44), disjoint from representative IDs. These are suspected risks, not confirmed mistakes. Both review files are new Step 3C copies; Step 3B files remain untouched. Use `notebooks/03_validate_classification.ipynb` to enter/check labels and compute metrics only after explicit human verification. Do not pool samples or treat unresolved human judgments as truth. Duplicate-related sample dependence means ad-level metrics are not unique-vacancy metrics.

Policy flags separate core industry categories, research, statistics and management, plus independent doctoral/internship/thesis/postdoc/student title indicators. They do not change predictions and may miss unlabeled arrangements or ambiguous titles. `core_dashboard_candidate` is an optional scope filter, not a validated count or an implemented dashboard. See `docs/classification_policy.md`.

## Example review groups

'''+'\n\n'.join(examples)
    (output/'duplicate_report.md').write_text(report,encoding='utf-8')
    saved = read_csv(output/'jobs_duplicate_review.csv')
    pd.testing.assert_frame_equal(saved[original.columns],original)
    for path,before in hashes.items():
        if sha256(path)!=before:
            raise AssertionError(f'Protected input changed: {path}')
    verification = dict(input_rows=len(original),output_rows=len(saved),all_original_columns_preserved=True,
        protected_files_unchanged=len(hashes),records_deleted=0,records_merged=0,
        representative_sample_reused=review['representative_reused'],review_samples_disjoint=review['samples_disjoint'])
    (output/'verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['protected_input_sha256','review_preparation']},ensure_ascii=True,indent=2))
    print(f'Output: {output.resolve()}')
    return output,summary


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processed',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--near-threshold',type=float,default=.85)
    parser.add_argument('--title-threshold',type=float,default=.65)
    parser.add_argument('--min-tokens',type=int,default=40)
    parser.add_argument('--shingle-size',type=int,default=5)
    args = parser.parse_args()
    run(args.processed,args.output,Settings(args.near_threshold,args.title_threshold,args.shingle_size,args.min_tokens))
