"""Step 3C review preparation and metrics using verified human labels only."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.classify_jobs_v2 import CATEGORIES, RELEVANCES, text

ROOT = Path(__file__).resolve().parents[1]
HUMAN_FIELDS = ['human_relevance','human_primary_category','human_verified','reviewer','review_notes']
ABSTAIN = '(abstain / not predicted relevant)'


def read_csv(path):
    return pd.read_csv(path,dtype=str,keep_default_na=False,encoding='utf-8-sig')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def latest_step3b(root=ROOT / 'data' / 'processed'):
    candidates = [p.parent for p in Path(root).glob('*/step3b_summary.json')
                  if (p.parent / 'jobs_classified_v2.csv').is_file()]
    if not candidates:
        raise FileNotFoundError('No completed Step 3B dataset found')
    return max(candidates,key=lambda p:p.name)


def representative_sample(frame, size=200, seed=42):
    """Fallback only: proportional strata guarantee each available prediction."""
    n = min(size,len(frame))
    groups = [(key,group) for key,group in frame.groupby('relevance',sort=True)]
    if n < len(groups):
        raise ValueError('Sample too small to represent each relevance prediction')
    quotas = {key:1 for key,_ in groups}
    populations = {key:len(group) for key,group in groups}
    while sum(quotas.values()) < n:
        key = max((k for k in quotas if quotas[k] < populations[k]),
                  key=lambda k:(n*populations[k]/len(frame)-quotas[k],k))
        quotas[key] += 1
    parts = []
    for offset,(key,group) in enumerate(groups):
        part = group.sample(n=quotas[key],random_state=seed+offset).copy()
        part['inclusion_probability'] = quotas[key]/len(group)
        parts.append(part)
    sample = pd.concat(parts).sample(frac=1,random_state=seed).copy()
    sample['sampling_design'] = 'proportional_stratified_minimum_one_per_prediction'
    sample['sampling_seed'] = seed
    sample['sample_order'] = range(1,len(sample)+1)
    for col in HUMAN_FIELDS:
        sample[col] = ''
    return sample


def diagnostic_sample(frame, representative_ids, size=150, seed=44):
    """Balanced error probes, not confirmed errors or a population sample."""
    pool = frame.loc[~frame.advertisement_id.isin(representative_ids)].copy()
    reasons = []
    for row in pool.itertuples():
        review_reasons = json.loads(row.review_reasons or '[]')
        candidates = json.loads(row.candidate_categories or '[]')
        evidence = json.loads(row.evidence or '{}')
        if any('conflict' in reason for reason in review_reasons) or (
                row.relevance == 'relevant' and not row.primary_category):
            reasons.append(('conflicting_categories','Category/title/duty conflict in saved classification'))
        elif row.relevance == 'relevant':
            reasons.append(('suspected_false_positive',
                'Relevant prediction with a review trigger or rejected incidental evidence' if review_reasons or evidence.get('ignored_description_matches')
                else 'Relevant prediction selected for false-positive audit; not a confirmed error'))
        elif candidates or evidence.get('description_matches') or any(
                token in text(row.job_title).split() for token in ['data','ai','ml','bi','snowflake','idq','statistiker']):
            reasons.append(('suspected_false_negative','Not predicted relevant despite a domain title or candidate-category evidence'))
        elif row.relevance == 'irrelevant':
            reasons.append(('suspected_false_negative','Irrelevant prediction selected for false-negative audit; not a confirmed error'))
        else:
            reasons.append(('', ''))
    pool['diagnostic_stratum'] = [reason[0] for reason in reasons]
    pool['diagnostic_reason'] = [reason[1] for reason in reasons]
    pool = pool.loc[pool.diagnostic_stratum.ne('')]
    # Prioritize triggered cases within each stratum, randomize ties deterministically.
    pool = pool.sample(frac=1,random_state=seed)
    pool['diagnostic_priority'] = [int(row.relevance == 'relevant' and bool(
        json.loads(row.review_reasons or '[]') or json.loads(row.evidence or '{}').get('ignored_description_matches',[])))
        for row in pool.itertuples()]
    pool = pool.sort_values('diagnostic_priority',ascending=False,kind='stable')
    groups = [group for _,group in pool.groupby('diagnostic_stratum',sort=True)]
    selected,depth,target = [],0,min(size,len(pool))
    while len(selected) < target:
        for group in groups:
            if depth < len(group) and len(selected) < target:
                selected.append(group.index[depth])
        depth += 1
    sample = pool.loc[selected].copy()
    sample['sampling_design'] = 'targeted_error_audit_excluding_representative'
    sample['inclusion_probability'] = ''
    sample['sampling_seed'] = seed
    sample['sample_order'] = range(1,len(sample)+1)
    for col in HUMAN_FIELDS:
        sample[col] = ''
    return sample


def prepare_review_files(processed, output, annotated, seed=42):
    processed,output = Path(processed),Path(output)
    original = read_csv(processed / 'jobs_classified_v2.csv')
    prior_path = processed / 'manual_review_representative.csv'
    if prior_path.exists():
        sample = read_csv(prior_path)
        if len(sample) != min(200,len(original)) or sample.advertisement_id.duplicated().any():
            raise ValueError('Existing representative sample has unexpected size or repeated IDs')
        if not set(sample.advertisement_id) <= set(original.advertisement_id):
            raise ValueError('Existing sample contains foreign IDs')
        source = original.set_index('advertisement_id')
        for col in original:
            if col != 'advertisement_id':
                pd.testing.assert_series_equal(sample[col].reset_index(drop=True),
                    source.loc[sample.advertisement_id,col].reset_index(drop=True),check_names=False)
        if set(sample.sampling_design) != {'simple_random_without_replacement'}:
            raise ValueError('Existing sample is not the expected uniform random design')
        if len(set(sample.sampling_seed)) != 1:
            raise ValueError('Existing sample mixes seeds')
        old_seed = int(sample.sampling_seed.iloc[0])
        if sample.advertisement_id.tolist() != original.sample(n=len(sample),random_state=old_seed).advertisement_id.tolist():
            raise ValueError('Existing sample does not reproduce from its recorded seed')
        if not set(original.relevance) <= set(sample.relevance):
            raise ValueError('Existing sample does not cover all available relevance predictions')
        if not pd.to_numeric(sample.inclusion_probability).sub(len(sample)/len(original)).abs().lt(1e-12).all():
            raise ValueError('Unexpected representative inclusion probabilities')
        reused = True
    else:
        sample = representative_sample(original,200,seed)
        reused = False
    # Append only new investigation metadata. Existing sample labels/predictions survive.
    annotations = annotated.set_index('advertisement_id')
    for col in annotated:
        if col not in sample:
            sample[col] = annotations.loc[sample.advertisement_id,col].to_numpy()
    target = diagnostic_sample(annotated,sample.advertisement_id,size=150,seed=44)
    for name,frame in [('manual_review_representative.csv',sample),('manual_review_diagnostic.csv',target)]:
        path = output/name
        if path.exists():
            raise FileExistsError(path)
        frame.to_csv(path,index=False,encoding='utf-8-sig')
    metadata = dict(representative_reused=reused, representative_source=str(prior_path.resolve()) if reused else None,
        representative_size=len(sample), representative_seed=int(sample.sampling_seed.iloc[0]),
        representative_ids=sample.advertisement_id.tolist(), diagnostic_size=len(target), diagnostic_seed=44,
        diagnostic_strata={k:int(v) for k,v in target.diagnostic_stratum.value_counts().items()},
        samples_disjoint=set(sample.advertisement_id).isdisjoint(target.advertisement_id),
        human_verified_not_automatically_assigned=True,
        scope='Separate ad-level samples. Duplicate clusters induce dependence; do not interpret as unique-vacancy validation.')
    (output/'review_preparation.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    return metadata


def metric_table(truth, predictions, labels, macro_labels):
    confusion = pd.crosstab(pd.Series(truth,name='Human'),pd.Series(predictions,name='Predicted')).reindex(
        index=labels,columns=labels,fill_value=0)
    rows = []
    for label in labels:
        tp = int(confusion.loc[label,label])
        fp,fn = int(confusion[label].sum())-tp,int(confusion.loc[label].sum())-tp
        rows.append(dict(label=label,precision=tp/(tp+fp) if tp+fp else None,
            recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            support=int(confusion.loc[label].sum()),tp=tp,fp=fp,fn=fn))
    table = pd.DataFrame(rows)
    supported = table.loc[table.label.isin(macro_labels) & table.support.gt(0)]
    n = len(truth)
    summary = dict(evaluated_rows=n, accuracy=float(sum(a==b for a,b in zip(truth,predictions))/n) if n else None,
        macro_precision_supported=float(supported.precision.fillna(0).mean()) if len(supported) else None,
        macro_recall_supported=float(supported.recall.fillna(0).mean()) if len(supported) else None,
        macro_f1_supported=float(supported.f1.fillna(0).mean()) if len(supported) else None,
        weighted_f1=float((supported.f1.fillna(0)*supported.support).sum()/n) if n else None,
        undefined_metrics='None when denominator is zero; macro/weighted averages use zero for undefined scores on supported true classes')
    return dict(summary=summary,per_class=table,confusion_matrix=confusion)


def evaluate_review(review):
    """Unverified labels never enter metrics; unresolved human labels are excluded."""
    required = set(HUMAN_FIELDS) | {'advertisement_id','relevance','primary_category','sampling_design'}
    if missing := required-set(review):
        raise ValueError(f'Missing review fields: {sorted(missing)}')
    data = review.copy(deep=True).fillna('')
    if data.advertisement_id.duplicated().any():
        raise ValueError('Repeated review IDs would inflate metrics')
    if data.sampling_design.nunique() > 1:
        raise ValueError('Representative and targeted samples must be evaluated separately')
    if not set(data.relevance) <= set(RELEVANCES) or not set(data.primary_category) <= set(CATEGORIES)|{''}:
        raise ValueError('Invalid predicted relevance/category values')
    verified = data.human_verified.astype(str).str.strip().str.casefold()
    if set(verified)-{'','true','false'}:
        raise ValueError('human_verified must be blank, true or false')
    rows = data.loc[verified.eq('true')].copy()
    for col in HUMAN_FIELDS:
        rows[col] = rows[col].astype(str).str.strip()
    if not set(rows.human_relevance) <= set(RELEVANCES):
        raise ValueError('Verified relevance must be relevant, irrelevant or uncertain')
    if not set(rows.human_primary_category) <= set(CATEGORIES)|{''}:
        raise ValueError('Unknown verified human category')
    if rows.reviewer.eq('').any():
        raise ValueError('Verified labels require a human reviewer identifier')
    if (rows.human_relevance.ne('relevant') & rows.human_primary_category.ne('')).any():
        raise ValueError('Human category must be empty for irrelevant/uncertain labels')
    resolved = rows.loc[rows.human_relevance.isin(['relevant','irrelevant'])]
    category_rows = rows.loc[rows.human_relevance.eq('relevant') & rows.human_primary_category.ne('')]
    counts = dict(total_records=len(data),verified_records=len(rows),unreviewed_records=int(verified.ne('true').sum()),
        entered_but_unverified=int((verified.ne('true') & (data.human_relevance.ne('')|data.human_primary_category.ne(''))).sum()),
        unresolved_human_relevance=int(rows.human_relevance.eq('uncertain').sum()),
        unresolved_human_category=int((rows.human_relevance.eq('relevant') & rows.human_primary_category.eq('')).sum()),
        predicted_uncertain_relevance=int(data.relevance.eq('uncertain').sum()),
        predicted_relevant_unresolved_category=int((data.relevance.eq('relevant') & data.primary_category.eq('')).sum()),
        relevance_evaluable=len(resolved),category_evaluable=len(category_rows),
        sampling_design=data.sampling_design.iloc[0] if len(data) else '',
        scope='Verified reviewed advertisements only; partial/targeted review and duplicate clusters limit generalization. No unique-vacancy accuracy.')
    result = dict(counts=counts,status='No verified human labels; metrics unavailable' if rows.empty else
                  'Verified labels present; unresolved human truth excluded',relevance=None,primary_category=None)
    if len(resolved):
        result['relevance'] = metric_table(resolved.human_relevance.tolist(),resolved.relevance.tolist(),
                                          RELEVANCES,['relevant','irrelevant'])
    if len(category_rows):
        predictions = [r.primary_category if r.relevance == 'relevant' and r.primary_category else ABSTAIN
                       for r in category_rows.itertuples()]
        result['primary_category'] = metric_table(category_rows.human_primary_category.tolist(),predictions,
                                                  CATEGORIES+[ABSTAIN],CATEGORIES)
        result['primary_category']['summary']['scope'] = 'Verified human-relevant rows with resolved human categories. Predicting irrelevant/uncertain/null counts as abstention/error; spurious relevance on human-irrelevant rows is assessed in relevance metrics.'
    return result


def save_review_edits(path, edits):
    """Only explicit user-entered human fields can be changed in a Step 3C copy."""
    path = Path(path)
    if not (path.parent/'step3c_summary.json').is_file() or path.name not in {
            'manual_review_representative.csv','manual_review_diagnostic.csv'}:
        raise ValueError('Only the review CSVs in a Step 3C workspace may be edited')
    review = read_csv(path)
    for identifier,updates in edits.items():
        mask = review.advertisement_id.eq(str(identifier))
        if mask.sum()!=1 or set(updates)-set(HUMAN_FIELDS):
            raise ValueError('Unknown ID or attempted edit to a protected prediction/source field')
        for col,value in updates.items():
            review.loc[mask,col] = str(value)
    evaluate_review(review)  # Reject inconsistent claims of verification before saving.
    if edits:
        review.to_csv(path,index=False,encoding='utf-8-sig')
    return review
