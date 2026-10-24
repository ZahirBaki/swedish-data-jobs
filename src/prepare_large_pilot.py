"""Prepare a capped proportional pilot; never calls a model API."""
import argparse
from pathlib import Path
import pandas as pd

from src.classify_jobs_v2 import RELEVANCES
from src.llm_classifier import atomic_json
from src.step3e_classifier import multi_role_candidate
from src.validate_classification import ROOT, latest_step3b, read_csv, sha256


def proportional_sample(frame,excluded_ids,size=200,seed=20261022):
    required={'advertisement_id','job_title','description_text','relevance','primary_category'}
    if required-set(frame):raise ValueError('Missing original rule-based columns')
    if not 1<=size<=200:raise ValueError('Only 1..200-ad sampling is supported')
    if frame.advertisement_id.eq('').any() or frame.advertisement_id.duplicated().any():
        raise ValueError('Source IDs must be nonempty and unique')
    if set(frame.relevance)-set(RELEVANCES):raise ValueError('Invalid source relevance')
    eligible=frame.loc[~frame.advertisement_id.isin(set(excluded_ids))].copy()
    if len(eligible)<size:raise ValueError('Not enough eligible records')
    full=frame.relevance.value_counts().to_dict()
    quotas={key:size*count//len(frame) for key,count in full.items()}
    order=sorted(full,key=lambda key:(-(size*full[key]%len(frame)),key))
    for key in order[:size-sum(quotas.values())]:quotas[key]+=1
    parts=[];breakdown=[]
    for key in sorted(full):
        group=eligible.loc[eligible.relevance.eq(key)].sort_values('advertisement_id')
        n=quotas[key]
        if n>len(group):raise ValueError('Requested stratum quota exceeds eligible population')
        selected=group.sample(n=n,random_state=seed).copy()
        selected['sampling_group']=key
        selected['sampling_seed']=seed
        selected['sampling_design']='proportional_full_candidate_quotas_excluding_development_pilot'
        selected['inclusion_probability']=n/len(group) if len(group) else 0
        selected['sampling_weight']=len(group)/n if n else 0
        selected['multi_role_review_candidate']=[multi_role_candidate(t,d) for t,d in zip(selected.job_title,selected.description_text)]
        parts.append(selected)
        breakdown.append(dict(group=key,full_population=full[key],eligible_population=len(group),selected=n,
                              inclusion_probability=n/len(group) if len(group) else 0))
    sample=pd.concat(parts,ignore_index=True).sample(frac=1,random_state=seed).reset_index(drop=True)
    sample['sample_order']=range(1,len(sample)+1)
    return sample,breakdown


def prepare(output,source=None,previous=None,assistant=None,seed=20261022):
    output=Path(output)
    if output.exists():raise ValueError('Use a new preparation folder')
    source=Path(source) if source else latest_step3b()/'jobs_classified_v2.csv'
    previous=Path(previous) if previous else ROOT/'data/processed/step3d-combined-v2-20261022/dataset/pilot_results.csv'
    assistant=Path(assistant) if assistant else ROOT/'data/reviews/pilot_classification_assistant_reviewed.csv'
    corpus,old,opinions=read_csv(source),read_csv(previous),read_csv(assistant)
    if len(corpus)!=5023 or len(old)!=30 or old.advertisement_id.nunique()!=30:
        raise ValueError('Expected 5023 original candidates and 30 unique development-pilot IDs')
    if not set(old.advertisement_id)<=set(corpus.advertisement_id):
        raise ValueError('Development-pilot IDs must belong to the candidate population')
    if set(opinions.advertisement_id)!=set(old.advertisement_id) or len(opinions)!=30:
        raise ValueError('Assistant review IDs must match development pilot')
    sample,breakdown=proportional_sample(corpus,old.advertisement_id,200,seed)
    output.mkdir(parents=True,exist_ok=False)
    sample.to_csv(output/'representative_200.csv',index=False,encoding='utf-8-sig')
    # Independent review starts without machine predictions, suggestions or labels.
    metadata=[c for c in ['advertisement_id','job_title','description_text','employer','publication_date',
                         'municipality','region','original_advertisement_url','sample_order'] if c in sample]
    human=sample[metadata].copy()
    for name in ['human_relevance','human_primary_category','human_review_notes','human_verified','reviewer']:human[name]=''
    human.to_csv(output/'independent_human_review.csv',index=False,encoding='utf-8-sig')
    # Reuse policy-development cases as separate diagnostics, never representative metrics.
    targeted=corpus.loc[corpus.advertisement_id.isin(old.advertisement_id)].copy()
    targeted['diagnostic_scope']='previous_30_policy_development_not_representative'
    targeted.to_csv(output/'targeted_diagnostic_30.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(breakdown).to_csv(output/'sampling_breakdown.csv',index=False,encoding='utf-8-sig')
    manifest=dict(stage='step3e',sample_size=200,seed=seed,sampling_breakdown=breakdown,
        population_size=len(corpus),eligible_population_size=len(corpus)-len(old),
        excluded_ids=sorted(old.advertisement_id),sample_ids=sample.advertisement_id.tolist(),
        sample_sha256=sha256(output/'representative_200.csv'),
        diagnostic_sha256=sha256(output/'targeted_diagnostic_30.csv'),
        source_path=str(source.resolve()),source_sha256=sha256(source),
        previous_path=str(previous.resolve()),previous_sha256=sha256(previous),
        assistant_path=str(assistant.resolve()),assistant_sha256=sha256(assistant),
        assistant_suggestion_counts=opinions.assistant_suggested_relevance.value_counts().to_dict(),
        assistant_provenance='AI-assisted policy-development opinions, never verified ground truth or prompt inputs',
        sample_scope='Proportional quotas from 5023 candidates, sampled from 4993 eligible ads. Excluded 30 have zero inclusion probability; design-based evaluation applies to eligible frame, not complete Swedish demand.',
        new_paid_requests=0)
    atomic_json(output/'sampling_manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seed',type=int,default=20261022)
    args=parser.parse_args();manifest=prepare(args.output,seed=args.seed)
    print(pd.DataFrame(manifest['sampling_breakdown']).to_string(index=False))


if __name__=='__main__':main()
