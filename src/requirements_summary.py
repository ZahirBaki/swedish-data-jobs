"""Counts with explicit denominators for the extracted pilot, never market demand."""
import json

import pandas as pd

from src.extract_requirements import SKILLS


def _boolean(value):return value is True or str(value).casefold()=='true'


def summarize(jobs,facts):
    jobs=jobs.copy();jobs['ai_primary_category']=jobs.ai_primary_category.fillna('(unresolved)')
    total=len(jobs);usable=int(jobs.requirements_data_usable.map(_boolean).sum())
    requested=facts[facts.modality.isin(['required','preferred','alternative_required'])] if len(facts) else facts
    skills=requested[requested.kind.isin(SKILLS)] if len(requested) else requested
    skill_jobs=skills.advertisement_id.nunique() if len(skills) else 0
    education=requested[requested.kind.eq('education')] if len(requested) else requested
    experience=requested[requested.kind.eq('experience')] if len(requested) else requested
    denominators=dict(selected_jobs=total,usable_requirements_jobs=usable,jobs_with_requested_skills=skill_jobs,
        jobs_with_education_requirement=education.advertisement_id.nunique() if len(education) else 0,
        jobs_with_explicit_experience_years=experience.advertisement_id.nunique() if len(experience) else 0,
        skill_counts_unit='distinct advertisements; alternative skills are not individually mandatory',
        scope='Accepted relevant pilot subset only; not representative Swedish job-market demand')
    def counts(frame,columns):
        if frame.empty:return []
        grouped=frame.groupby(columns,dropna=False).advertisement_id.nunique().reset_index(name='job_count')
        grouped['denominator_selected_jobs']=total
        return grouped.sort_values(['job_count']+columns,ascending=[False]+[True]*len(columns)).to_dict('records')
    skill_counts=counts(skills,['kind','value','modality'])
    for r in skill_counts:r['denominator_jobs_with_requested_skills']=skill_jobs
    join=skills.merge(jobs[['advertisement_id','ai_primary_category','region','reporting_scope']],on='advertisement_id',validate='many_to_one') if len(skills) else skills
    by_category=counts(join,['ai_primary_category','value','modality'])
    category_sizes=jobs.groupby('ai_primary_category',dropna=False).size().to_dict()
    for r in by_category:
        r['denominator_category_jobs']=category_sizes.get(r['ai_primary_category'],0)
        r['denominator_category_jobs_with_requested_skills']=int(join.loc[join.ai_primary_category.eq(r['ai_primary_category'])].advertisement_id.nunique())
    languages={}
    for language in ['swedish','english']:
        column=language+'_requirement';specified=int(jobs[column].ne('not_specified').sum())
        languages[language]=[dict(status=status,job_count=int(count),denominator_selected_jobs=total,
            denominator_language_specified_jobs=specified) for status,count in jobs[column].value_counts(dropna=False).items()]
    regions=[]
    for region,group in jobs.groupby('region',dropna=False):
        regions.append(dict(region=region or 'not_specified',selected_jobs=len(group),
            usable_requirements_jobs=int(group.requirements_data_usable.map(_boolean).sum()),
            jobs_with_requested_skills=int(skills[skills.advertisement_id.isin(group.advertisement_id)].advertisement_id.nunique()) if len(skills) else 0,
            denominator_selected_jobs=total))
    seniority=[dict(seniority=k,job_count=int(v),denominator_selected_jobs=total) for k,v in jobs.seniority.value_counts().items()]
    experience_counts=counts(experience,['stated_years','minimum_years','maximum_years','bound_type','modality']) if len(experience) else []
    # JSON has no NaN; null means an absent upper/lower bound, never an inferred one.
    experience_counts=json.loads(pd.DataFrame(experience_counts).to_json(orient='records')) if experience_counts else []
    return dict(denominators=denominators,requested_skills=skill_counts,skills_by_category=by_category,
        language_requirements=languages,education_requirements=counts(education,['value','modality']),
        experience_requirements=experience_counts,seniority=seniority,regions=regions,
        skills_by_region=counts(join,['region','value','modality']),
        reporting_scopes=[dict(scope=k,job_count=int(v),denominator_selected_jobs=total) for k,v in jobs.reporting_scope.value_counts().items()],
        quality_note='Missing/not_specified is not absence. Evidence matching is technical validation, not extraction accuracy.')


def evaluate_requirement_review(review):
    if review.requirement_id.duplicated().any():raise ValueError('Duplicate reviewed requirement IDs')
    flags=review.human_verified.astype(str).str.strip().str.casefold()
    if set(flags)-{'','true','false'}:raise ValueError('Invalid human_verified')
    verified=review[flags.eq('true')]
    if verified.reviewer.str.strip().eq('').any():raise ValueError('Verified extraction labels need a reviewer')
    valid=verified.human_valid.astype(str).str.strip().str.casefold()
    if set(valid)-{'true','false','uncertain'}:raise ValueError('Verified human_valid must be true, false or uncertain')
    resolved=verified[valid.isin(['true','false'])]
    good=resolved.human_valid.astype(str).str.casefold().eq('true')
    modality=resolved.loc[good & resolved.human_modality.ne('')]
    return dict(total_extracted_records=len(review),verified=len(verified),unreviewed=int(flags.ne('true').sum()),
        unresolved_human_judgments=int(valid.eq('uncertain').sum()),
        extraction_precision=float(good.mean()) if len(resolved) else None,
        precision_denominator=len(resolved),
        modality_accuracy=float(modality.modality.eq(modality.human_modality).mean()) if len(modality) else None,
        modality_denominator=len(modality),recall=None,
        note='Output-level manual review only; recall requires an independently enumerated complete gold set. AI output is not ground truth.')
