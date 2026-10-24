"""Read-only Data Engineer funnel investigation; writes separate analysis artifacts only."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src import corpus_workflow as corpus
from src.dashboard_data import latest_dataset,load_dataset,count_ads
from src.classify_jobs_v2 import TITLE_RULES,text
from src.step4_jobs import SOURCE,validate_classification
from src.run_step4 import write_csv
from src.recover_large_pilot import resilient_atomic_json as atomic_json
from src.validate_classification import ROOT,read_csv,sha256

EXAMPLES=[
 ('30934475','Design och utveckling av datapipelines och dataprodukter','plausible_false_negative','Noun phrase and concatenated/nonstandard Swedish duty heading are missed.'),
 ('30939169','Bygga datapipelines och dataprodukter','plausible_false_negative','Employee duty list under Vad du kommer arbeta med is not recognized as a duties section.'),
 ('31587297','Build and maintain data pipelines using Spark and Flink','plausible_false_negative','What You Will Work On duty heading is not recognized; introductory experienced also triggers the qualification veto.'),
 ('30954479','You will migrate data pipelines','plausible_false_negative','Migrate is absent from the action vocabulary.'),
 ('30932646','you will design and maintain data models and pipelines','plausible_false_negative','Data models and pipelines does not match the contiguous data pipelines object pattern.'),
 ('31447811','Maintain and improve our data pipelines','plausible_false_negative','What you will be doing is not recognized as a duties section.'),
 ('31047493','Designa, utveckla och underhålla skalbara datapipelines','plausible_false_negative','A concrete exact-source duty was found, but not fully covered by selected model spans.'),
 ('30958232','Du bygger, förbättrar och förvaltar CI/CD-pipelines','retain_cautious_exclusion','Ordinary .NET/backend and CI/CD duties do not establish analytical data engineering.'),
 ('30951309','Bygga och förbättra integrationer mellan SAP och andra system','retain_cautious_exclusion','SAP/EDI application integration alone does not establish an analytical data engineering occupation.'),
]


def run(output):
    output=Path(output)
    if output.exists():raise FileExistsError('Use a new analysis folder')
    protected={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'data').rglob('*') if p.is_file()}
    publication=latest_dataset(ROOT/'data/processed')
    jobs,facts=load_dataset(publication)
    source=read_csv(SOURCE);canonical=source.set_index('advertisement_id',drop=False).to_dict('index')
    predictions=json.loads((publication/'classification_provenance.json').read_text(encoding='utf-8'))
    coverage=read_csv(publication/'corpus_coverage.csv').set_index('advertisement_id')
    receipts={r['advertisement_id']:r for r in [json.loads(x) for x in (publication/'inclusion_audit_receipts.jsonl').read_text(encoding='utf-8').splitlines()]}
    de={i:v for i,v in predictions.items() if v.get('ai_relevance')=='relevant' and v.get('ai_primary_category')=='Data Engineer'}
    selected=jobs[jobs.ai_primary_category.eq('Data Engineer')]
    included=set(selected.advertisement_id);all_included=set(jobs.advertisement_id)
    assert included<=set(de)
    engine=corpus.client(ROOT/'data/processed/corpus-run-20261024')
    current_cov,current,_=corpus.reconcile(source,engine,load_dataset(corpus.V3_BASE)[0])
    current_de={i for i,v in current.items() if v.get('ai_relevance')=='relevant' and v.get('ai_primary_category')=='Data Engineer'}
    assert current_de==set(de),'Publication does not match latest saved DE results'
    choices,_,_,_,_=corpus.historical(source)
    direct={i for i,(_,v,_) in choices.items() if v.get('ai_relevance')=='relevant' and v.get('ai_primary_category')=='Data Engineer'}
    assert direct<=set(de)
    by_job=jobs.set_index('advertisement_id').to_dict('index')
    comparison=[]
    for identifier,prediction in sorted(de.items()):
        assert validate_classification(prediction,canonical[identifier])[0]
        receipt=receipts.get(identifier)
        state=coverage.loc[identifier,'final_processing_status']
        reasons=receipt['reasons'] if receipt else ['legacy_classifier_not_compatible_with_v3'] if state=='legacy_relevant' else ['preserved_existing_acceptance_provenance']
        original=canonical[identifier]
        comparison.append(dict(advertisement_id=identifier,job_title=original['job_title'],
            employer=original['employer'],original_advertisement_url=original['original_advertisement_url'],
            ai_model=prediction.get('ai_model'),ai_prompt_version=prediction.get('ai_prompt_version'),
            ai_relevance=prediction['ai_relevance'],ai_primary_category=prediction['ai_primary_category'],
            ai_secondary_categories=prediction.get('ai_secondary_categories'),
            source_validated=True,in_dashboard=identifier in included,
            saved_processing_status=state,inclusion_reasons=reasons,
            audit_execution_status='executed_abstained' if state=='abstention' else
                'legacy_not_compatible_for_v3_audit' if state=='legacy_relevant' else
                'executed_separate_scope' if state in {'adjacent','opportunity'} else 'included',
            original_confidence=prediction.get('ai_confidence'),original_review_flag=prediction.get('ai_review_required'),
            original_multi_role=prediction.get('ai_multi_role'),original_scope=prediction.get('ai_scope'),
            original_opportunity_types=prediction.get('ai_opportunity_types'),
            direct_saved_result=identifier in direct,exact_source_reused=identifier not in direct,
            source_sha256=corpus.source_hash(original['job_title'],original['description_text']),
            ai_evidence=prediction.get('ai_evidence'),ai_evidence_spans=prediction.get('ai_evidence_spans'),
            deterministic_duty_evidence=receipt.get('duty_evidence',[]) if receipt else [],
            inclusion_method=by_job.get(identifier,{}).get('inclusion_method'),
            inclusion_policy_version=by_job.get(identifier,{}).get('inclusion_policy_version'),
            description_text=original['description_text']))
    excluded=[r for r in comparison if not r['in_dashboard']]
    reasons=Counter(reason for r in excluded for reason in r['inclusion_reasons'])
    pure_pattern=[r for r in excluded if r['inclusion_reasons']==['no_category_supported_assigned_duties']]
    pure_span=[r for r in excluded if r['inclusion_reasons']==['duties_not_corroborated_by_validated_description_evidence']]
    new_approved={i for i,r in receipts.items() if r.get('approved') and r.get('saved_category')=='Data Engineer'}
    base_jobs,_=load_dataset(corpus.V3_BASE)
    baseline_de=set(base_jobs[base_jobs.ai_primary_category.eq('Data Engineer')].advertisement_id)
    assert new_approved|baseline_de==included,'Approved/baseline Data Engineers lost or altered in publication'
    assert not jobs.advertisement_id.duplicated().any()
    secondary={i:v for i,v in predictions.items() if v.get('ai_relevance')=='relevant' and 'Data Engineer' in (v.get('ai_secondary_categories') or [])}
    pattern=next(p for _,category,p in TITLE_RULES if category=='Data Engineer')
    uncertain=[dict(advertisement_id=i,job_title=canonical[i]['job_title'],source='title_hint_only_not_a_resolved_category',**v)
        for i,v in predictions.items() if v.get('ai_relevance')=='uncertain' and re.search(pattern,text(canonical[i]['job_title']))]
    examples=[]
    for identifier,quote,assessment,mechanism in EXAMPLES:
        original=canonical[identifier];start=original['description_text'].find(quote)
        if start<0:raise ValueError('Example quote missing from original '+identifier)
        assert identifier in de and identifier not in included
        examples.append(dict(advertisement_id=identifier,job_title=original['job_title'],
            assessment=assessment,assessment_provenance='Assistant qualitative inspection; not independently human verified; no classification or approval change',
            hypothesis=mechanism,source_field='description_text',evidence_start=start,evidence_end=start+len(quote),
            evidence_text=quote,saved_exclusion_reasons=receipts[identifier]['reasons']))
    summary=dict(publication=str(publication),successfully_classified_ads=len(predictions),
        saved_relevance_counts=dict(Counter(v['ai_relevance'] for v in predictions.values())),
        ai_relevant_primary_data_engineer_ads=len(de),direct_saved_primary_de_ads=len(direct),
        exact_source_reused_primary_de_ads=len(de)-len(direct),
        unique_primary_de_source_hashes=len({r['source_sha256'] for r in comparison}),
        dashboard_primary_de=len(included),dashboard_analyzed_jobs=len(jobs),
        dashboard_methods=dict(Counter(selected.inclusion_method)),
        dashboard_primary_de_percent_of_analyzed=100*len(included)/len(jobs),
        primary_de_inclusion_rate=100*len(included)/len(de),
        excluded_primary_de=len(excluded),excluded_statuses=dict(Counter(r['saved_processing_status'] for r in excluded)),
        excluded_reasons_overlapping=dict(reasons),
        exclusively_duty_pattern_abstentions=len(pure_pattern),exclusively_span_coverage_abstentions=len(pure_span),
        solely_deterministic_evidence_gate=len(pure_pattern)+len(pure_span),
        compatible_v3_audits_pending=0,legacy_relevant_without_compatible_v3_audit=sum(r['saved_processing_status']=='legacy_relevant' for r in excluded),
        explicit_uncertain_de_primary_labels=sum(v.get('ai_relevance')=='uncertain' and v.get('ai_primary_category')=='Data Engineer' for v in predictions.values()),
        uncertain_de_title_hints=len(uncertain),all_valid_uncertain_categories_are_null=all(v.get('ai_primary_category') is None for v in predictions.values() if v.get('ai_relevance')=='uncertain'),
        secondary_only_relevant_de_ads=len(secondary),secondary_de_included_under_other_primary=len(set(secondary)&all_included),
        new_approved_primary_de=len(new_approved),baseline_primary_de=len(baseline_de),
        approved_primary_de_lost=0,baseline_primary_de_lost=0,current_saved_de_matches_publication=True,
        api_requests=0,classifications_modified=0,inclusion_rules_changed=False,
        qualitative_examples=len(examples),precision_or_recall_estimated=False)
    output.mkdir(parents=True)
    frame=pd.DataFrame(comparison)
    write_csv(frame,output/'all_ai_data_engineers_vs_dashboard.csv')
    write_csv(frame[frame.in_dashboard],output/'included_data_engineers.csv')
    write_csv(frame[~frame.in_dashboard],output/'excluded_data_engineers.csv')
    write_csv(pd.DataFrame(uncertain),output/'uncertain_data_engineer_title_hints.csv')
    write_csv(pd.DataFrame([dict(advertisement_id=i,job_title=canonical[i]['job_title'],in_dashboard=i in all_included,primary_category=v['ai_primary_category'],ai_secondary_categories=v['ai_secondary_categories']) for i,v in secondary.items()]),output/'secondary_data_engineer_labels.csv')
    write_csv(pd.DataFrame(examples),output/'qualitative_examples.csv')
    atomic_json(output/'summary.json',summary)
    if any(not (ROOT/p).is_file() or sha256(ROOT/p)!=digest for p,digest in protected.items()):raise RuntimeError('Original records changed during analysis')
    assert latest_dataset(ROOT/'data/processed')==publication,'Analysis altered dashboard discovery'
    atomic_json(output/'preservation.json',dict(unchanged_files_verified=len(protected),files=protected,
        source_classifications_sha256=sha256(publication/'classification_provenance.json'),
        saved_audits_sha256=sha256(publication/'inclusion_audit_receipts.jsonl')))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from unittest.mock import patch
    with patch('requests.sessions.Session.request',side_effect=AssertionError('No API calls permitted in this investigation')):
        print(json.dumps(run(args.output),indent=2))

if __name__=='__main__':main()
