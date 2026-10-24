"""Offline Step 7 publication. No classification, requests, retries or paid execution."""
import argparse
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.corpus_workflow import client, reconcile
from src.dashboard_data import load_dataset, validate, coverage as requirement_coverage
from src.extract_requirements import extract_job
from src.preview_duties_v3 import RESTRICTED_TITLE, RESEARCH_TITLE
from src.recover_large_pilot import resilient_atomic_json as atomic_json
from src.run_step4 import write_csv, json_rows
from src.step4_jobs import SOURCE, reporting_scope, validate_classification
from src.validate_classification import ROOT, read_csv, sha256

POLICY = 'step7-ai-relevant-v1'
BASE = ROOT / 'data/processed/corpus-final-20261024'
DEFAULT_OUTPUT = ROOT / 'data/processed/step7-expanded-20261024'
POLICY_PATH = ROOT / 'docs/step7_ai_relevant_policy_v1.md'


def eligible(prediction, source):
    valid, reason = validate_classification(prediction, source)
    if not valid:
        return False, reason
    if prediction.get('ai_relevance') != 'relevant':
        return False, 'saved_' + str(prediction.get('ai_relevance'))
    return True, 'accepted_schema_valid_source_grounded_ai_relevant'


def scope_metadata(row):
    """Disjoint reporting lanes; preserve raw model scopes and independent flags."""
    raw = row.get('ai_scope') or row.get('ai_role_scope')
    opportunities = list(row.get('ai_opportunity_types') or [])
    for flag, value in [('ai_is_doctoral','doctoral'), ('ai_is_postdoctoral','postdoctoral'),
                        ('ai_is_internship','internship'), ('ai_is_thesis','thesis'),
                        ('ai_is_student_work','student')]:
        if row.get(flag) is True and value not in opportunities:
            opportunities.append(value)
    restricted_title = bool(RESTRICTED_TITLE.search(row['job_title']))
    if restricted_title and not opportunities:
        opportunities.append('restricted_title_indicator')
    research = raw == 'academic_ai_ml_research' or bool(RESEARCH_TITLE.search(row['job_title']))
    scope = ('opportunities' if opportunities else 'academic_ai_ml_research' if research else
             'adjacent' if raw == 'adjacent_borderline' else
             'governance' if raw == 'data_ai_governance' or row.get('ai_primary_category') == 'Data Governance / Data Quality' else
             'core_industry' if raw == 'core_industry' else reporting_scope(row.get('ai_primary_category')))
    if scope == 'research': scope = 'academic_ai_ml_research'
    return dict(analysis_scope=scope, scope_derivation='saved_scope_plus_saved_flags_and_title_indicators' if raw else 'legacy_category_plus_saved_flags_and_title_indicators',
                analysis_opportunity_types=sorted(opportunities),
                opportunity_group='restricted_or_student_opportunity' if opportunities else 'ordinary_employment',
                multi_role_group='multi_role' if row.get('ai_multi_role') is True or row.get('ai_multi_role_review_candidate') is True else 'not_flagged_multi_role',
                primary_category_label=row.get('ai_primary_category') or 'Unresolved primary category')


def audit_history():
    """Retain actual saved decisions, never promote a historical acceptance to v3 approval."""
    result = {}
    for path in sorted((ROOT / 'data/processed').glob('*/inclusion_audit_receipts.jsonl'), key=lambda p:(p.stat().st_mtime_ns,str(p))):
        digest = sha256(path)
        for line in path.read_text(encoding='utf-8').splitlines():
            receipt = json.loads(line)
            if receipt.get('policy_version', receipt.get('inclusion_policy_version')) != 'precision-first-v3':
                continue
            result.setdefault(str(receipt['advertisement_id']), []).append(dict(receipt=receipt, path=str(path), file_sha256=digest))
    return result


def verify_evidence(jobs, facts):
    by_id = jobs.set_index('advertisement_id').to_dict('index')
    if facts.requirement_id.duplicated().any():
        raise ValueError('Duplicate requirement IDs')
    for fact in facts.to_dict('records'):
        source = by_id[fact['advertisement_id']].get(fact['source_field']) or ''
        for start, end, text in [('evidence_start','evidence_end','evidence_text'), ('match_start','match_end','matched_text')]:
            if source[int(fact[start]):int(fact[end])] != fact[text]:
                raise ValueError('Requirement evidence does not match original source')


def breakdown(jobs, column):
    return {str(k):int(v) for k,v in jobs.groupby(column, dropna=False).advertisement_id.nunique().items()}


def publish(output=DEFAULT_OUTPUT, baseline=BASE):
    output, baseline = Path(output), Path(baseline)
    if output.exists(): raise FileExistsError('Use a new Step 7 publication directory')
    protected = {str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'data').rglob('*') if p.is_file()}
    source = read_csv(SOURCE)
    if len(source) != 5023 or source.advertisement_id.duplicated().any():
        raise ValueError('Expected 5,023 unique collected advertisements')
    base_jobs, base_facts = load_dataset(baseline)
    # The read-only client supplies cache validation and estimates; it never classifies.
    # Hard network guard remains in force for the entire offline discovery/extraction operation.
    with patch('requests.sessions.Session.request', side_effect=AssertionError('Step 7 forbids network requests')):
        cover, predictions, snapshots = reconcile(source, client(output), base_jobs)
        source_by_id = source.set_index('advertisement_id', drop=False).to_dict('index')
        base_by_id = {r['advertisement_id']:r for r in json_rows(base_jobs)}
        origins = cover.set_index('advertisement_id').classification_origin.to_dict()
        histories = audit_history()
        selected, decisions, new_facts = [], [], []
        reused_ids, extracted_ids, chosen_audits = set(), set(), []
        for identifier in sorted(predictions):
            prediction, original = predictions[identifier], source_by_id[identifier]
            passing, reason = eligible(prediction, original)
            decisions.append(dict(advertisement_id=identifier, included=passing, reason=reason,
                                  ai_relevance=prediction.get('ai_relevance'), classification_origin=origins[identifier]))
            if not passing: continue
            if identifier in base_by_id:
                row = dict(base_by_id[identifier])
                # Reuse only when saved AI decision and original source remain identical.
                for field in ['job_title','description_text','ai_relevance','ai_primary_category','ai_prompt_version']:
                    expected = original[field] if field in original and not field.startswith('ai_') else prediction.get(field)
                    if row.get(field) != expected: raise ValueError('Baseline decision/source conflict: ' + identifier + '/' + field)
                reused_ids.add(identifier)
            else:
                row = dict(original, **prediction)
                row.update(classification_source_valid=True, classification_human_verified=False,
                           classification_selected_run=origins[identifier], step4_disposition='eligible_relevant',
                           reporting_scope=reporting_scope(prediction.get('ai_primary_category')))
                row, facts = extract_job(row)
                new_facts.extend(facts); extracted_ids.add(identifier)
            if not eligible(row, original)[0]:
                raise ValueError('Preserved classification failed validation: ' + identifier)
            history = histories.get(identifier, [])
            receipt = history[-1]['receipt'] if history else None
            strict = bool(receipt and receipt.get('approved') is True)
            # Some baseline publications have v3 inclusion metadata but no combined receipt file.
            if not history and row.get('inclusion_policy_version') == 'precision-first-v3' and row.get('inclusion_method') == 'automated_deterministic':
                strict = True
            row.update(scope_metadata(dict(original, **prediction)))
            row.update(dashboard_inclusion_policy_version=POLICY, dashboard_inclusion_method='accepted_source_validated_ai',
                       strict_audit_approved=strict,
                       v3_audit_status='approved' if strict else receipt.get('inclusion_status',receipt.get('preview_decision','recorded_not_approved')) if receipt else 'no_recorded_v3_audit',
                       v3_audit_reasons=receipt.get('reasons',[]) if receipt else [],
                       v3_audit_provenance=history,
                       classification_provisional=True)
            chosen_audits.append(dict(advertisement_id=identifier, strict_audit_approved=strict, history=history))
            selected.append(row)
        jobs = pd.DataFrame(selected)
        facts = pd.concat([base_facts[base_facts.advertisement_id.isin(reused_ids)],pd.DataFrame(new_facts)], ignore_index=True)
        validate(jobs, facts); verify_evidence(jobs, facts)
        missing_previous = sorted(set(base_by_id)-set(jobs.advertisement_id))
        if missing_previous: raise ValueError('Previously included jobs missing: ' + str(missing_previous))
        if not all(eligible(predictions[i],source_by_id[i])[0] for i in jobs.advertisement_id):
            raise ValueError('Ineligible classification in publication')
        core = jobs[jobs.analysis_scope.eq('core_industry')]
        report = dict(policy_version=POLICY, total_collected=len(source),
            total_successfully_classified=int(cover.successfully_classified.sum()),
            ai_classified_relevant=len(jobs), expanded_total=len(jobs), core_industry_total=len(core),
            previous_analyzed=len(base_jobs), previous_preserved=len(reused_ids), new_unique_jobs=len(extracted_ids),
            usable_requirements_jobs=int(jobs.requirements_data_usable.eq(True).sum()),
            core_usable_requirements_jobs=int(core.requirements_data_usable.eq(True).sum()),
            strict_audit_approved=int(jobs.strict_audit_approved.sum()),
            categories=breakdown(jobs,'primary_category_label'), core_categories=breakdown(core,'primary_category_label'),
            scopes=breakdown(jobs,'analysis_scope'), multi_role=breakdown(jobs,'multi_role_group'),
            previous_not_preserved=missing_previous, reextracted_existing_jobs=0,
            same_content_additional_ids=int(jobs.duplicated(['job_title','description_text']).sum()),
            paid_api_requests=0, api_cost_usd=0, technical_failures_in_analysis=0,
            uncertain_in_analysis=0, irrelevant_in_analysis=0,
            selection_precedence='Existing corpus reconciler: latest valid Step 3F 1.1 preferred; otherwise latest valid historical decision. Source-identical cache reuse keeps distinct advertisement IDs.',
            default_view='All accepted AI-relevant jobs; core_industry scope; all multi-role statuses',
            limitations='Provisional AI classifications, not independently verified. Requirements are deterministic source-backed extractions. Advertisement counts are not vacancy counts or representative whole-market estimates.')
        if sum(report['categories'].values()) != len(jobs): raise ValueError('Primary-category reconciliation failed')
        cover['step7_status'] = cover.advertisement_id.map({r['advertisement_id']:('included_ai_relevant' if r['included'] else r['reason']) for r in decisions}).fillna(cover.request_action)
        old = base_jobs.groupby('ai_primary_category',dropna=False).advertisement_id.nunique().to_dict()
        comparisons = [dict(primary_category=k, previous=int(old.get(k,0)), expanded=v, core_industry=report['core_categories'].get(k,0), added=v-int(old.get(k,0))) for k,v in report['categories'].items()]
        output.mkdir(parents=True)
        write_csv(jobs,output/'jobs_requirements.csv')
        (output/'jobs_requirements.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in json_rows(jobs)),encoding='utf-8')
        write_csv(facts,output/'requirements_evidence.csv')
        write_csv(requirement_coverage(jobs,facts),output/'requirement_coverage.csv')
        write_csv(cover,output/'corpus_coverage.csv')
        write_csv(pd.DataFrame(decisions),output/'inclusion_decisions.csv')
        write_csv(pd.DataFrame(comparisons),output/'category_reconciliation.csv')
        write_csv(jobs.groupby(['analysis_scope','primary_category_label'],dropna=False).advertisement_id.nunique().reset_index(name='advertisements'),output/'category_scope_breakdown.csv')
        atomic_json(output/'classification_provenance.json',{i:predictions[i] for i in jobs.advertisement_id})
        atomic_json(output/'v3_audit_history.json',chosen_audits)
        atomic_json(output/'step7_summary.json',report)
        (output/'report.md').write_text('# Step 7 reconciliation\n\n'+
            f"Expanded {len(jobs)}; core industry {len(core)}; previous {len(base_jobs)}; preserved {len(reused_ids)}; newly included {len(extracted_ids)}.\n\n"+
            f"Usable requirements: {report['usable_requirements_jobs']} overall; {report['core_usable_requirements_jobs']} core. Strict v3-approved: {report['strict_audit_approved']}.\n\n"+
            'The difference is the explicitly authorized replacement of the v3 mandatory gate by accepted, schema-valid, source-grounded AI relevance. No classifications changed. V3 abstentions and exclusions remain in audit metadata; they no longer block the main dataset. Unknown outcomes, technical failures, irrelevant and uncertain decisions are not included.\n\n'+
            'Primary categories count each advertisement once. Secondary categories never add advertisements. Identical source copies retain distinct advertisement IDs. Eight unresolved primary categories, if present, are labelled without changing the saved AI category.\n\n'+
            'Core is the default scope. Opportunity flags/title indicators take precedence over academic, adjacent, governance and core reporting lanes; original scopes and overlapping flags remain available. Multi-role status is independently filterable, with both statuses selected by default. Legacy scope is explicitly marked as derived. No duties audit or confidence gate was silently retained.\n\n'+
            'Existing extracted jobs and facts were reused unchanged. New jobs use the existing deterministic extractor with exact validated evidence and match offsets. Requirement modalities remain required, preferred, alternative, mentioned, ambiguous or negated. Missing evidence does not establish absence.\n\n'+
            'All counts describe a provisional selected AI-relevant subset, not independently verified occupations or total Swedish market demand. See category_reconciliation.csv, category_scope_breakdown.csv, corpus_coverage.csv and inclusion_decisions.csv for every count and ID.\n',encoding='utf-8')
        changed = [name for name,digest in protected.items() if not (ROOT/name).exists() or sha256(ROOT/name)!=digest]
        if changed: raise ValueError('Historical data changed: '+str(changed))
        atomic_json(output/'preservation_manifest.json',dict(files=protected, verified_unchanged=True, verified_file_count=len(protected)))
        # Manifest is written last: incomplete outputs cannot become the dashboard source.
        atomic_json(output/'manifest.json',dict(stage='step7-ai-relevant', policy_version=POLICY,
            policy_sha256=sha256(POLICY_PATH),baseline=str(baseline),source_sha256=sha256(SOURCE),
            history_snapshots=snapshots, jobs_sha256=sha256(output/'jobs_requirements.jsonl'),
            evidence_sha256=sha256(output/'requirements_evidence.csv'), report=report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--baseline',type=Path,default=BASE)
    args=parser.parse_args()
    print(json.dumps(publish(args.output,args.baseline),ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
