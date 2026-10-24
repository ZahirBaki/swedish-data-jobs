"""Local portfolio dashboard. Run: python -m streamlit run dashboard/app.py"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.dashboard_data import (PERIOD, REQUESTED, SKILLS, count_ads, coverage,
                                dashboard_dataset, filter_data, language_counts, load_dataset, skill_counts)

st.set_page_config(page_title='Swedish Data & AI | Requirements Observatory', page_icon='📊', layout='wide')
st.markdown('''<style>
.stApp {background:#f7f9fc;color:#14243b} h1,h2,h3 {letter-spacing:-.035em}
[data-testid="stMetric"] {background:white;border:1px solid #e1e8f0;border-radius:14px;padding:18px}
[data-testid="stSidebar"] {background:#edf2f8}
</style>''', unsafe_allow_html=True)

directory = dashboard_dataset(ROOT / 'data/processed', os.environ.get('JOBS_DASHBOARD_DATA'))


@st.cache_data
def read_data(path, jobs_mtime, evidence_mtime):
    return load_dataset(path)


if not all((directory / name).is_file() for name in
           ('jobs_requirements.jsonl', 'requirements_evidence.csv')):
    st.error('Dashboard data is unavailable. Set JOBS_DASHBOARD_DATA to a completed publication directory, '
             'or use examples/demo_dataset for the fictional demo. See README.md.')
    st.stop()
try:
    all_jobs, all_facts = read_data(str(directory), (directory / 'jobs_requirements.jsonl').stat().st_mtime_ns,
                                   (directory / 'requirements_evidence.csv').stat().st_mtime_ns)
except (OSError, ValueError, KeyError):
    st.error('The selected dataset is incomplete, unreadable, or invalid. Choose a completed publication '
             'or examples/demo_dataset. Existing files have not been changed.')
    st.stop()
is_demo = (directory / 'demo_manifest.json').exists()
expanded = (directory / 'step7_summary.json').exists()
category_column = 'primary_category_label' if expanded else 'ai_primary_category'
scope_column = 'analysis_scope' if expanded else 'reporting_scope'
st.caption('SWEDISH DATA & AI · REQUIREMENTS OBSERVATORY')
st.title('What do these Data & AI jobs ask for?')
if is_demo:
    st.warning('FICTIONAL DEMO: these six handwritten examples are not real advertisements, AI classifications, '
               'audit decisions or market findings. The audit filter demonstrates synthetic UI states.')
st.markdown('**Synthetic examples** · Explore the dashboard without local research data.' if is_demo else
            f'**Collected advertisements: {PERIOD}** · Explore source-backed requirements in the available accepted subset.')
st.info('AI classifications are provisional and have not been independently verified. Statistics describe only the accepted analyzed subset selected by the filters, not the entire collection or Swedish job-market demand. Missing evidence does not mean a requirement is absent.' if expanded else 'This selected pilot subset is not representative of the Swedish job market. AI classifications and deterministic extractions have not been independently human-validated. Missing evidence does not mean a requirement is absent.')

with st.sidebar:
    st.header('Explore the subset')
    st.caption(f'{len(all_jobs)} fictional examples; no real AI classifications or audits.' if is_demo else
               f'{len(all_jobs)} accepted, source-validated relevant advertisements. No reclassification or API calls.')
    filters = {}
    quality = st.selectbox('Inclusion quality', ['All accepted AI-relevant jobs', 'Audit-approved only'], key='inclusion_quality') if expanded else None
    if expanded:
        filters['strict_audit_approved'] = [True] if quality == 'Audit-approved only' else None
        saved_scopes = sorted(all_jobs.ai_scope.fillna('not_specified').replace('', 'not_specified').unique().tolist())
        filters['ai_scope'] = st.multiselect('Saved AI scope (legacy: not_specified)', saved_scopes, default=saved_scopes, key='ai_scope')
        types = sorted({kind for types in all_jobs.analysis_opportunity_types for kind in (types or ['not_flagged_opportunity'])})
        filters['analysis_opportunity_types'] = st.multiselect('Opportunity type', types, default=types, key='analysis_opportunity_types')
        for label, column in [('Opportunity status', 'opportunity_group'), ('Multi-role advertisements', 'multi_role_group')]:
            options = sorted(all_jobs[column].unique().tolist())
            filters[column] = st.multiselect(label, options, default=options, key=column)
    for label, column in [('Occupation scope', scope_column), ('Job category', category_column),
                           ('Region', 'region'), ('Seniority', 'seniority')]:
        options = sorted(all_jobs[column].fillna('not_specified').replace('', 'not_specified').unique().tolist())
        filters[column] = st.multiselect(label, options, default=['core_industry'] if expanded and column == scope_column else options, key=column)
    if not expanded and 'inclusion_method' in all_jobs.columns:
        methods = sorted(all_jobs.inclusion_method.unique().tolist())
        filters['inclusion_method'] = st.multiselect('Inclusion method', methods, default=methods, key='inclusion_method')
    st.caption('Opportunity reporting scope takes precedence over academic scope; use saved AI scope and opportunity-type filters to inspect overlaps. Legacy scope is category-derived.' if expanded else 'Research, governance and adjacent roles remain separate. Legacy scope is category-derived; inspect original AI scope in the job table.')
    st.caption('An empty filter selects no jobs. Reset filters by reloading the app.')
jobs, facts = filter_data(all_jobs, all_facts, **filters)
n = len(jobs)
if not n:
    st.warning('No advertisements match these filters. Select at least one value in each filter.')
    st.stop()
requested = facts[facts.modality.isin(REQUESTED)]
if expanded:
    totals = json.loads((directory / 'step7_summary.json').read_text(encoding='utf-8'))
    cols = st.columns(6)
    for col, label, value in zip(cols,
            (['Synthetic collected examples', 'Real AI classifications', 'Synthetic relevant examples',
              'Examples in current analysis', 'Synthetic audit-approved examples', 'Examples with usable requirements evidence'] if is_demo else
             ['Collected advertisements', 'Successfully classified advertisements', 'AI-classified relevant advertisements',
              'Relevant advertisements in current analysis', 'Strict audit-approved advertisements', 'Jobs with usable requirements evidence']),
            [totals['total_collected'], totals['total_successfully_classified'], totals['expanded_total'], n,
             int(jobs.strict_audit_approved.sum()), int(jobs.requirements_data_usable.eq(True).sum())]):
        col.metric(label, value)
    st.caption("All counts and audit states in this demo are fictional. Requirement offsets refer to the handwritten examples." if is_demo else f"Collection and classification totals are unfiltered. Strict audit-approved and usable-evidence counts use the current filters. {totals['strict_audit_approved']} recorded v3 approvals across the full expanded dataset. V3 audit results are metadata, not an inclusion gate for the main view.")
    with st.expander('Processing decisions for all collected advertisements'):
        st.write(totals['scopes'])
        st.download_button('Download processing status for every collected ID', (directory / 'corpus_coverage.csv').read_bytes(), 'corpus_coverage.csv', 'text/csv')
else:
    cols = st.columns(4)
    cols[0].metric('Analyzed advertisements', n, help=f'{n} of {len(all_jobs)} available accepted jobs match the filters.')
    cols[1].metric('Usable requirements evidence', int(jobs.requirements_data_usable.eq(True).sum()), help='At least one qualified skill, language, education or professional-experience fact. Seniority and metadata alone do not count.')
    cols[2].metric('With requested skills', requested.loc[requested.kind.isin(SKILLS), 'advertisement_id'].nunique())
    cols[3].metric('Categories represented', jobs.ai_primary_category.nunique())
st.caption(f'Every chart uses distinct advertisement IDs. Percentages use {n} filtered advertisements unless explicitly stated otherwise. Skill and education categories can overlap; their percentages need not sum to 100%.')


activation_path = directory / 'policy_activation.json'
if activation_path.exists():
    activation = json.loads(activation_path.read_text(encoding='utf-8'))
    st.info(f"Approved policy {activation['policy_version']}: {activation['new_unique_jobs']} new advertisements passed the automatic duties audit. "
            "Historical jobs retain their original acceptance provenance. Automatic inclusion is not independent human verification.")

audit_report_path = directory / 'automatic_duties_summary.json'
if audit_report_path.exists():
    audit_report = json.loads(audit_report_path.read_text(encoding='utf-8'))
    st.info(f"Automatic duties audit: {audit_report['candidates_pending_policy_approval']} candidates await approval of the proposed policy; "
            f"{audit_report['automatically_approved']} automatically approved. Active v1 requires a person's audit. "
            "Candidates are excluded from all analyzed-job counts and requirement denominators.")

report_path = directory / 'step6_summary.json'
if report_path.exists():
    totals = json.loads(report_path.read_text(encoding='utf-8'))
    st.subheader('Classification coverage of the collected dataset')
    columns = st.columns(6)
    for column, (label, key) in zip(columns, [
        ('Total collected', 'total_collected'), ('Successfully classified', 'total_successfully_classified'),
        ('Relevant and analyzed', 'total_relevant_analyzed'), ('Excluded / irrelevant', 'excluded_irrelevant'),
        ('Uncertain / awaiting audit', 'uncertain'), ('Technical failures', 'technical_failures')]):
        column.metric(label, totals[key])
    if totals.get('active_inclusion_policy') == 'precision-first-v3':
        with st.expander('Processing decisions for all collected advertisements' if totals.get('coverage_scope') == 'all_collected_ids' else 'Inclusion decisions for the 500-job expansion'):
            st.dataframe(pd.DataFrame([{'decision_group': key, 'advertisements': value}
                                      for key, value in totals['expansion_statuses'].items()]),
                         hide_index=True, width='stretch')
            st.caption('Academic, adjacent, governance and restricted opportunities are separate groups. Abstentions and technical failures remain outside requirement statistics.')
            coverage_file = directory / 'corpus_coverage.csv'
            if coverage_file.exists():
                st.download_button('Download processing status for every collected ID', coverage_file.read_bytes(), 'corpus_coverage.csv', 'text/csv')
                st.caption(f"{totals.get('pending_new_requests', 0)} new request units remain unprocessed. Requirement charts use only accepted analyzed advertisements.")
    st.caption('Collection totals are unfiltered. All requirement statistics use only the accepted analyzed subset and the active filters. Newly classified relevant jobs awaiting a duties audit are excluded from those statistics.')


def chart(frame, label, title, color=None):
    st.subheader(title)
    if frame.empty:
        st.caption('No matching evidence in the selected advertisements.')
        return
    if len(frame) > 20:
        st.caption('Showing the 20 most frequent groups. Full evidence remains available in the advertisement explorer.')
    frame = frame.head(20).copy()
    fig = px.bar(frame, x='job_count', y=label, orientation='h', color=color,
                 hover_data=['denominator_jobs', 'percent'], text='job_count',
                 color_discrete_sequence=['#167d9a', '#805ad5', '#ef9a40', '#65a692'])
    fig.update_layout(template='plotly_white', height=max(290, 30 * len(frame) + 80),
                      margin=dict(l=0, r=20, t=10, b=10), yaxis=dict(autorange='reversed'),
                      xaxis_title='Distinct advertisements', yaxis_title=None, legend_title=None,
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, width='stretch')


overview, skills, qualifications, geography, evidence = st.tabs(['Overview', 'Skills & platforms', 'Qualifications', 'Geography', 'Advertisements & evidence'])
with overview:
    a, b = st.columns([2, 1])
    with a: chart(count_ads(jobs, category_column, n), category_column, 'Advertisements by job category')
    with b: chart(count_ads(jobs, scope_column, n), scope_column, 'Occupation scope')
    st.subheader('Evidence coverage')
    st.caption('Qualified requirements or preferences, including alternatives. Language coverage excludes conflicted or unspecified statuses. Metadata and explicit seniority are reported separately.')
    st.dataframe(coverage(jobs, facts), hide_index=True, width='stretch')
with skills:
    choices = {'Required': 'required', 'Preferred': 'preferred', 'Required alternative': 'alternative_required',
               'Merely mentioned': 'mentioned_only', 'Ambiguous mention': 'ambiguous'}
    selected = st.multiselect('Evidence strength', list(choices), default=['Required', 'Preferred', 'Required alternative'])
    st.caption('Required alternatives are choices, not individually mandatory skills. Merely mentioned and ambiguous technologies are not counted as requirements. A job is counted once per skill across the selected strengths.')
    modalities = [choices[x] for x in selected]
    for kinds, title in [(('programming_language',), 'Programming languages'),
                          (('data_tool',), 'Data tools & technologies'), (('cloud_platform',), 'Cloud platforms')]:
        chart(skill_counts(jobs, facts, kinds, modalities), 'value', title)
    st.subheader('Counts by evidence strength')
    detail = facts[facts.kind.isin(SKILLS) & facts.modality.isin(modalities)]
    st.dataframe(count_ads(detail, ['kind', 'value', 'modality'], n), hide_index=True, width='stretch')
    st.caption('Counts by strength may overlap. Do not add them to obtain unique jobs.')
with qualifications:
    a, b = st.columns(2)
    with a: chart(language_counts(jobs, 'swedish'), 'swedish_requirement', 'Swedish language')
    with b: chart(language_counts(jobs, 'english'), 'english_requirement', 'English language')
    st.caption('not_specified means insufficient or conflicting evidence, not “language not required”. Explicitly not required needs a source statement.')
    education = requested[requested.kind.eq('education')]
    chart(count_ads(education, ['value', 'modality'], n), 'value', 'Education requirements & preferences', 'modality')
    st.caption(f'{education.advertisement_id.nunique()} / {n} jobs contain qualified education evidence. Multiple levels may be alternatives; inspect excerpts.')
    chart(count_ads(jobs, 'seniority', n), 'seniority', 'Explicit seniority')
    exp = requested[requested.kind.eq('experience')]
    st.subheader('Explicit professional experience')
    st.caption(f'{exp.advertisement_id.nunique()} / {n} jobs have explicit requested years. Years are not inferred from seniority or academic study durations.')
    if not exp.empty:
        st.dataframe(exp[['advertisement_id', 'stated_years', 'minimum_years', 'maximum_years', 'bound_type', 'modality', 'evidence_text']], hide_index=True, width='stretch')
with geography:
    regional = jobs.assign(region=jobs.region.fillna('not_specified').replace('', 'not_specified'))
    chart(count_ads(regional, 'region', n), 'region', 'Advertised locations by Swedish region')
    st.caption('Source locations are retained without inferred coordinates. Remote-work arrangements do not imply that a job is available in every region.')
    st.dataframe(count_ads(jobs.assign(municipality=jobs.municipality.fillna('not_specified').replace('', 'not_specified')), 'municipality', n), hide_index=True, width='stretch')
    metadata = facts[facts.kind.isin(['employment_type', 'working_time', 'work_arrangement'])]
    st.dataframe(count_ads(metadata, ['kind', 'value'], n), hide_index=True, width='stretch')
with evidence:
    st.subheader('Inspect the advertisements behind a statistic')
    stat = st.selectbox('Restrict to evidence value', ['All advertisements'] + sorted(facts.value.unique().tolist()))
    candidate = jobs if stat == 'All advertisements' else jobs[jobs.advertisement_id.isin(facts.loc[facts.value.eq(stat), 'advertisement_id'])]
    tablecols = ['advertisement_id', 'job_title', 'ai_primary_category', 'reporting_scope', 'region', 'seniority', 'swedish_requirement', 'english_requirement', 'education_level', 'requirements_data_usable']
    tablecols += [c for c in ['inclusion_method', 'inclusion_policy_version', 'dashboard_inclusion_policy_version', 'analysis_scope', 'analysis_opportunity_types', 'multi_role_group', 'strict_audit_approved', 'v3_audit_status'] if c in candidate.columns]
    st.dataframe(candidate[tablecols], hide_index=True, width='stretch')
    st.download_button('Download filtered job table', candidate[tablecols].to_csv(index=False).encode('utf-8-sig'), 'filtered_jobs.csv', 'text/csv')
    if not candidate.empty:
        labels = dict(zip(candidate.advertisement_id, candidate.job_title))
        identifier = st.selectbox('Open original job and supporting evidence', candidate.advertisement_id.tolist(), format_func=lambda x: f'{x} · {labels[x]}')
        job = candidate.set_index('advertisement_id').loc[identifier]
        st.markdown('**' + job.job_title + '**')
        st.caption(f'{job.employer} · {job.region} · Published {job.publication_date}')
        st.write('Classification:', job.ai_primary_category, '· Scope:', job.reporting_scope)
        st.write('Classification provenance:', job.classification_selected_run)
        if 'inclusion_method' in job.index:
            st.write('Inclusion:', job.inclusion_method, ' / ', job.inclusion_policy_version)
            duty_evidence = job.get('inclusion_duties_evidence')
            if isinstance(duty_evidence, list):
                with st.expander('Automatic duties audit evidence'):
                    st.dataframe(pd.DataFrame(duty_evidence), hide_index=True, width='stretch')
        if expanded:
            st.write('Step 7 inclusion:', job.dashboard_inclusion_method, ' / ', job.dashboard_inclusion_policy_version)
            st.write('Historical v3 audit:', job.v3_audit_status, job.v3_audit_reasons)
            st.json(dict(model=job.get('ai_model'), prompt_version=job.get('ai_prompt_version'), saved_source_evidence=job.get('ai_evidence'), saved_source_spans=job.get('ai_evidence_spans')))
        st.write('Original AI scope:', job.get('ai_scope') or job.get('ai_role_scope') or 'Not supplied by legacy classifier')
        st.write('AI reasoning:', job.get('ai_reasoning', ''))
        url = str(job.get('original_advertisement_url', ''))
        if url.startswith(('https://', 'http://')): st.link_button('Open source advertisement', url)
        with st.expander('Full original description', expanded=False): st.text(job.description_text)
        st.caption('Offsets are Python Unicode character offsets [start, end) in the original source field. Context-heading offsets are retained in the downloadable evidence table.')
        evidence_rows = facts[facts.advertisement_id.eq(identifier)]
        st.dataframe(evidence_rows[['kind', 'value', 'modality', 'source_field', 'evidence_start', 'evidence_end', 'evidence_text']], hide_index=True, width='stretch')
        st.download_button('Download full evidence for this job', evidence_rows.to_csv(index=False).encode('utf-8-sig'), f'evidence_{identifier}.csv', 'text/csv')
st.caption('Deterministic extraction · No paid API calls · Counts describe advertisements, not unique vacancies or whole-market demand.')
