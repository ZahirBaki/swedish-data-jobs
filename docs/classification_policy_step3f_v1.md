# Proposed final occupational policy — Step 3F v1

Policy version: `step3f-v1`. Proposed for human approval and quality evaluation;
the assistant review is provisional, not independently verified ground truth.
The unit is one advertisement, not a deduplicated vacancy or complete market demand.

## Relevance and reporting scope

Judge the advertised employee's substantial responsibilities. Relevant means a
dedicated Data/AI occupation, including the separately tracked scopes below.
Irrelevant requires affirmative evidence of an unrelated occupation whose duties
do not establish substantial qualifying work. Missing detail means uncertain.
Technical failures are not occupational judgments: leave relevance/category null
and record `Technical failure / requires retry`. Never turn a failed API response
into irrelevant or uncertain. Valid uncertain predictions remain a separate outcome.

| Scope | Inclusion and boundaries |
| --- | --- |
| core_industry | Dedicated engineering, analysis, BI, modeling, ML/AI delivery or data architecture; includes public-sector industry-equivalent work |
| data_ai_governance | Dedicated data stewardship, metadata, data quality, data access or accountable AI governance; exclude generic quality assurance, legal work and general compliance without substantial Data/AI ownership |
| academic_ai_ml_research | Substantive AI/ML methods, algorithms, model development or evaluation in academic research; a doctorate, scientific simulation, imaging, chemistry or occasional off-the-shelf ML alone is insufficient |
| adjacent_borderline | Dedicated statistics, data/AI management and other qualifying adjacent work; relevance can remain uncertain where substantial duties are unclear |
| unrelated | Affirmatively unrelated duties with incidental data/AI use only |
| unresolved | Insufficient or conflicting evidence, or a relevant multi-role ad with no defensible scope |

Internship, thesis, doctoral, postdoctoral and student opportunities are independent
attributes. They neither prove relevance nor automatically exclude it. Store an
array of these attributes (empty for none established). Do not classify a domain
PhD as AI/ML research just because it is a doctoral position. Core reporting should
filter scope and opportunity attributes explicitly; governance, research, management,
statistics and student opportunities remain separately selectable.

## Canonical primary categories

| Stored category | Required substantial responsibilities |
| --- | --- |
| Data Engineer | Build or operate analytical ingestion/transformation pipelines and data platforms |
| Data Analyst | Dedicated analysis, insight and decision support; ordinary operational reporting is insufficient |
| Data Scientist | Statistical/predictive model development or experimentation as substantial work, beyond dashboards or optional ML skills |
| ML Engineer | Develop, deploy, evaluate or operate ML models/systems and their production lifecycle |
| AI Engineer | Develop production AI/LLM applications, agents or AI integrations; using Copilot or AI tools is insufficient |
| Analytics Engineer | Own tested analytical transformations and analytical data models, e.g. dbt |
| BI Developer / BI Analyst | Own BI models, dashboards and reporting solutions |
| Data Warehouse / ETL Developer | Build warehouses or dedicated ETL/ELT workflows |
| Data Architect | Design analytical/enterprise data architecture; generic cloud/DevOps architecture is insufficient |
| Data Governance / Data Quality | Own data or AI governance, stewardship, quality, metadata or accountable model governance |
| AI / ML Research | Dedicated substantive AI/ML methodological research; industry research may have core_industry scope |
| Statistician | Dedicated statistical/biostatistical methods and analysis, tracked separately from core |
| Data / AI Management | Accountable leadership of Data/AI teams, products, delivery or programmes; coding is not required, generic transformation enthusiasm is insufficient |
| Other relevant Data & AI roles | Clearly dedicated qualifying work fitting none of the above; never a fallback for uncertain relevance |

## Boundary decisions

- GIS/cartography, geodata, EDI, domain science and computational engineering:
  evaluate actual duties. Dedicated analytical pipelines or statistical methods
  can qualify; geography, numerical simulation, graphs/databases or software alone
  do not establish a Data/AI occupation.
- Backend/fullstack, SQL, Neo4j, Kubernetes, ORM and hosting alone are ordinary
  software work. Substantial production AI or analytical-platform responsibility
  may qualify; an employer's AI branding or future ambition does not.
- Recruitment, sales, teaching and operations may use AI assistants, dashboards
  and KPIs. Such use and collaboration with data scientists do not qualify.
  Dedicated people analytics or BI ownership can qualify when substantial.
- Management and AI adoption: require accountable Data/AI delivery, governance or
  programme responsibilities; no hands-on-coding requirement for management.
  Vague change-management titles without responsibilities remain uncertain.
- Hybrid jobs: choose the dominant defensible category, list secondary categories
  only when independently supported by substantial duties, and require review
  where dominance is unclear. Multiple distinct occupations in one ad require
  review; relevant with null primary and supported secondary categories is allowed.
- Competency-only, title-only, low-confidence and borderline judgments require
  review. Job duties outrank keywords, optional competencies and colleagues' work.
- Irrelevant and uncertain predictions must have null primary category and empty
  secondary categories. Do not encode speculative categories as accepted labels.

## Source-grounded evidence and validation

Prompt `step3f-1.0` supplies the complete, unmodified title/description as ordered
indexed source spans. Select 1–3 span IDs supporting a resolved decision; an
uncertain decision may select none. The application resolves each selected ID to
the original field, Python Unicode character offsets [start,end), and exact source
substring. It never asks the model to count offsets, rewrite or normalize quotes.
Unknown/duplicate IDs or schema/semantic violations are technical failures. There
is no fallback substitution of text for a failed quotation. Selecting a real span
proves source grounding, not that it logically supports the decision: humans must
assess that distinction. Reasoning is a brief explanation, not another quotation.

All predictions are unverified. Qualitative confidence is not probability. Review
uncertain, low/medium confidence, hybrids, multi-role, unresolved-category and
special-scope cases before publishing statistics. Assistant suggestions and prior
rule/model labels are comparison material only, never supplied as truth or used to
calculate accuracy. Independent human labels and reviewer verification are needed
for precision, recall, F1 or accuracy. The 200 development ads are no longer an
independent holdout for this revised policy; reserve new human-reviewed ads for
quality evaluation before interpreting full-dataset counts.
