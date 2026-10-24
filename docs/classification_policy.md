# Classification and reporting policy

This policy applies to Step 3B predictions and Step 3C review preparation. It does not change source advertisements, infer complete Swedish job-market coverage, or assert that an advertisement is a unique vacancy. The unit of the current dataset and representative review sample is an **advertisement**. Predictions, human labels, duplicate evidence and reporting scope are separate concepts.

## Relevance and primary category

- **Relevant:** an explicit Data/AI role title or specific advertised duties/required competencies support inclusion. A relevant advertisement may have an unresolved primary category.
- **Irrelevant:** explicit unrelated occupational duties/title support exclusion. Missing keywords or missing descriptions are not proof of irrelevance.
- **Uncertain:** role-specific evidence is insufficient, contradictory or cannot be attributed to the advertised role. Keep it for human review; do not force a category.

`primary_category` is one canonical category or null. Legacy `Irrelevant` and `Uncertain / Needs Review` category strings are preserved in original classifications but are relevance states in the new schema. No automated record becomes human-verified merely because a rule matched or review_required is false.

## Core industry categories

| Canonical category in the dataset | Included work |
| --- | --- |
| Data Engineer | Data ingestion, data pipelines and production data platforms |
| Data Analyst | Data analysis with reporting, insights and decision-support deliverables |
| Data Scientist | Predictive/statistical learning models and analytical modeling as substantive duties |
| ML Engineer | Building, deploying and operating machine-learning systems |
| AI Engineer | Building AI/LLM applications, agents and AI systems |
| Analytics Engineer | Tested analytical models and transformation layers, including dbt models |
| BI Developer / BI Analyst | Business intelligence reporting, dashboards and analytical solutions; this is the stored spelling of “BI Developer / Analyst” |
| Data Warehouse / ETL Developer | Warehouse/datalager development and ETL/ELT workflows |
| Data Architect | Explicit data/AI architecture and enterprise data-design roles |
| Data Governance / Data Quality | Data governance, quality, stewardship, ownership and master-data management; this is the stored spelling of “Data Governance / Quality” |

Default **core industry** scope means relevant advertisements with a resolved category in this table, excluding identified doctoral, internship, thesis, postdoctoral and student-work positions. This is a proposed reporting filter, not a dashboard or a final validated demand count. Broad Swedish dataingenjör/datavetare and generic data administrator/specialist titles still require context review. Job titles alone do not establish employment permanence.

## Separate optional scopes and position indicators

Step 3C appends `role_scope` without changing relevance or primary_category:

| Scope | Inclusion policy |
| --- | --- |
| `core_industry` | The ten categories above; independent position flags still apply |
| `ai_ml_research` | AI/ML research roles, including explicitly scoped academic research |
| `statistician` | Statistician/biostatistician positions; toggle separately from the industry core |
| `data_ai_management` | Data/AI/analytics team or programme management; generic managers at AI companies are insufficient |
| `other_relevant` | Original Other relevant Data & AI roles; retain but inspect rather than mixing into core |
| `relevant_unresolved_category` | Relevant but primary role unresolved; do not allocate to a core category silently |
| `uncertain`, `irrelevant` | Prediction states kept outside relevant-category totals |

Independent title-based Boolean flags are `is_doctoral`, `is_internship`, `is_thesis`, `is_postdoctoral` and `is_student_work`. They may overlap: a doctoral internship can have both flags. They are retained **even when relevance/category is uncertain**, so these cases remain separately reviewable. A bare PhD credential does not automatically mean a doctoral position. Title indicators can miss unlabeled positions or misread abbreviations such as LIA; they are provisional and not verified employment-type facts. No education requirements are extracted from descriptions.

`core_dashboard_candidate` is true only for core-category relevant predictions with none of these position flags. Future reporting may toggle research, statistics, management, internship and doctoral scopes independently. Do not sum overlapping position flags, treat uncertain predictions as negatives, or add resolved-category counts to relevance totals without accounting for null categories. Document any selected scope and human-validation status with published statistics.

## Hybrid roles and category conflicts

Read actual responsibilities and required experience. Generic software/engineering, GIS, cybersecurity, chemistry, physics and management titles can hide relevant Data/AI work, but occasional data analysis or generic database use is insufficient by itself. Sales, recruitment, legal and finance roles at AI businesses are not automatically AI roles.

An explicit `Data Engineer / Data Architect` title can be relevant with null primary_category. Retain candidate categories and evidence for review rather than counting the same ad once in each category. Documented nested patterns can select the narrower warehouse/ETL category over a generic pipeline rule, and BI deliverables over generic analysis. Research titles can scope AI/ML work to research. Other incompatible primary roles require review. Managerial titles with concrete Data/AI responsibilities are scoped as management rather than automatically counted as hands-on developers.

## AI tools, colleagues and optional competencies

Using ChatGPT, Copilot, AI-assisted coding, an AI recruitment tool or general office automation does not establish AI engineering. Collaboration with data scientists or a team containing an HR Data Analyst does not make the advertised employee a data scientist/analyst. Employer service portfolios and company branding are background, not duties.

Concrete candidate responsibilities (for example building data pipelines or AI agents) and required experience performing those tasks can support relevance. Optional/nice-to-have capabilities do not establish primary duties. Required-competence-only decisions need review. If generic software work is combined with substantive Data/AI duties, retain the evidence and review the dominant role instead of using a single AI keyword. No paid LLM classification or skill extraction is implemented.

## Duplicate investigation is not vacancy adjudication

Exact normalized employer/title/description groups and near-text groups are candidates for inspection. Compare publication dates, locations, vacancy counts, requisition details in the text and full descriptions. Different IDs or a different date do not prove a separate opening; identical text does not prove a duplicate vacancy. One advertisement may declare several vacancies, and multi-location advertising may represent one or many openings.

All records and source predictions remain intact. Connected near-duplicate groups are not necessarily pairwise equivalent. Only a subsequent explicit human adjudication could justify a vacancy-level grouping policy. Do not automatically delete rows, select a representative advertisement, sum duplicate vacancy counts, or report group counts as unique job demand.

## Human validation and metrics

Only labels with `human_verified=true` and an identified human reviewer enter evaluation. Unverified typed labels, AI predictions, prior assistant inspection notes and automated review flags are not ground truth. Verified human `uncertain` judgments remain unresolved and are excluded from resolved-ground-truth metrics. A verified relevant row with blank category is evaluated for relevance but excluded from primary-category metrics.

Evaluate representative and diagnostic samples separately. The reused representative sample is uniform over advertisements; the targeted FP/FN/conflict sample is diagnostic and cannot establish population accuracy. Partial/selective review and correlated repeated advertisements limit generalization. Do not reinterpret ad-level metrics as independent unique-vacancy metrics.

Relevance precision, recall and F1 use verified resolved human relevant/irrelevant labels. Predicted uncertain is an abstention and counts as a missed positive/negative in recall; it is shown in the confusion matrix. Primary-category metrics use verified human-relevant rows with resolved human categories. Predicting irrelevant, uncertain or null for these rows counts as category abstention/error. Spurious relevance on human-irrelevant rows is measured in relevance metrics, not the conditional primary-category table. Confusion matrices use human rows and predicted columns. Undefined per-class scores are shown as null; macro averages use supported human classes only and zero for undefined scores on those classes.

Always report reviewed/verified, unreviewed, unverified-entered and unresolved counts, sample design, evaluated denominators and abstentions alongside metrics. No metrics are claimed before verified labels exist.
