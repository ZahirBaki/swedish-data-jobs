# Data & AI occupational classification policy — Step 3E v1

This is a new policy for new Step 3E predictions. The original
`classification_policy.md` remains unchanged and continues to describe earlier
outputs. The classification unit is an advertisement, not a verified unique
vacancy. Candidate collection is not complete Swedish job-market demand.

## Relevance comes from responsibilities

- **Relevant:** substantial advertised responsibilities involve data pipelines or
  platforms, dedicated analysis/BI, statistical or predictive modeling,
  development/operation of ML or production AI systems, data architecture,
  governance/quality, or explicit Data/AI research or management.
- **Irrelevant:** explicit duties describe another occupation and no substantial
  qualifying responsibility is established. Ordinary software development, SQL,
  database use, occasional analysis or AI tool use do not establish relevance.
- **Uncertain:** duties are insufficiently described, contradictory, peripheral,
  or cannot be attributed to the advertised employee. Missing keywords or a
  missing description never prove irrelevance. Do not invent duties from titles.

A generic title can qualify when substantive duties clearly meet this policy.
An explicit occupational title can support evidence, but responsibilities take
precedence when the title and work conflict. Required-competency-only decisions
or unclear dominance in hybrid roles require review.

## Categories (canonical stored spelling)

| Category | Substantive work |
| --- | --- |
| Data Engineer | Building/operating data ingestion, transformation pipelines and analytical data platforms |
| Data Analyst | Analysis, reporting, business insights and decision support |
| Data Scientist | Statistical/predictive modeling, experimentation or model development as substantial duties |
| ML Engineer | Developing, deploying and operating ML models and systems |
| AI Engineer | Developing production AI/LLM applications, agents and integrations |
| Analytics Engineer | Tested analytical transformations and models, e.g. dbt |
| BI Developer / BI Analyst | BI data models, dashboards and reporting solutions |
| Data Warehouse / ETL Developer | Data warehouses and ETL/ELT workflows |
| Data Architect | Data platform/enterprise data architecture and design |
| Data Governance / Data Quality | Governance, quality, stewardship, catalog/metadata and access-control practices |
| AI / ML Research | Academic or dedicated AI/ML methodological research; separate from industry core |
| Statistician | Dedicated statistical/biostatistical work; separate optional scope |
| Data / AI Management | Managing Data/AI delivery, teams or programmes; separate optional scope |
| Other relevant Data & AI roles | Substantial qualifying duties that genuinely fit no named category; not a catch-all for AI tool users |

## Boundaries clarified by the small pilot

- **Fullstack/backend:** application development, ORM/entity modeling (including
  EF Core), SQL databases, cloud hosting and microservices alone are ordinary
  software work. Require actual pipelines/platforms, substantive ML model
  development or production AI responsibilities for inclusion. A vague promise
  to explore AI or optional Copilot use is insufficient.
- **HR/recruitment:** AI recruiting tools, operational KPIs, occasional dashboards
  or participating in analysis do not automatically establish a Data/AI role.
  Dedicated ownership of substantial people-analytics/BI deliverables may qualify
  as analyst/BI work. Unclear extent or dominance requires uncertainty/review.
- **Analyst versus scientist:** dashboards, reporting, SQL, visualizations and
  business insights support analyst/BI categories. Optional ML experience or
  using an AI-powered analysis product does not establish Data Scientist.
  Require substantial modeling/experimentation responsibilities for scientist.
- **Generic titles:** explicitly developing ML models, analytical pipelines or
  production AI can qualify despite a generic engineer/developer title. Neither
  generic title exclusion nor a single AI mention is sufficient.
- **Research:** AI/ML academic research is relevant in its separate category and
  scope. Do not rebrand domain research as AI/ML research solely because it uses
  an off-the-shelf tool. Concrete ML/model development duties matter.
- **Internships:** internship, doctoral, postdoctoral, thesis and student status
  are independent attributes, never reasons to change relevance. Track them
  separately; a proposed industry-core reporting filter may exclude these scopes
  without declaring their advertisements irrelevant.
- **Multi-role advertisements:** identify explicit lists of distinct occupations
  as multi-role review candidates. If there is no defensible dominant role, use
  relevant with null primary category, retain plausible secondary categories and
  require review. Do not count the same ad once per category. A candidate flag
  is not a verified vacancy count or a complete multi-role detector.

## Evidence and review

Return the structured seven-field AI schema. Evidence is a short source quotation
of at most 160 characters, preferably a few words copied from one field. Keep the
model quotation unchanged. Exact matching is preferred; the existing
`evidence-whitespace-1` policy may collapse whitespace runs only, with source
offsets/original span retained and review required. Never accept invented words,
paraphrases, changed punctuation/case/prepositions, omissions, or cross-field joins.
Relevant/irrelevant decisions need nonblank matching evidence; uncertain can use
an empty string. Reasoning explains duties and inclusion/exclusion without
fabricating quotations. Confidence is qualitative, not a calibrated probability.

Uncertain relevance, null relevant primary categories, low confidence,
competency-only evidence, unclear hybrid dominance and multi-role candidates
require review. No AI-generated output is human-verified.

## Evaluation and reporting limits

The assistant's 10 relevant/20 irrelevant judgments about the 30-job pilot are
provisional AI-assisted opinions, not independent human labels. Neither rules,
another AI model, nor those opinions are ground truth. Accuracy/precision/recall/F1
require independently entered labels, explicit human verification and an
identified reviewer; unresolved human judgments remain separately reported.

Use a new proportional-strata pilot excluding the 30 policy-development ads.
Report technical coverage, disagreements and abstentions separately from quality.
The sampling frame excludes those 30 and consists of collected candidates, not
the complete job market. Report inclusion probabilities and strata weights, and
never pool targeted diagnostics with representative metrics as an unweighted
population sample. Duplicates remain advertisements pending human adjudication.
