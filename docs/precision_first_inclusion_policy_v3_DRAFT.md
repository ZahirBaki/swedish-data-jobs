# Precision-first v3: practical automatic duties audit (PROPOSAL, NOT ACTIVE)

Version: precision-first-v3-draft. Engine: duties-v3-preview-1.0.
The user approved moving to a versioned automatic-inclusion approach and requested
this preview before activation. This proposal is not yet applied to any dataset.
Existing v1 and v2 documents and all historical classifications remain unchanged.

## Concrete changes from v1/v2

Replace the per-person duties/category audit with a versioned, deterministic source
responsibilities audit for clear core-industry jobs. Do not require two sentences,
a prescribed pronoun, compound verbs, or empty secondary-category arrays. A single
explicit substantial responsibility establishing the saved primary occupation can
qualify. Secondary labels do not establish primary-category support and do not
alone imply a multi-role advertisement. Preserve those labels unchanged.

Recognize English and Swedish employee statements, responsibilities sections,
imperative duty lists and noun-form responsibilities (e.g. development of analytical
data platforms). Evaluate actions with category-specific work objects in the same
clause. Do not use bare AI/tool terms or titles as duties. Role-overview sections
may establish employee context; qualifications, company descriptions and benefits
end that context. Statements about colleagues, optional knowledge, employer
branding and ordinary software remain insufficient. Weak or missing support means
abstain; there is no goal or quota for acceptance.

## All required gates for automatic core inclusion

1. A saved complete Step 3F 1.1 relevant classification revalidates against the
   unmodified collected source and exact original span evidence/offsets.
2. Primary category is named and supported by its own concrete action-object rule.
   No inference from a secondary category and no generic/unknown category fallback.
3. At least one employee-assigned category-specific responsibility is explicit in
   description text. The entire matched action/object finding must be covered by the union of
   source-validated selected description spans, including adjacent span boundaries.
   A match shows a documented rule finding, not independent proof of correctness.
4. High model confidence is a screen only. Require no outstanding model review
   flag and no multi-role flag. Do not auto-resolve these flags or change them.
5. Core industry reporting scope; no opportunity attributes, restricted opportunity
   title or academic-research title. A university employer alone does not establish
   an academic occupation: a data engineering staff role may be industry-equivalent.
6. All gates must pass. Negation, conditionals, qualifications-only evidence,
   incidental collaboration, absent employee attribution or category mismatch
   result in abstention. No human approval is required per clear passing job after
   this policy has been approved and activated as a versioned publication policy.

## Separate identifiable populations

Keep academic research, internships/theses/doctoral/postdoctoral/student openings,
adjacent management/statistics, governance, generic categories and unresolved jobs
in separate lanes. Preview their duty support and all flags, but exclude them from
this core automatic lane. Selecting a separate scope/opportunity for publication
requires an explicit policy/configuration choice, not reclassification. Preserve
valid uncertain decisions separately from technical failures and unknown outcomes.
Do not retry, reclassify, call a model, or treat failures as irrelevant.

## Category responsibilities and safeguards

Frozen category patterns live in src/preview_duties_v3.py. Examples: analytical
pipelines/platforms for data engineering; warehouses/ETL for warehousing; analysis
of data and delivery of insights for analysis; predictive/statistical models for
data science; model engineering/lifecycle for ML; AI/LLM/RAG applications or agents
for AI engineering; analytical models/dbt for analytics engineering; BI solutions,
models or dashboards for BI; data architecture for architecture. Tool experience
alone never meets the responsibilities gate. Rules for other scopes are assessed
only to describe their support, not to add them to core statistics.

## Provenance and activation boundary

Every preview decision records the original ID, category, scope, secondary labels,
confidence and flags; exact matched original description and duty offsets; source
hash; rule version; and every inclusion/abstention reason. Keep all 294 relevant
rows, all 500 operational outcomes, earlier 108 job records, extraction evidence,
checkpoints, caches and prior publications unchanged.

Preview is not approval. It has no activation flag and must not publish a dataset
or change dashboard discovery. If adopted, a dedicated v3 publication gate must
consume verified audit receipts with source/rules/policy identities and record
approval_method=automated_deterministic, human_reviewed=false, and the approved
policy version. Never populate v1 person-audit fields or human ground-truth labels.
Deterministic extraction, ID deduplication and dashboard refresh follow only after
explicit activation authorization; preserve the original 108 rows on ID collision.

## Limits

This is a deterministic audit of occupational evidence, not independently verified
human truth. Confidence and matching source quotes are not independent proof.
One matching duty may miss competing or dominant responsibilities, and section
parsing can fail. All accuracy/precision estimates remain unknown pending separate
human evaluation. Treat this 500-ad run as policy-development data, not a fresh
accuracy test; do not use the reserved 200 to tune these rules. Missing matches can
stay unused indefinitely; the user need not manually review every abstention.
The preview count is descriptive and must be shown before activation.
