# Precision-first inclusion policy v1

Version `precision-first-v1`. This is a downstream inclusion/reporting policy, not
a rewrite of any historical model classification or the Step 3F 1.1 API prompt.
The three-case reliability test keeps that prompt fixed to test engineering changes.

Include an advertisement only when its actual substantial responsibilities clearly
establish a named Data/AI occupation AND its primary category is defensible. A title,
AI/SQL/tool mention, colleague's responsibilities, employer branding, ordinary software,
generic research or optional competence is insufficient. Do not infer duties.

## Outcomes and reason codes

- `included`: passes all configured gates; inclusion is not a calibrated guarantee.
- `excluded`: an accepted irrelevant prediction, or explicit evidence-audit finding
  `incidental_only` / `unrelated_duties`. Store the actual reason and evidence, not
  an inference from missing keywords. Unverified exclusions still need quality review.
- `abstained`: insufficient/contradictory duties, unresolved or unsupported category,
  uncertain prediction, low confidence, outstanding review, missing audit or evidence.
- `technical_failure`: incomplete/invalid model response, unknown network outcome,
  incompatible/corrupt evidence or no executed prediction. Never a semantic exclusion.
- `out_of_scope`: a reporting filter excludes an otherwise potentially relevant
  research/adjacent/opportunity record. This is not occupational irrelevance.
- `not_executed`: no prediction yet; an operational pending state, not a failure
  or a semantic judgment. Report its rate separately.

Reasons are recorded separately, e.g. `incidental_tools_only`, `unrelated_duties`,
`semantic_uncertainty`, `category_unresolved`, `category_not_supported`,
`duties_not_confirmed`, `confidence_below_threshold`, `review_outstanding`,
`description_evidence_required`, `scope_not_selected`, `opportunity_not_selected`,
`invalid_source_evidence` and the underlying technical status. Preserve every row.

## Separate reporting scopes

Core industry, dedicated data/AI governance, academic AI/ML research and adjacent
occupations stay separate. Default reporting includes only core industry and
governance, with named categories; exclude `Other relevant Data & AI roles` by
default. Governance remains separately broken out. Research, management/statistics,
borderline work, internships/theses/doctoral/postdoctoral/student opportunities
require explicitly selected scopes/opportunities. Primary management/statistics
uses adjacent scope. Hybrid secondaries never establish a primary category by proxy.

## Default conservative gate

Require a technically accepted complete response, exact valid source-span evidence
including description text, relevant judgment, named supported primary category,
high qualitative confidence and no outstanding model review flag. Also require a
documented independent inclusion audit of duties AND category: `duties_assessment`
and `category_assessment` both `clear`, with `duties_assessor` and explanatory
`inclusion_audit_notes`. These fields are blank until a person explicitly audits
the advertisement; no model label populates them. This deliberately abstains from
unaudited predictions, even high-confidence ones. It is a conservative publication
gate, not a claim that a model can prove its own semantic correctness.
An attributed inclusion auditor may explicitly set inclusion_review_resolved=true
after checking duties/category and documenting the result; this resolves the
downstream review gate without changing the original model review flag.

Configuration can relax the audit requirement, review flag, confidence level,
scope/opportunity filters or generic-category restriction for candidate analysis.
Record configuration and threshold results. Model confidence is uncalibrated and
must never be presented as probability. Compare high versus high+medium versus all
on independently verified examples; select thresholds on development/calibration
data, then freeze them before final evaluation. Do not optimize on the held-out 200
and subsequently claim it as untouched final-test performance.

## Evidence and API safety remain mandatory

Keep Step 3F 1.1 per-ad atomic-ID enums, full source text, exact offsets/excerpts,
scope/category validation and finish_reason=stop requirement. Never repair a
combined ID, accept partial labels, invent quotations or turn an API failure into
irrelevance. Preserve unknown SSL outcomes, global pending markers and budget holds
across prompt versions and alternate IDs. No automatic retries or full-corpus calls.

## Independent evaluation

Use only a separately supplied human sheet with matching original IDs/text,
human_verified=true, named reviewer, resolved relevance and canonical categories.
Reject AI-assisted-review provenance and assistant/model labels as ground truth.
Verify membership and hashes against the reserved independent 200-input frame.
Freeze predictions, inclusion audits and configuration before joining human truth.
An inclusion auditor must differ from the independent ground-truth reviewer.

Relevance precision = human-relevant included / included with resolved verified
human relevance. Report the denominator, unverified included and unresolved truth.
Category precision among included = exact category matches / included with a
resolved verified category or verified human-irrelevant judgment (the latter is
an incorrect category). Relevant truth with unresolved category is not evaluable.
Report per-category and scope precision, included coverage, semantic abstention,
technical failure and out-of-scope rates separately. Precision is null with no
evaluable included predictions; do not invent zero or perfect performance.

False-positive rows retain full source, evidence and predictions for human cause
analysis. Distinguish observed wrong relevance/category from a hypothesized cause;
human_false_positive_cause is optional and never automatically adjudicated.
All metrics apply only to the reviewed frame. This reserve is stratified over
distinct source inputs excluding policy-development ads, not an unweighted sample
of all 5,023 ads or complete Swedish job-market demand.
