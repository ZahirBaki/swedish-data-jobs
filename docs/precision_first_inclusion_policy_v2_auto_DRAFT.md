# Precision-first v2 automated duties audit — DRAFT FOR USER REVIEW

Proposed version: `precision-first-v2-auto-draft`. **Not approved, not active.**
Active policy remains `precision-first-v1`. Creating this file or candidate reports
never authorizes publication. No automatic or human audit fields are populated.

## Requested change

v1 requires an independent person to record clear duties and category assessments.
Replace that requirement only for the narrow automated lane below with a separately
identified deterministic duties assessment. Human review remains required for all
other relevant candidates. This is a change to the publication gate, not permission
to change classifications, model prompts, old datasets or independent truth labels.

## Proposed automated lane

Require all of the following, conjunctively:

1. An existing complete Step 3F 1.1 classification is source-validated, including
   original description span IDs, exact offsets/excerpts and category/scope invariants.
2. Relevant judgment, high confidence, core_industry scope, no review flag,
   no multi-role flag and no secondary categories. Confidence is a screening gate
   only: neither confidence nor a valid quotation proves the judgment is correct.
3. No restricted opportunity attributes or restricted opportunity/research title.
   Research, academic, governance, management, statistics, generic categories,
   hybrids and adjacent roles stay outside this automatic lane.
4. The primary category has an explicit supported rule in the frozen category map
   in `src/automatic_duties_audit.py`. There is no fallback category inference.
5. At least two distinct responsibility clauses, or one explicit compound duty
   containing two concrete operations on the same category-specific object, pass
   the rule engine: explicit employee address or imperative bullet under an explicit
   duties heading; concrete action and category-specific work object in that clause.
   Exclude qualification-only text, optional/conditional/negative statements and
   employer/team/colleague claims. Title or technology mentions cannot qualify.
6. At least one independently found clause overlaps a validated model-selected
   description span. This checks consistency only, not independent correctness.
7. Store the full candidate findings, exact original text offsets, source hash,
   rule/category identifier, all classifier metadata, engine and policy hashes,
   approval method and reasons. Keep original predictions immutable.

The executable implementation currently evaluates these criteria **as candidates
only**. A candidate is not automatically eligible under active v1. The implementation
has no flag that enables approvals or bypasses the policy requirement.

## Provenance if this proposal is approved

A subsequent implementation must use a dedicated versioned gate that consumes the
frozen deterministic audit receipt; it must not forge v1 person-audit CSV fields.
Record `approval_method=automated_deterministic`, `human_reviewed=false`, and
`classification_human_verified=false`, plus actual approved policy version, rules
hash, source hash, matching offsets and receipt identity. Human approvals remain
separate with attributed reviewer identity. Never set `inclusion_review_resolved`
automatically or resolve a model's mandatory review flag. Historical 108 records
retain their original provenance and acceptance policy.

## Limitations and evaluation

These intentionally narrow rules may miss many suitable jobs. Failed matches mean
needs review, not occupational irrelevance. Template text, misleading context and
mixed responsibilities can still cause false candidates. Multiple operations do not prove
that the duties dominate the job. No precision, probability or accuracy claim is
supported until assessed on independently reviewed data. Existing source quotations
and model confidence are not ground truth. Do not tune on the reserved 200 then
claim an untouched evaluation. Rules and this proposal must be reviewed before any
new approval lane is enabled. Dataset statistics must identify the approval lane
and continue to describe only accepted analyzed advertisements.

## Release boundary

User approval of this concrete versioned proposal is required before implementing
and applying the automatic publication lane. Until then, new automated approvals
are zero and candidates are shown separately from analyzed jobs. No paid calls,
retries, reclassification, or changes to the original 108 are part of this proposal.
