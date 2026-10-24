# Step 6 automatic duties audit: policy review report

Active policy: precision-first-v1. Proposed policy: precision-first-v2-auto-draft
(not approved). Rule engine: duties-candidate-rules-1.1. No paid calls, retries or
reclassification. Network access was explicitly blocked during the final audit.

## Policy finding

The active policy says: "These fields are blank until a person explicitly audits
the advertisement; no model label populates them." It also requires the independent
inclusion audit of duties AND category. The setting permitting relaxed audits is
explicitly limited to candidate analysis. Automation cannot silently supply a
person's audit or turn a screening result into v1 publication eligibility.

The implementation therefore analyzes candidates and creates a proposed change;
it has no execution flag that approves them. No person-audit fields or model flags
were synthesized, changed or resolved. The proposal requires user review before a
separate versioned publication gate can be implemented and used.

## Outcomes

Of the 246 previously pending relevant advertisements:

| Candidate assessment | Ads |
| --- | ---: |
| Narrow automatic-lane candidates, pending policy approval | 3 |
| Require human review under the draft lane | 240 |
| Outside draft lane due to opportunity/research title safeguard | 3 |
| Total pending assessed | 246 |

The title safeguard is an additional candidate restriction, not a reclassification
as occupationally irrelevant. Existing academic/special-scope decisions remain
separate, along with multi-role, hybrids, low confidence, unsupported categories
and mandatory model review flags. The other 48 relevant ads retain their saved
opportunity exclusion. Across all 500 saved outcomes the candidate report contains
3 proposed candidates, 274 review cases (including 34 uncertain predictions),
221 exclusions from the candidate lane (170 irrelevant, 48 opportunities, 3 title
safeguards), and 2 technical failures. Rules never retry technical failures.

Automatically eligible under current v1: **0**. Automatically approved: **0**.
New jobs extracted or merged: **0**. Final analyzed dataset: **108**.

Rules inspect original description clauses independently of model reasoning and
retain exact offsets for employee/action/category-object matches. Confidence and
source-span validation are necessary screening gates, not independent proof.
No accuracy or precision estimates are claimed. Missing rules or missing matches
mean review, not evidence of irrelevance. These intentionally narrow candidates
are not a promise of broader coverage after policy approval.

## Files

- src/automatic_duties_audit.py: reproducible offline candidate assessment, immutable
  classification handling, network-free publication and prior-data hash checks.
- tests/test_automatic_duties_audit.py: 19 tests for the policy boundary, exact
  offsets, compound duties, qualifications, title/tool-only evidence, negation,
  unsupported categories/version, scopes, opportunities, duplicates and review flags.
- docs/precision_first_inclusion_policy_v2_auto_DRAFT.md: concrete proposed lane,
  safeguards, provenance and approval boundary.
- dashboard/app.py: displays candidates separately with the active-policy boundary.
- data/processed/step6-automatic-duties-policy-review-20261024/: final publication,
  automatic_duties_summary.json, automatic_duties_decisions.csv/.jsonl (all 500),
  automatic_candidates.csv, human_review.csv, policy_excluded.csv, frozen proposed
  policy, original predictions/provenance, extraction evidence and preservation
  verification. Earlier intermediate audit runs remain preserved as well.

The dashboard selects this completed publication using existing dataset discovery.
The candidate notice explicitly excludes the 3 pending candidates from all analyzed
counts and requirement denominators. The original 108 job records and extraction
facts remain the baseline, and skill/education coverage uses their 108 IDs (or the
active filtered subset).

Reproduce offline into a NEW directory:

```powershell
.venv/Scripts/python -m src.automatic_duties_audit --output data/processed/step6-automatic-duties-next
```

This command cannot approve candidates, enable a policy change, or send an API
request. The final audit was additionally run with requests.Session.request
patched to fail if any network call were attempted.

Validation command:

```powershell
.venv/Scripts/python -m unittest tests.test_automatic_duties_audit tests.test_step6 tests.test_step5 tests.test_step4 tests.test_step3f_reliability -q
```

Final verification: 88 related tests passed, including 19 new audit tests. The
Streamlit app loaded the final publication, showed 3 pending candidates and zero
automatic approvals, retained 108 analyzed jobs, and passed scope filtering.
Baseline job rows and extraction facts compare identically; coverage denominators
remain 108 before filtering. Streamlit emitted its existing temporary-directory
cleanup permission warning after the successful tests (exit code 0).

The proposed candidates are advertisement IDs 30941105 (Data Engineer), 31127669
(ML Engineer), and 31609306 (Data Engineer). They are not accepted analyzed jobs.
Per-advertisement excerpts, offsets, source hashes and all decision reasons are in
the final automatic_candidates.csv and automatic_duties_decisions.jsonl artifacts.
