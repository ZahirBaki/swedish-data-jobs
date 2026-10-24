# Step 6: approved v3 activation

The user approved automatic v3 inclusion for the final preview's qualifying jobs.
The active policy is docs/precision_first_inclusion_policy_v3.md; prior proposals
and policies remain unchanged. src/activate_duties_v3.py verifies the frozen rule,
proposal, selection, checkpoint and source identities, recomputes every audit and
allows only approved preview members that actually pass. No expected count is
forced. Publication ran with network requests explicitly blocked.

## Final dataset

- 37 newly accepted advertisements passed recomputation.
- 108 historical advertisements were retained without changing their original
  source, classification or extraction fields.
- 145 unique analyzed advertisement IDs in the combined publication.
- 92 jobs with usable requirements evidence: 70 historical plus 22 new.
- The other 15 additions remain analyzed jobs with no qualifying skill/language/
  education/experience evidence under the existing deterministic usability rules;
  their mention-only or ambiguous evidence is preserved, not promoted.

Usable means at least one qualified skill, language, education or professional
experience fact under the existing deterministic extractor. Seniority and metadata
alone do not count. Requirement percentages use 145 analyzed IDs before filtering,
and the selected distinct IDs after filtering. The data remains an unverified
selected subset, not representative market demand or independently measured accuracy.

## Separate saved groups within the 500-job expansion

| Group | Ads |
|---|---:|
| Included by v3 | 37 |
| Core abstention | 143 |
| Academic research | 11 |
| Adjacent occupations | 49 |
| Restricted opportunities | 48 |
| Governance | 6 |
| Saved semantic uncertainty | 34 |
| Saved irrelevant | 170 |
| Technical failures | 2 |
| Total | 500 |

Opportunity attributes remain arrays on each receipt, distinguishing internships,
theses, doctoral, postdoctoral and student jobs. Academic/opportunity lane precedence
and category-derived governance are documented in the approved proposal; saved
model scopes and classifications remain unchanged. Groups are not merged into
occupational irrelevance. Unknown request outcomes remain unretried.

## Provenance and files

Publication: data/processed/step6-v3-active-20261024/.

- policy_activation.json records explicit authorization, approved preview identity,
  proposal, source, policy, engine, publisher and extractor hashes.
- inclusion_audit_receipts.jsonl retains all 500 recomputed outcomes, source duty
  offsets, reasons, policy versions, original flags and receipt hashes.
- classification_decisions.csv and group_*.csv retain all separate populations.
- jobs_requirements.jsonl and jobs_requirements.csv contain the 145 unique jobs.
- requirements_evidence.csv retains all original evidence plus extracted additions.
- job_inclusion_provenance.csv identifies policy and method for every analyzed job.
- requirement_coverage.csv and step6_summary.json hold refreshed denominators,
  usability, requirement, category and scope analytics.
- preservation_manifest.json records hashes of every earlier data file, including
  original datasets, preview reports, checkpoints, caches and pending markers.
- manifest.json is written atomically last as the dashboard discovery commit.

New jobs record automated_deterministic under precision-first-v3, human reviewed
false, full matched duty evidence and immutable classifier provenance. The original
108 have additive copied-publication metadata historical_source_validated_subset /
historical-step5-accepted-subset, not retroactive v3 approvals or invented human
verification. Their original fields and evidence compare identically.

The dashboard uses the latest completed publication, shows active policy and
inclusion-method filters, exposes duty evidence per new job, and separates excluded
populations from analyzed requirement charts. The Step 5 dashboard regression now
explicitly pins its historical 108-job fixture; v3 has separate refresh tests.

Reproduce into a NEW publication directory only:

```powershell
.venv/Scripts/python -m src.activate_duties_v3 --output data/processed/step6-v3-active-next
.venv/Scripts/python -m streamlit run dashboard/app.py
```

These commands use saved classifications and deterministic extraction. No API
requests, retries or reclassification are part of activation. Existing output
folders are rejected to preserve previous runs.

Run all automated tests:

```powershell
.venv/Scripts/python -m unittest discover -s tests -q
```

Validation completed: **335 tests passed**, including 12 new activation/publication
tests. Default dashboard discovery and rendering show 145 analyzed jobs and 92
with usable requirements evidence. Method filtering isolates the 37 v3 additions
and their 22 usable extractions. All requirement coverage denominators were checked.
All **2,695** pre-existing data files matched their preservation hashes after the
full suite; the historical 108 field projection and evidence compare identically.
No paid requests, retries or reclassification occurred. Streamlit emitted its
existing temporary-directory cleanup permission warning after successful testing;
the suite exit code was 0. Final verification is saved in the publication's
verification.json.
