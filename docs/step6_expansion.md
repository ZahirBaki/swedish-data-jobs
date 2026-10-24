# Step 6 — classification expansion and dashboard refresh

Implemented 24 October 2026. No paid calls were executed. The dashboard now selects the most recently completed Step 6 publication on each rerun, falling back to Step 5. JOBS_DASHBOARD_DATA still overrides discovery. Jobs and evidence modification timestamps invalidate the Streamlit data cache.

Files added: src/step6.py, tests/test_step6.py and this guide. Files updated: src/dashboard_data.py, dashboard/app.py and the README. Preparation and publications are new folders; earlier preparations and all historical data, caches and checkpoints are retained.

Canonical artifacts:

- data/processed/step6-prepared500-final-20261024/: selected_jobs.csv, request_plan.csv, dry_run_report.json, manifest.json.
- data/processed/step6-dry-run500-final-20261024/: identity.json, preflight.json, run.lock; no paid checkpoint.
- data/processed/step6-combined108-final-20261024/: combined jobs CSV/JSONL, requirements_evidence.csv, classification_decisions.csv, historical_classification_registry.csv, classification_provenance.json, inclusion audit queue/template, step6_summary.json and publication manifest.

The published dataset currently contains the original 108 jobs. New acceptance counts cannot be known until an explicitly started live run and duties audit. No classification of the full corpus is required.

## Selection and cost

Uniform sampling without replacement using pandas random_state=20261024, from advertisement-ID-sorted untouched sources, deduplicated by exact original title/body hash. Independent evaluation reserve excluded. This is a development sample of collected ads, not a representative Swedish market sample. Selection does not rely on the old keyword labels.

| Population disposition | Ads |
|---|---:|
| Eligible before source deduplication | 4,448 |
| Previously attempted/processed | 243 |
| Same source text as previously processed | 127 |
| Independent evaluation reserve | 200 |
| Missing description | 4 |
| Unknown request outcome | 1 |
| Total | 5,023 |

4,096 eligible unique sources remain; 500 selected, all unprocessed. Keyword labels in selected_jobs.csv are historical labels, not accepted AI classifications: 352 unspecified, Data Engineer 49, research 22, AI Engineer 16, Data Scientist 15, BI 10, Data Analyst 9, ML Engineer 8, Data Architect 7, other relevant 4, ETL 3, management 2, governance 1, Statistician 1, Analytics Engineer 1.

Expected classification cost: $1.4194288. Conservative reservation total: $4.921336. Application budget default: $10, configurable with --budget-usd on each invocation. Rates are inherited project planning inputs ($0.40/M input, $1.60/M output), not a verified current provider quote. Forecasts are heuristics, not billed usage. The existing engine reserves conservative per-request costs before sending, retains unknown-outcome holds, stops when a request cannot fit, and records returned usage. A provider reporting usage beyond the reservation stops subsequent calls. Deterministic extraction costs $0.

## Exact commands

Run from the project root in PowerShell. These commands use the already prepared final selection.

Unpaid preflight (the folder already exists, hence --resume):

```powershell
.venv/Scripts/python -m src.step6 run --prepared data/processed/step6-prepared500-final-20261024 --output data/processed/step6-dry-run500-final-20261024 --budget-usd 10 --dry-run --resume
```

Live execution, only when you deliberately choose to incur API costs. OPENAI_API_KEY must already be set in the process environment; this command does not load .env:

```powershell
.venv/Scripts/python -m src.step6 run --prepared data/processed/step6-prepared500-final-20261024 --output data/processed/step6-live500-20261024 --budget-usd 10 --execute
```

Safe live resume (same output and preparation):

```powershell
.venv/Scripts/python -m src.step6 run --prepared data/processed/step6-prepared500-final-20261024 --output data/processed/step6-live500-20261024 --budget-usd 10 --execute --resume
```

No automatic retries: uncertain predictions, invalid responses, HTTP errors and unknown outcomes remain terminal on resume. A reservation persisted before a crash is treated as unknown unless an exact completed cache receipt safely settles it. Historic attempts are excluded even if their old classifier version was incompatible. This intentionally prefers less coverage over repeat billing. Exact cache/pending markers and OS file locks guard matching requests across runs. Source/code/prompt/policy hashes prevent resuming a modified preparation.

Create an unpaid post-live publication and audit queue:

```powershell
.venv/Scripts/python -m src.step6 publish --prepared data/processed/step6-prepared500-final-20261024 --run-dir data/processed/step6-live500-20261024 --output data/processed/step6-postlive-review-20261024
```

Copy inclusion_audit_template.csv to a new file such as data/reviews/step6-inclusion-audit.csv. Record duties_assessment=clear and category_assessment=clear only when supported by the full original duties, with a named duties_assessor and inclusion_audit_notes. Resolve flagged reviews explicitly via inclusion_review_resolved=true only after review. Never auto-fill those audit assertions. Audit columns cannot replace model or source fields. Low confidence, uncertain, irrelevant, invalid or unsupported evidence stays excluded by precision-first-v1. Research, governance and adjacent scopes remain separately reported; opportunity and generic-category exclusions retain the existing policy defaults.

Publish accepted additions through the same deterministic Step 4 extractor and Step 5 analytics:

```powershell
.venv/Scripts/python -m src.step6 publish --prepared data/processed/step6-prepared500-final-20261024 --run-dir data/processed/step6-live500-20261024 --audit data/reviews/step6-inclusion-audit.csv --output data/processed/step6-combined-expanded-20261024
.venv/Scripts/python -m streamlit run dashboard/app.py
```

Each publication uses a new output folder. Merge preserves original 108 rows and facts on ID collisions, adds each new ID once, and retains predictions, source evidence and inclusion audit provenance. Manifest is committed last so incomplete outputs do not become the dashboard default. Requirement denominators are unique accepted analyzed ads after active filtering; missing evidence is not absence of a requirement. Global classification metrics are unfiltered. Accepted means source-validated under this workflow, not independently verified ground truth.

To prepare a different sample, always choose a new destination:

```powershell
.venv/Scripts/python -m src.step6 prepare --output data/processed/step6-prepared500-next --size 500 --seed 20261024 --budget-usd 10
```

Validation:

```powershell
.venv/Scripts/python -m unittest tests.test_step6 tests.test_step5 tests.test_step4 tests.test_step3f_reliability -q
```

69 tests passed, including 9 new expansion tests, original data preservation, requirement denominators, dashboard rendering and filters, duplicate requests, mocked timeout/invalid outcomes, budget cutoff, safe resume, audited extraction and publication discovery. The sandbox emitted a Streamlit temporary-directory cleanup permission warning after the successful test run; test exit code was 0.
