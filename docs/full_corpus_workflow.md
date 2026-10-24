# Full collected-corpus workflow: prepared, unpaid

The new workflow covers every original collected advertisement ID and is ready for
explicit paid execution. No paid API requests were made during implementation.
Compatible saved source receipts were reused immediately: **5 additional jobs**
passed unchanged v3 rules, so the dashboard now analyzes **150 jobs**, **97 with
usable requirements evidence**. The earlier 145 rows and extraction facts remain
identical. The full paid classification workload has not been executed.

## Reconciliation and request units

| Item | Count |
|---|---:|
| Original distinct collected IDs | 5,023 |
| IDs with direct historical successful results | 705 |
| IDs never directly successfully classified | 4,318 |
| IDs with a successful result after exact-source reuse | 862 |
| IDs without a successful result after reuse | 4,161 |
| Compatible Step 3F 1.1 cache receipts (unique request keys) | 501 |
| IDs served by compatible receipts | 544 |
| Preserved legacy completed results, including identical-source reuse | 318 |
| New unique paid requests planned | 3,789 |
| IDs covered by planned requests | 4,105 |
| Extra IDs sharing those new request sources | 316 |
| IDs left aside due to previous failures/attempts | 50 |
| IDs blocked by unknown outcomes | 2 |
| Missing descriptions | 4 |

The 50 previous-attempt cases contain 37 original technical failures and 13
additional IDs sharing source text with an attempted case. The two unknowns are
separate. Neither known failed requests nor unknown outcomes are retried, across
IDs and classifier versions. Missing descriptions are not occupational irrelevance.

A request unit is the exact original title and description, hashed without text
normalization or truncation. Matching source bodies may reuse predictions, while
all advertisement IDs remain distinct in the analysis (not deduplicated vacancies).
The initial workload uses all remaining unique sources in advertisement-ID order,
not a sample. Historical checkpoints, archived copies and receipts are reconciled
by ID/content; archives are never treated as independent new requests. The
independent reserve's labels remain untouched; predictions are not human truth.

## Forecast and safeguards

Expected new cost: **$10.700046**. Conservative reservations: **$37.0991204**.
Configurable cumulative application budget: **$60**. Model remains the latest
working project configuration, gpt-4.1-mini-2025-04-14, Step 3F 1.1. Rates are the
project's inherited planning inputs, $0.40/M input and $1.60/M output, not a current
provider quote or invoice. Expected token counts are heuristic; reservations use
the existing conservative UTF-8 input bounds plus the output cap. No paid tokenizer
or estimation requests are made. Historical costs do not consume this new run's
budget; new usage and unknown-outcome reservation holds do.

Every request is checkpointed/reserved before sending. Valid exact receipts and
per-key locks prevent repeat billing; pending markers/unknown content guards block
ambiguous outcomes. One corpus execution lock serializes corpus runners, and the
history is refreshed each batch. No automatic retries are enabled. Authentication,
payment, permission, rate-limit responses, three consecutive new unknown outcomes,
or an insufficient remaining budget stop the invocation. Failures remain terminal
when resumed. No source truncation, partial labels or inferred negative judgments.
A crash after a completed cache receipt can reuse the saved classification without
another request; unknown usage may remain conservatively held until reconciled.

Batch size and invocation count can change on resume. Checkpoint identity and
source/prompt/policy/code hashes must stay compatible. If implementation inputs
change, prepare into a fresh versioned directory; historical request guards and the
shared cache continue to prevent paying again. A budget can be raised explicitly
on resume; lowering it below already accounted usage is rejected.

## Exact commands (PowerShell, project root)

The preparation and unpaid run folder already exist. OPENAI_API_KEY must be in the
process environment before live execution; the module does not automatically load
.env or print credentials.

Repeat the full unpaid preflight:

```powershell
.venv/Scripts/python -m src.corpus_workflow run --prepared data/processed/corpus-prepared-20261024 --output data/processed/corpus-run-20261024 --budget-usd 60 --batch-size 100 --max-batches 50 --dry-run --resume
```

Explicit live full run (NOT executed). The same command safely resumes after an
interruption, without retrying completed/failed/unknown jobs. It permits at most
5,000 request units in this invocation; only the 3,789 remaining units are planned:

```powershell
.venv/Scripts/python -m src.corpus_workflow run --prepared data/processed/corpus-prepared-20261024 --output data/processed/corpus-run-20261024 --budget-usd 60 --batch-size 100 --max-batches 50 --execute --resume
```

To limit live progress to one batch, use --max-batches 1 with the same other flags.
Budget protection is cumulative across all batches and resumptions of this run.

After the live run, produce the final analysis, deterministic extraction, all-ID
coverage table and refreshed dashboard publication (no paid calls):

```powershell
.venv/Scripts/python -m src.corpus_workflow publish --prepared data/processed/corpus-prepared-20261024 --run-dir data/processed/corpus-run-20261024 --output data/processed/corpus-final-20261024 --finalize
.venv/Scripts/python -m streamlit run dashboard/app.py
```

--finalize explicitly leaves any remaining unprocessed IDs aside and labels them;
it never invents a classification. A final analysis can therefore be completed
without every advertisement being classified. Omit --finalize for an interim
publication. Every publication requires a new output directory, so subsequent
refreshes should use a new dated/suffixed destination. Never overwrite a prior run.

To prepare a completely fresh workload snapshot after source/code changes:

```powershell
.venv/Scripts/python -m src.corpus_workflow prepare --output data/processed/corpus-prepared-next --budget-usd 60
```

## Policy and publication behavior

The scope extension is explicit in docs/precision_first_v3_corpus_scope.md. The user
requested applying approved v3 to the full remaining corpus; the old 37-ID preview
allowlist is not reused as a corpus-wide restriction. Duties rules and thresholds
remain unchanged. There is no relaxation of source validation, confidence, mandatory
review, category support, multi-role, academic, adjacent, governance or opportunity
safeguards. Relevant legacy classifiers remain labeled legacy when not Step 3F 1.1
compatible; they are not paid to be reclassified. Existing accepted analyzed jobs
keep their historical inclusion policy and method unchanged.

All new accepted jobs use the deterministic extractor for skills, technologies,
Swedish/English, education, experience and seniority. Merge preserves every baseline
ID/field/evidence fact and adds each new ID once. Newly accepted jobs retain the
model, prompt/version, selected source spans, duties audit, source hash, automatic
method and policy version. Uncertain, irrelevant, abstained, research, opportunities,
adjacent, governance, legacy, failures and unprocessed states remain separate.

The dashboard chooses the newest completed publication only after its final atomic
manifest. Charts and coverage denominators use unique accepted analyzed IDs after
filters, never all 5,023 IDs. Global processing counts include exact-source reuse
and are distinct from paid request counts. Full coverage is downloadable in the
processing-decisions expander. Language, education, experience, seniority, scope,
category and requested skill summaries are regenerated from combined evidence.
The report is descriptive of this collected search-based corpus, not representative
of the entire Swedish market or independently human-verified classification accuracy.

## Artifacts

- src/corpus_workflow.py: preparation, opt-in batches/resume, v3 extraction/merge and
  full-corpus publication commands.
- tests/test_corpus_workflow.py: 17 mock/offline regressions, including duplicate
  requests, compatible caches, unknown guards, resume, budgets, v3 scope, all-ID
  coverage, preserved baseline and dashboard refresh.
- data/processed/corpus-prepared-20261024/: collected_jobs.csv (all 5,023),
  initial_coverage.csv, request_plan.csv, reusable_predictions.json, dry_run_report.json
  and an immutable manifest.
- data/processed/corpus-run-20261024/: unpaid preflight and identity. Live execution
  later adds a cumulative checkpoint, per-batch receipts and accounting summary.
- data/processed/corpus-offline-coverage-20261024/: current 150-job publication,
  corpus_coverage.csv (all 5,023 IDs), group_*.csv, regenerated evidence and coverage,
  inclusion receipts, classification provenance, policy authorization, report.md,
  baseline preservation checks and final publication manifest.

The active corpus run is intentionally mutable for safe resume; each publication
freezes run_snapshot copies. Historical datasets, earlier runs, cache receipts,
preview reports and publications are preserved and hashed. A publication excludes
its own resumable run directory from immutable-data hash assertions, while recording
that exclusion and freezing its then-current identity/checkpoint/accounting files.

Full test command:

```powershell
.venv/Scripts/python -m unittest discover -s tests -q
```

Final validation: **352 tests passed** (17 new corpus tests); suite exit code 0.
Default dashboard rendering shows 150 analyzed jobs, 97 with usable requirements
evidence, all 5,023 collected IDs and 862 IDs with successful saved/reused results.
All 2,724 immutable pre-existing data files in the publication preservation manifest
matched their hashes after the full test suite. No live checkpoint was created:
classification has not started and API spend for this work is $0. The existing
Streamlit temporary-directory cleanup permission warning appeared after successful
checks. Verification is saved in the current publication's verification.json.
