# $3.50 live-budget verification

No paid API requests were made. Production code and prepared manifests were not
changed. Added tests/test_budget_350.py; 25 tests passed when run with the corpus
regressions (8 new targeted budget tests, 17 existing corpus tests).

The current corpus-run-20261024 preflight specifies budget 3.5, execute=false,
accounted cost 0. There is no live checkpoint in that run directory yet.

## Verified enforcement

- corpus_workflow.run passes the invocation budget to the classifier and reloads
  the same output/checkpoint ledger. Saved accounting over the budget is rejected.
- llm_classifier.reserve sums all accounted_cost_usd ledger entries using Decimal,
  including unknown-outcome and pre-request reservations. It checks spent + next
  conservative reservation <= budget before allowing the request.
- step3f_reliable_classifier calls reserve and durably saves the reservation and
  pending marker before session.post. The check happens per request, not merely
  once per batch or once per run. BudgetExceeded stops the corpus invocation.
- A timeout leaves the entire conservative reservation charged as an unknown hold.
  Resume does not clear it or reset the spend counter. Global pending markers and
  history guards prevent retry under an alternate ID.
- Config max_retries=0, retry_unknown=false, and Step3F rejects enabling them.
  Invalid responses and HTTP failures also are not automatically retried.
- Mock tests exercise exact $3.50 equality, a request that cannot fit, durable
  pre-request reservation, cumulative resume spend, unknown holds, failure retries,
  corpus-level stopping, the default-budget hazard and unexpected provider usage.

## Important limits

The $3.50 limit is an invocation parameter, not a permanently saved ceiling.
The CLI defaults to $60. A previous dry run at $3.50 does not change that default.
Always supply --budget-usd 3.50, including on resume, and keep the same --output.
A different output folder creates a different budget ledger. Historic runs' charges
and unknown holds are not deducted from this new corpus run's budget, though those
historical inputs stay blocked from paid retries.

The guard strictly controls admission against its configured reservations; it is
not an absolute provider invoice guarantee. Provider-reported tokens are priced at
configured rates. If a provider reports usage above the conservative reservation,
the already-sent request can put accounting over budget; the receipt is preserved
and subsequent requests stop. A dedicated mock test explicitly verifies this
failure mode. No claim of an external/provider spending cap is made.

## Approximate capacity

Using the prepared request order and checking the next reservation before every
simulated request, with a fresh $3.50 ledger:

| Simulation | Paid request units | Additional advertisement IDs incl. aliases | Accounted cost |
|---|---:|---:|---:|
| Expected planned usage | 1,251 | 1,419 | $3.4922936 |
| Each request consumes its full conservative reservation | 364 | 417 | $3.4957580 |

These estimate processed source inputs, not relevant jobs accepted by v3. Actual
lengths, provider usage and failures can change throughput. Previously cached IDs
remain reusable at zero new API cost and are not part of these additional counts.

## Correct live/resume command (not executed)

```powershell
.venv/Scripts/python -m src.corpus_workflow run --prepared data/processed/corpus-prepared-20261024 --output data/processed/corpus-run-20261024 --budget-usd 3.50 --batch-size 100 --max-batches 50 --execute --resume
```

Large batch limits do not bypass the per-request budget guard. Repeat this exact
command to resume without resetting the $3.50 application ledger. The older command
with --budget-usd 60, or a command omitting the budget option, does not enforce the
user's new $3.50 limit. The exact shell command the user intends to execute was not
provided; the saved unpaid preflight alone cannot certify that future command.

Validation:

```powershell
.venv/Scripts/python -m unittest tests.test_budget_350 tests.test_corpus_workflow -q
```

Exit code 0. The existing Streamlit temporary-directory cleanup permission warning
appeared after the successful tests; it did not change the results.
