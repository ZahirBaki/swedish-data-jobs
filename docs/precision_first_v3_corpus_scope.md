# Precision-first v3: authorization for the collected corpus

Scope extension dated 24 October 2026. The user requested processing the remaining
original collected advertisements using the approved precision-first v3 policy.
This authorizes the same frozen v3 duties rules for new or reusable compatible
classifications throughout the 5,023-ID collected corpus, not only the original
37-job preview allowlist. The approved occupational/duties thresholds, evidence
validation, confidence/review/multi-role gates and separate scopes are unchanged.
The original activation and policy documents remain preserved.

Automatic publication method remains automated_deterministic under precision-first-v3,
with human_reviewed=false. Preserve the existing 145 analyzed records and all their
policy provenance; do not retroactively reject or reclassify historical records.
Relevant legacy predictions that cannot pass the Step 3F 1.1 compatibility gate
stay identified as legacy rather than paying for reclassification. Completed
irrelevant/uncertain classifications and technical failures are not retried.
Unknown outcomes remain blocked by source content across IDs and prompt versions.

Paid classification is NOT authorized for automatic execution: explicit --execute
is required. Resumable batches use a cumulative application budget and the existing
engine's per-request reservations, pending markers and atomic receipts. An unpaid
prepare/run never calls an API. Deterministic publication can proceed without paid
calls whenever compatible saved results become available.

Coverage reports retain every collected ID, operational status, semantic result,
inclusion decision, source hash, request action and provenance. Identical original
title/body inputs may share a completed classification without duplicate billing;
IDs stay distinct in reporting. Uncertain, ambiguous, opportunity, academic,
adjacent, governance and failed cases may remain aside indefinitely. Missing
classification is never converted into irrelevance. Independent reserve labels
remain untouched; classification predictions do not constitute human truth or
justify market-wide representativeness/accuracy claims.
