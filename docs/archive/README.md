# Historical project stages

Historical commands and policy documents are retained for reproducibility.
They are not the default portfolio quick start.

| Stage | Preserved artifacts | Status |
|---|---|---|
| API/heuristic exploration | `notebooks/archive/`, `src/classify_jobs*.py` | Superseded for main dashboard relevance |
| AI pilots | `src/run_ai_classification.py`, `src/run_large_pilot.py`, Step 3E/3F preparation, recovery and analysis modules | Kept for saved provenance and tests |
| Strict inclusion v1/v2 | `src/precision_inclusion.py`, `src/automatic_duties_audit.py`, corresponding policy documents | Historical gates/proposals |
| Approved duties v3 | `src/preview_duties_v3.py`, `src/activate_duties_v3.py`, v3 policy and corpus-scope documents | Audit metadata and optional dashboard quality filter |
| Proposed v3.1 | `src/duties_audit_v31.py` | Inactive preview; no automatic activation |
| Expanded AI relevance | `src/run_step7.py`, [Step 7 policy](../step7_ai_relevant_policy_v1.md) | Current dashboard publication policy |

Referenced Python modules, prompts and policies stay at their original paths to
preserve imports, tests and prepared-run hash identities. Detailed per-ad
investigations and checkpoint archives stay in ignored local `data/` directories.
