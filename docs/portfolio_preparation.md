# Portfolio preparation and verification

## Prepared changes

- Rewrote the README around the current pipeline, dashboard quick start,
  technologies, evidence semantics, reproducibility and limitations.
- Published only aggregate results: 5,023 collected candidates, 1,778 accepted
  AI-relevant advertisements and 1,111 core-industry advertisements. All figures
  are explicitly provisional findings from the collected subset.
- Added six handwritten fictional dashboard examples and an offline exploration
  notebook. No original advertisement, employer/contact record, ID or model
  response is copied into the public fixture.
- Added demo fallback and actionable missing/invalid-data messages. Existing local
  completed research publications still take priority, with the original filters,
  denominators, evidence inspection and inclusion policies preserved.
- Removed repeated publication-discovery logic, sorted the touched dashboard
  imports and documented the current versus historical module responsibilities.
  Frozen source/policy/prompt identities were deliberately not reformatted.
- Moved early API/heuristic notebooks to `notebooks/archive/`; cleared execution
  state and outputs from public notebooks. Original notebooks and the previous
  README remain in the ignored local archive.
- Strengthened ignores for all local `data/`, credentials, caches, logs, temporary
  artifacts and local archives. Kept empty `.env.example` credential placeholders.
- Pinned direct dependencies to the working environment; added public-clone CI,
  a reusable heuristic secret/history checker and 13 portfolio regression tests.
- Replaced private-data dependencies in nine older Step 6 unit tests with the
  committed fictional baseline. Real-data preservation tests remain separate.

## Verification

| Check | Result |
|---|---|
| Full local suite | 406 tests passed, no failures |
| Isolated public-file copy without private datasets | 369 passed; 37 optional local-data/Git checks skipped |
| New portfolio tests | 13 passed |
| Module imports | All 45 source modules imported offline |
| Dashboard rendering | Demo and existing research publication tested through Streamlit AppTest |
| Real server launch | Temporary localhost Streamlit server returned HTTP 200 and was stopped |
| Installed dependencies | Pinned requirements resolved offline; `pip check` passed |
| Git whitespace check | `git diff --check` passed |
| Secret scan | No matches in publishable files or 129 reachable historical blobs |
| Current notebook outputs and hardcoded home paths | None in publishable files |
| Original data | All 6,915 pre-existing data files verified byte-for-byte unchanged |
| Classification/research result changes | None |
| Paid API calls | None |

Logs and detailed hashes are local under
`data/analysis/portfolio-preparation-20261024/` and intentionally ignored.
This is a copy-of-publishable-files check, not a new remote clone. It uses the
working installed dependencies. A pristine online install was not performed;
the added CI performs installation from the pinned requirements when run.

## Remaining issues and limits

Two historical notebooks retain saved outputs and local paths in Git history.
Current notebook files are clean, and no credential patterns or exact local
credential matches were found. History has not been rewritten or pushed.
The scanner is heuristic and only covers locally reachable refs, not unseen
remote-only or unreachable objects. See [the repository guide](repository_guide.md).

The sandbox prevents writing `.git/index.lock`, including for `git add --dry-run`.
Read-only Git inventories were used instead; no files were staged or committed.
Streamlit tests emit an existing Windows temporary-directory cleanup warning at
interpreter exit; their assertions and Python process exit status pass.

There was no deletion of uncertain-use source modules: reviewed modules remain
test dependencies, runnable commands or hash-bound historical workflow components.
Further package restructuring should be a separate, explicitly versioned change.

## Git preparation

Include `.gitignore`, `README.md`, `requirements.txt`, `.env.example`, public
`.streamlit/config.toml`, `.github/workflows/`, `config/`, `src/`, `dashboard/`,
`prompts/`, `tests/`, `docs/`, `tools/`, `examples/` and output-free `notebooks/`.
Stage the notebook move as well as its removal at the old location.

Keep `data/`, `.env`, `.streamlit/secrets.toml`, caches, checkpoints, logs,
`.venv/`, `.test-tmp/` and `.local-archive/` ignored. Do not use force-add for them.
No commit, push, history rewrite or branch deletion was performed.
