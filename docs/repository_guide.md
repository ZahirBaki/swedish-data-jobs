# Repository and reproducibility guide

## Current entry points

| Task | Module or file | Inputs |
|---|---|---|
| Dashboard | `dashboard/app.py` | Completed local publication or fictional demo |
| Analytics and filters | `src/dashboard_data.py` | Unique-ID jobs and requirement facts |
| Collection | `python -m src.collect_jobs --help` | Public search terms and API contracts |
| Corpus dry run and safe execution | `python -m src.corpus_workflow --help` | Original local collection and saved classifications |
| Latest classifier | `src/step3f_reliable_classifier.py` | Full original source, versioned prompt and schema |
| Requirement rules | `src/extract_requirements.py` | Original job fields; no network |
| Expanded offline publication | `python -m src.run_step7 --help` | Saved outputs and previous local publication |
| Demo generation | `python -m tools.build_demo` | Handwritten fictional examples only |
| Repository checks | `python -m tools.check_repository --scan-history` | Local candidate files and reachable Git blobs |

## Why legacy modules remain

The project was developed in stages. An import-graph review found 44 source
modules; the modules without incoming imports are runnable investigation or
extraction entry points, not confirmed dead code. Earlier classifiers,
preparation/recovery commands and audit code are imported by current workflows
or regression tests. Several prepared runs hash source files, policies and
prompts to detect incompatible resumes. Formatting or relocating those files
would invalidate saved run identities.

Accordingly, no tested workflow was deleted or rewritten for cosmetic cleanup.
Core changes are limited to dashboard dataset resolution/error messages,
maintenance tooling, public examples and documentation. Early notebooks moved
to `notebooks/archive/`; their root discovery supports the new location. Outputs
were cleared from all public notebooks. Original notebooks and the previous
README are backed up under an ignored `.local-archive/` directory.

See [the historical-stage index](archive/README.md). Draft policy files stay at
their original paths because frozen code and saved identities reference them.
The inactive v3.1 preview is preserved, not activated.

## What belongs in Git

Commit source, tests, dashboard code, public configuration/search terms,
versioned prompts and methodology, output-free notebooks, synthetic examples,
dependency pins, CI and aggregate research findings. `.streamlit/config.toml`
contains only public theme/local-server settings. `.env.example` contains empty
credential placeholders and is explicitly allowed by `.gitignore`.

Keep all of `data/` local: raw ads, source metadata, request receipts, unknown
outcomes, classification checkpoints/caches, extraction evidence, run manifests
and detailed per-ad reports. Keep `.env`, credential files, Streamlit secrets,
virtual environments, local archives, editor settings, temporary files and logs
ignored. Important original research reports remain intact locally; the public
aggregate summary is [results.md](results.md).

Before staging:

```bash
python -m tools.check_repository --scan-history
python -m unittest discover -s tests -q
python -m pip check
git status --short
git add --dry-run .
```

The checks print only finding types and locations, never credentials. They scan
tracked files, unignored additions, and optionally all blobs reachable from local
Git refs, using token patterns plus exact local credential matches. This is a
heuristic, not proof of absence. No remote refs are fetched, history is not
rewritten, and nothing is staged, committed or pushed by these commands.

The current execution environment makes `.git` read-only: even
`git add --dry-run .` cannot acquire an index lock here. Use the read-only
`git status --short` and `git ls-files --cached --others --exclude-standard`
inventories for review, then stage manually in your own shell when ready.

## Known historical publication issue

The local history scan found no credential matches, but two historical notebook
blobs contain old cell outputs (including advertisement data) and local user
paths: `notebooks/01_explore_api.ipynb` and `notebooks/02_explore_jobs.ipynb`.
Their current public versions are output-free in the notebook archive; clearing
the working tree does not remove earlier Git blobs. No history was rewritten.
Review this historical content before publishing the existing Git history.
No secret values or advertisement contents are copied into this guide.

## Public-clone versus local-data tests

Unit tests and the demo use synthetic inputs and mocked classification responses.
The complete test suite also contains dataset-preservation and UI regressions
against historical local runs. Those tests are skipped when the artifacts are
absent; no real advertisements or keys should be added just to satisfy them.
CI runs the public suite, notebook/repository checks and dependency consistency.

Direct package versions match the working environment. `pip check` and a local
`pip install --dry-run --no-index -r requirements.txt` verify installed dependency
resolution without network. A pristine online install is a separate check and
depends on package-index availability; no installation is silently claimed from
an already populated environment.

## Deferred restructuring

For future major-version work, group modules into `collection/`,
`classification/`, `extraction/` and `publication/` packages, with compatibility
entry points. That migration should intentionally version prepared-run code
identities and be tested separately. Renaming these modules now would add risk
without improving the recruiter-facing demo.
