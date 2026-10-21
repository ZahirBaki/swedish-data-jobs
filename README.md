# Swedish Data & AI Job Market Analyzer

A beginner-friendly Data Engineering portfolio project exploring demand for data and AI professionals in Sweden. The long-term goal is to examine technical skills, Swedish-language requirements, education, experience levels, and hiring trends.

Planned job categories:
- Data Analysts
- Data Engineers
- Data Scientists
- Machine Learning Engineers
- AI Engineers

## Time window and scope

The analysis will focus on the last 6-12 months to reflect recent demand while allowing comparisons across months. A shorter sample can be affected by seasonality, and advertisements are only a partial view of hiring activity.

**Step 1 does not apply a date or country filter.** It requests one page of up to 100 keyword matches for `data engineer`. Results are not guaranteed to be from the last six months, located in Sweden, or representative of the whole market. A keyword match does not establish a role category. Before trend analysis, we will validate dates and geography and investigate historical coverage and the appropriate historical-advertisement source.

## Data source

Step 1 uses Arbetsförmedlingen's official Swedish JobTech JobSearch API:
- Endpoint: https://jobsearch.api.jobtechdev.se/search
- Parameters: `q="data engineer"`, `limit=100`, `offset=0`
- [Official getting-started documentation and response fields](https://gitlab.com/arbetsformedlingen/job-ads/jobsearch-apis/-/blob/main/docs/GettingStartedJobSearchEN.md)
- [Official advertisement field reference](https://gitlab.com/arbetsformedlingen/job-ads/jobsearch/jobsearch-api/-/blob/main/docs/AdFields.md)

The notebook uses `hits`, `total.value`, `headline`, `employer.name`, `publication_date`, and `workplace_address` fields. Optional values can be missing or null. Requests need internet access; HTTP, connection, and timeout problems produce readable errors. A failed request stops execution; fix the problem and rerun from the top.

## Project layout

```text
swedish-data-jobs/
  data/raw/                         # Step 1 downloads, preserved
  data/collections/                 # Step 2 run folders and logs
  notebooks/01_explore_api.ipynb     # Step 1, unchanged
  src/__init__.py
  src/collect_jobs.py
  config/search_terms.json
  docs/api_notes.md
  tests/test_collect_jobs.py
  requirements.txt
  .gitignore
  README.md
```

## Install and run (Windows PowerShell)

From the parent workspace folder:

```powershell
cd C:\Users\zahir\Desktop\projects\swedish-data-jobs
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name swedish-data-jobs --display-name "Python (swedish-data-jobs)"
.\.venv\Scripts\python.exe -m jupyter notebook notebooks/01_explore_api.ipynb
```

If `py` is unavailable, use `python` for the environment-creation command. These commands call the virtual environment directly, so PowerShell activation is unnecessary.

In the notebook, select **Python (swedish-data-jobs)** and run all cells from top to bottom. Alternatively, open the project folder in VS Code, install Microsoft's Python and Jupyter extensions if needed, open the notebook, select the `.venv` Python interpreter as its kernel, and choose **Run All**. Start from the project root or `notebooks` folder so the output path resolves correctly.

## What Step 1 produces

- A live API-reported total and the number downloaded (up to 100; fewer if the API returns fewer).
- A DataFrame, its first 10 rows, and every column's data type.
- A simplified title, employer, publication date, and location table.
- One full advertisement printed as nested JSON, when available.
- The original advertisement list saved as `data/raw/jobtech_data_engineer_<UTC timestamp>.json`, with its exact absolute path printed on success.

The JSON file stores the unmodified advertisements, not the full response envelope or the simplified table. UTF-8 preserves Swedish characters. Raw downloads are excluded from Git and are created when you successfully execute the notebook. No downloaded dataset is bundled with this starter project. Empty responses are handled and saved as an empty list.

## Step 2: Six-month candidate collection

The reusable module is `src/collect_jobs.py`. The editable list in `config/search_terms.json` covers the five core roles, Analytics Engineer, BI Developer, Data Warehouse Developer, ML/AI Developer, and Swedish alternatives. All keyword matches are uncertain candidates for Step 3, including ads whose descriptions mention a role while the actual position is unrelated. No NLP or classification is implemented.

### Run with the existing virtual environment

```powershell
cd C:\Users\zahir\Desktop\projects\swedish-data-jobs
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.collect_jobs --start 2026-04-21 --end 2026-10-21
```

`tzdata` supplies Swedish timezone rules on Windows. No activation or new environment is needed. The dates above are also the defaults; both calendar dates are included. Requests are sequential, at least one second apart, with a 90-second timeout and up to three retries after the first attempt. Retries use exponential backoff and honor `Retry-After`. Step 1's notebook and raw downloads are unchanged.

For twelve months, change the dates:

```powershell
.\.venv\Scripts\python.exe -m src.collect_jobs --start 2025-10-21 --end 2026-10-21
```

Edit the terms JSON or pass `--terms-file path/to/terms.json`. Multiword terms use the documented quoted phrase syntax by default; a phrase may still match the description, so these remain uncertain candidates. Use `--broad-query` for unquoted searches if desired: these may match individual words, substantially increasing unrelated candidates and collection time. Other options include `--delay 2`, `--timeout 120`, `--retries 4`, `--page-size 100`, and `--output-dir data/collections`. Use `--help` for details. Each run creates a fresh UTC timestamped directory. Rerunning does not overwrite earlier results; resumable checkpoints are not implemented.

### Sources, dates and deduplication

Both official `/search` endpoints are used:

- `https://jobsearch.api.jobtechdev.se/search`: active advertisements.
- `https://historical.api.jobtechdev.se/search`: archived/removed advertisements.

The collector reads and saves each API's official Swagger specification, using supported country/publication filters, sorting and pagination limits. Date bounds are padded to avoid API boundary ambiguity; the exact inclusive interval is enforced locally using only `publication_date`. Offset-aware timestamps are converted to Europe/Stockholm; naive timestamps are treated as Swedish local time. Deadlines and updates never substitute for publication dates.

Both queries filter for Sweden. Only returned ads with identifiable Swedish country information and valid in-period publication dates enter `jobs.csv`. Missing municipality or region is allowed. Unknown/foreign locations and missing/out-of-period dates remain in the originals and an exclusion/review report.

Numeric Platsbanken ids deduplicate active records and link Historical `original_id` to active `id` when available. Otherwise a source-specific archive id is retained. Missing ids receive an explicit content-hash fallback and a warning. Titles and employer-specific external ids never determine identity. All original source/version objects survive. The cleaned row prefers an eligible active version; otherwise it uses the first eligible historical version. Conflicting publication dates are flagged. See [API decisions](docs/api_notes.md) for official references and verified API behavior.

### Outputs

```text
data/collections/
  <run-id>.log
  <run-id>/
    configuration.json
    jobsearch_swagger.json
    historical_swagger.json
    pages/000001.json ...
    candidates.jsonl
    jobs.csv
    excluded_or_review.json
    summary.json
    summary.txt
```

- `pages/`: complete original API responses with source, term, parameters and retrieval time. Advertisement objects are unmodified. Schema snapshots exist only when successfully retrieved.
- `candidates.jsonl`: one deduplicated candidate per line with every original JSON source/version, all source/term pairs and CSV inclusion status. No candidate is discarded based on title relevance.
- `jobs.csv`: advertisement id, title, employer, publication date, municipality, region, full API-provided plain-text description, occupation/group/field classification, vacancies, API-provided URL, sources and terms. Additional columns retain original API identifiers, provenance and quality/review flags. Lists/objects are JSON strings inside cells; optional missing values are empty. Missing descriptions are flagged, not invented. UTF-8 with a BOM supports convenient Windows/Excel opening.
- `excluded_or_review.json`: ids and date/geography details for candidates outside the requested scope or requiring review. Their originals remain available.
- `summary.json` and `summary.txt`: downloaded occurrences, unique candidates, unique in-period Swedish candidates, observed publication-date range, counts by source/term, per-query/window status and unresolved issues. Overlapping windows, terms and sources cause repeated downloads; term counts are not additive.

The collector prints absolute output paths. Downloads, reports and logs are Git-ignored. Candidates are held in memory; raw API pages are saved immediately. This implementation is intended for a portfolio-sized keyword collection, not national archive ingestion.

### Completeness and bulk alternatives

Both live contracts currently cap `offset` at 2000 and `limit` at 100. Overflowing date windows split into smaller windows. An unresolved one-day overflow, failed request, missing/non-exact or changing total, stalled pagination or interruption is logged and marks the run **incomplete**. Successfully returned data remains saved. Exit code `2` indicates incomplete results (`$LASTEXITCODE` in PowerShell); `0` means configured queries were exhausted. Invalid configuration stops before collection.

**Query exhaustion does not establish complete market coverage.** Every summary includes `market_coverage_complete: false`: these are keyword candidates from Platsbanken, not every Swedish vacancy. Source coverage, archive lag, active-ad changes and the search-term list limit recall. Running during October 21 cannot cover later that day; rerun after the day closes and allow for ingestion lag. Review uncertain matches and quality flags in Step 3.

The official [bulk catalogue](https://data.jobtechdev.se/annonser/historiska/index.html) provides historical JSONL ZIP datasets. On October 21, 2026 it listed 2026 Q1-Q3 but no Q4 file. Q2/Q3 could supplement April-September if query limits remain unresolved, but cannot cover October. Bulk ingestion is an investigated alternative, not implemented or claimed complete. JobStream snapshots are a documented alternative for active data, not a replacement for historical coverage. See [API notes](docs/api_notes.md).

### Offline tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Mocked tests check pagination, date-window splitting and unresolved limits, retry/timeout behavior, cross-source identifier deduplication, archived versions, missing values, Swedish date boundaries, geography and partial-failure reporting. They use temporary project-local folders and never modify Step 1 data or contact the APIs.

## Live validation on October 21, 2026

The corrected collector completed all 80 configured source/term searches in run `20261021T124805411572Z` with no reported API or pagination issues:

- 11,887 downloaded occurrences across 183 raw API pages.
- 5,081 unique candidates before exact date/geography filtering.
- 5,023 in-period Swedish candidates in the [cleaned CSV](data/collections/20261021T124805411572Z/jobs.csv): 835 found in active data and 4,188 in historical data.
- 58 candidates excluded from the CSV or retained for date/geography review.
- Observed publication dates: April 21 through October 21, 2026.

See the [run summary](data/collections/20261021T124805411572Z/summary.txt) for term-level counts. These are uncertain candidates, not classified relevant jobs or complete market coverage. This run occurred during October 21. An earlier interrupted validation run is preserved separately and explicitly marked incomplete.

All 19 mocked tests passed in the existing virtual environment. The saved CSV's required columns, unique ids, dates, provenance and row counts were verified against the raw pages. SHA-256 checks confirmed that the Step 1 notebook and original raw download remained byte-for-byte unchanged.

## Future steps (not implemented)

1. Step 3: review uncertain candidates, resolve quality flags and define role classifications.
2. Evaluate coverage gaps and historical bulk supplementation if needed.
3. Extract technical skills, language, education and experience requirements.
4. Compare demand and monthly trends, documenting coverage limitations.

No databases, dashboards, machine learning, cloud infrastructure, scheduling or scraping of other job websites are included.
