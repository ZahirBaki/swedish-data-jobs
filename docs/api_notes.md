# Step 2 API decisions

Checked against official documentation and live responses on October 21, 2026.

## Official references

- [JobSearch guide](https://gitlab.com/arbetsformedlingen/job-ads/jobsearch-apis/-/blob/main/docs/GettingStartedJobSearchEN.md) and [live Swagger](https://jobsearch.api.jobtechdev.se/swagger.json): active ads only.
- [Historical Swagger](https://historical.api.jobtechdev.se/swagger.json) and [official repository](https://gitlab.com/arbetsformedlingen/job-ads/historical-ads/historical-api): archived/removed ads.
- [Official historical-data example](https://gitlab.com/arbetsformedlingen/job-ads/getting-started-code-examples/jupyter-notebook-on-historical-ads/-/tree/main): Swedish geography fields, including country concept `i46j_HmG_v64` and country code `199`.

Both APIs currently document `GET /search`, free text, country, publication bounds and `sort=id`. Each permits pages up to 100 hits and offsets from 0 through 2000. The collector saves and reads the live Swagger contracts before searching; unavailable or incompatible contracts cause an incomplete run rather than guessed requests.

The JobSearch guide documents quoted phrase queries. These are the default for multiword terms, and both sources were live-tested with them. This avoids treating unrelated individual words as alternatives while retaining description-only matches. `--broad-query` enables unquoted search as an explicitly broader alternative. The exact query parameters are recorded with every raw page.

JobSearch accepts publication timestamps; Historical also documents date-only bounds. Use those respective formats, pad by one day to avoid boundary ambiguity, then enforce the exact inclusive Swedish calendar period locally. Overflowing windows split recursively down to one day. Padding causes overlap; deduplication handles it. An unresolved one-day overflow remains incomplete.

Live Historical responses have archive hashes as `id`; recent records also expose numeric Platsbanken `original_id`. Link that original identifier to active numeric `id`, retaining all original versions and source/term pairs. Without a numeric original id, retain a source-specific archive id. Do not guess links from titles or employer-specific external ids. Historical totals count hits, including repeated archive ids, so pagination counts hits separately from deduplicated candidates. Identical repeated pages are treated as a pagination failure.

## Result limits and alternatives

The official [historical bulk catalogue](https://data.jobtechdev.se/annonser/historiska/index.html) offers full JSONL ZIP datasets. At inspection it listed Q1, Q2 and Q3 2026, alongside prior annual files, but no Q4. Q2 and Q3 could supplement April-September; they cannot establish October coverage. Recheck the catalogue before using it. Bulk import is a future extension if limits remain unresolved; large archives are not downloaded automatically.

For complete active snapshots, the official [JobStream guide](https://gitlab.com/arbetsformedlingen/job-ads/jobsearch-apis/-/blob/main/docs/GettingStartedJobStreamEN.md) documents `/snapshot`. This is an alternative for active data, not a way to recover six months of removed ads. It is not implemented here.

## Coverage interpretation

`queries_exhausted` means all configured queries reached their reported totals without a detected error, after splitting overflow windows where possible. This is not complete Swedish job-market coverage. Keyword recall, Platsbanken's scope, changes during pagination and archival ingestion lag limit coverage. A run during October 21 cannot include ads posted later that day; rerun after the day ends and allow for ingestion lag.

`incomplete` identifies request/schema failures, unresolved limits, missing/non-exact or changing totals, stalled pagination, interruption, or missing identifier/date quality problems. Successfully returned originals and usable rows remain saved. Inspect the log and summary; never interpret a partial count as complete.

The collector uses the API-supplied `publication_date`, never update/removal dates or deadlines. It cannot reconstruct an earlier timestamp that the API does not expose. Conflicting dates across versions are flagged in the CSV.
