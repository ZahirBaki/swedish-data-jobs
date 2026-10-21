"""Collect auditable JobTech candidates. Run: python -m src.collect_jobs --help."""

import argparse
import csv
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWEDEN = "i46j_HmG_v64"
SOURCES = {
    "jobsearch": "https://jobsearch.api.jobtechdev.se",
    "historical": "https://historical.api.jobtechdev.se",
}
FIELDS = [
    "advertisement_id", "job_title", "employer", "publication_date",
    "municipality", "region", "description_text", "occupation_classification",
    "number_of_vacancies", "original_advertisement_url", "source_api",
    "search_terms", "provenance", "api_identifiers", "needs_classification", "match_status",
    "geography_status", "quality_flags",
]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def obj(value):
    return value if isinstance(value, dict) else {}


def publication_day(value):
    """Compare Swedish calendar days; naive JobTech timestamps are Swedish local time."""
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is not None:
            stamp = stamp.astimezone(ZoneInfo("Europe/Stockholm"))
        return stamp.date()
    except (ValueError, TypeError):
        return None


def geography(ad):
    address = obj(ad.get("workplace_address"))
    if address.get("country_concept_id") == SWEDEN:
        return "sweden"
    if str(address.get("country_code", "")).upper() in {"199", "SE", "SWE"}:
        return "sweden"
    if str(address.get("country", "")).casefold() in {"sverige", "sweden"}:
        return "sweden"
    if any(address.get(k) for k in ("country", "country_code", "country_concept_id")):
        return "outside_sweden"
    return "unknown"


def stable_key(ad, source=None):
    """Link historical original_id to active AF id; never merge external_ids."""
    original = ad.get("original_id")
    if source == "historical" and original is not None and str(original).strip().isdigit():
        return str(original).strip(), False
    identifier = ad.get("id")
    if identifier is not None and str(identifier).strip():
        identifier = str(identifier).strip()
        # Non-numeric historical ids are archive ids, not active Platsbanken ids.
        return (f"{source}:{identifier}" if source and not identifier.isdigit() else identifier), False
    digest = hashlib.sha256(json.dumps(ad, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return "missing-id-sha256:" + digest, True


@dataclass
class Settings:
    start: date
    end: date
    terms: list
    page_size: int = 100
    delay: float = 1.0
    timeout: float = 90.0
    retries: int = 3
    phrase_search: bool = True

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError("Start date must be on or before end date.")
        if not self.terms or any(not isinstance(t, str) or not t.strip() for t in self.terms):
            raise ValueError("Provide at least one non-empty search term.")
        self.terms = list(dict.fromkeys(t.strip() for t in self.terms))
        if not 1 <= self.page_size <= 100 or self.delay < 0 or self.timeout <= 0 or self.retries < 0:
            raise ValueError("Invalid page size, delay, timeout, or retry count.")


class ApiClient:
    """Sequential requests with rate limiting, bounded retries and Retry-After support."""

    def __init__(self, settings, session=None, sleep=time.sleep, clock=time.monotonic):
        self.settings = settings
        self.session = session or requests.Session()
        self.sleep, self.clock = sleep, clock
        self.last_request = None

    def get(self, url, params=None):
        for attempt in range(self.settings.retries + 1):
            if self.last_request is not None:
                self.sleep(max(0, self.settings.delay - (self.clock() - self.last_request)))
            self.last_request = self.clock()
            response = None
            try:
                response = self.session.get(
                    url, params=params, timeout=self.settings.timeout,
                    headers={"Accept": "application/json", "User-Agent": "SwedishDataJobsPortfolio/0.2"},
                )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                status = response.status_code if response is not None else None
                retryable = status is None or status in {408, 429, 500, 502, 503, 504}
                LOGGER.warning("Request error url=%s params=%s attempt=%s status=%s: %s",
                               url, params, attempt + 1, status, exc)
                if not retryable or attempt == self.settings.retries:
                    raise RuntimeError(f"Request failed for {url}: {exc}") from exc
                wait = 2 ** attempt
                retry_after = response.headers.get("Retry-After") if response is not None else None
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after))
                    except ValueError:
                        try:
                            wait = max(wait, (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds())
                        except (ValueError, TypeError, OverflowError):
                            pass
                self.sleep(wait)


def search_contract(spec):
    """Read the live official Swagger; fail rather than guess pagination capabilities."""
    parameters = spec["paths"]["/search"]["get"]["parameters"]
    if not isinstance(parameters, list) or any(not isinstance(p, dict) for p in parameters):
        raise ValueError("Invalid search parameter specification.")
    params = {p["name"]: p for p in parameters if p.get("in") == "query"}
    if not {"q", "country", "limit", "offset"} <= params.keys():
        raise ValueError("Official specification lacks required search parameters.")
    offset_max = params["offset"].get("maximum")
    limit_max = params["limit"].get("maximum")
    if not isinstance(offset_max, int) or offset_max < 0 or not isinstance(limit_max, int) or limit_max < 1:
        raise ValueError("Official specification does not declare usable pagination limits.")
    return params


class Collector:
    def __init__(self, settings, run_dir, client=None):
        self.settings, self.run_dir = settings, Path(run_dir)
        self.client = client or ApiClient(settings)
        self.records, self.queries, self.issues = {}, [], []
        self.downloaded = 0
        self.page_number = 0

    def issue(self, message):
        self.issues.append(message)
        LOGGER.warning(message)

    def retain(self, ad, source, term):
        key, missing_id = stable_key(ad, source)
        record = self.records.setdefault(key, {"variants": [], "provenance": []})
        if not any(v["source_api"] == source and v["advertisement"] == ad for v in record["variants"]):
            record["variants"].append({"source_api": source, "advertisement": ad})
        found_by = {"source_api": source, "search_term": term}
        if found_by not in record["provenance"]:
            record["provenance"].append(found_by)
        if missing_id and len(record["provenance"]) == 1:
            self.issue(f"Advertisement lacks stable id; content hash used: {key}")
        if publication_day(ad.get("publication_date")) is None:
            message = f"Advertisement {key} has missing/invalid publication_date; retained for review."
            if message not in self.issues:
                self.issue(message)

    def query(self, source, term, contract, start, end):
        """Fetch a window. Return whether it hit a limit and can be split."""
        stat = {"source_api": source, "search_term": term, "start": str(start), "end": str(end),
                "downloaded": 0, "unique_ids": 0, "api_total": None, "status": "running"}
        self.queries.append(stat)
        query = '"' + term.replace('"', '') + '"' if self.settings.phrase_search and " " in term else term
        params = {"q": query, "country": SWEDEN, "limit": min(self.settings.page_size, contract["limit"]["maximum"])}
        if "sort" in contract and "id" in contract["sort"].get("enum", []):
            params["sort"] = "id"
        # Historical API supports absolute dates. JobSearch date parameters may use
        # relative minutes; only use absolute dates where the live schema documents them.
        date_filters = all(
            contract.get(k, {}).get("type") == "string"
            and "YYYY" in contract[k].get("description", "")
            for k in ("published-after", "published-before")
        )
        if date_filters:
            # Pad to avoid exclusive/inclusive API boundary differences, then filter locally.
            params.update({"published-after": str(start - timedelta(days=1)),
                           "published-before": str(end + timedelta(days=1))})
            if source == "jobsearch":
                # JobSearch documents a timestamp; Historical also accepts date-only values.
                for name in ("published-after", "published-before"):
                    params[name] += "T00:00:00"
        offset, seen, expected = 0, set(), None
        page_fingerprints = set()
        max_offset = contract["offset"]["maximum"]
        try:
            while offset <= max_offset:
                page_params = dict(params, offset=offset)
                payload = self.client.get(SOURCES[source] + "/search", page_params)
                self.page_number += 1
                # Save before validating so malformed/partial responses remain inspectable.
                write_json(self.run_dir / "pages" / f"{self.page_number:06d}.json", {
                    "source_api": source, "search_term": term, "parameters": page_params,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(), "response": payload,
                })
                if not isinstance(payload, dict) or not isinstance(payload.get("hits"), list):
                    raise ValueError("Expected a JSON object containing a hits list.")
                hits = payload["hits"]
                if any(not isinstance(ad, dict) for ad in hits):
                    raise ValueError("Search hits must be advertisement objects.")
                for ad in hits:
                    self.downloaded += 1
                    stat["downloaded"] += 1
                    seen.add(stable_key(ad)[0])
                    self.retain(ad, source, term)
                stat["unique_ids"] = len(seen)
                if payload.get("timed_out") or obj(payload.get("_shards")).get("failed", 0):
                    raise ValueError("API reported a partial/timed-out search response.")
                total = payload.get("total")
                value = total.get("value") if isinstance(total, dict) else total
                relation = total.get("relation", "eq") if isinstance(total, dict) else "eq"
                exact = isinstance(value, int) and not isinstance(value, bool) and value >= 0 and relation == "eq"
                if not exact:
                    raise ValueError("API total is missing or not exact; cannot verify pagination.")
                if expected is not None and expected != value:
                    raise ValueError("API total changed during pagination; rerun required.")
                expected = value
                stat["api_total"] = value
                fingerprint = hashlib.sha256(json.dumps(hits, sort_keys=True).encode()).hexdigest()
                if hits and fingerprint in page_fingerprints:
                    raise ValueError("Pagination stopped making progress: an identical page repeated.")
                page_fingerprints.add(fingerprint)
                if offset + len(hits) > expected:
                    raise ValueError("Received more advertisement hits than the reported total.")
                # Historical searches can contain multiple hits with the SAME archive id.
                # The total counts hits, not unique ids; deduplicate only after paging.
                if offset + len(hits) == expected:
                    stat["status"] = "exhausted"
                    LOGGER.info("Exhausted source=%s term=%s window=%s..%s hits=%s unique_api_ids=%s",
                                source, term, start, end, offset + len(hits), len(seen))
                    return False
                if not hits:
                    raise ValueError("Pagination stopped making progress before the reported total.")
                offset += len(hits)
                if offset > max_offset:
                    stat["status"] = "split" if date_filters and start < end else "incomplete"
                    LOGGER.warning("Pagination limit source=%s term=%s window=%s..%s total=%s unique=%s",
                                   source, term, start, end, expected, len(seen))
                    if stat["status"] == "split":
                        return True
                    raise ValueError("Pagination limit reached; official bulk data or narrower queries required.")
            raise ValueError("Pagination limit reached.")
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            stat["status"] = "incomplete"
            stat["error"] = str(exc)
            self.issue(f"Incomplete collection: {source} / {term} / {start}..{end}: {exc}")
            return False

    def collect_window(self, source, term, contract, start, end):
        if self.query(source, term, contract, start, end):
            middle = start + (end - start) // 2
            self.collect_window(source, term, contract, start, middle)
            self.collect_window(source, term, contract, middle + timedelta(days=1), end)

    def run(self):
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "pages").mkdir()
        config = asdict(self.settings)
        config.update(start=str(self.settings.start), end=str(self.settings.end))
        write_json(self.run_dir / "configuration.json", config)
        try:
            for source, url in SOURCES.items():
                try:
                    spec = self.client.get(url + "/swagger.json")
                    write_json(self.run_dir / f"{source}_swagger.json", spec)
                    contract = search_contract(spec)
                except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                    self.issue(f"Cannot collect {source}: official API specification unavailable/unsupported: {exc}")
                    for term in self.settings.terms:
                        self.queries.append({"source_api": source, "search_term": term,
                                             "status": "incomplete", "downloaded": 0, "error": str(exc)})
                    continue
                for term in self.settings.terms:
                    self.collect_window(source, term, contract, self.settings.start, self.settings.end)
        except KeyboardInterrupt:
            self.issue("Collection interrupted by user; results are partial.")
        except Exception as exc:
            # Keep the evidence already collected, including unexpected malformed responses.
            LOGGER.exception("Unexpected collection failure")
            self.issue(f"Collection stopped unexpectedly: {exc}")
        finally:
            summary = self.save()
        return summary

    def cleaned(self, key, record):
        # Prefer active ad when both APIs return the same id; retain ALL originals separately.
        eligible = [v for v in record["variants"]
                    if (day := publication_day(v["advertisement"].get("publication_date"))) is not None
                    and self.settings.start <= day <= self.settings.end
                    and geography(v["advertisement"]) == "sweden"]
        if not eligible:
            return None
        chosen = next((v for v in eligible if v["source_api"] == "jobsearch"), eligible[0])
        ad = chosen["advertisement"]
        address = obj(ad.get("workplace_address"))
        flags = []
        if key.startswith("missing-id-sha256:"):
            flags.append("missing_stable_id")
        if not obj(ad.get("description")).get("text"):
            flags.append("missing_plain_text_description")
        dates = {v["advertisement"].get("publication_date") for v in record["variants"]}
        if len(dates) > 1:
            flags.append("conflicting_publication_dates")
        return {
            "advertisement_id": key, "job_title": ad.get("headline"),
            "employer": obj(ad.get("employer")).get("name"),
            "publication_date": ad.get("publication_date"),
            "municipality": address.get("municipality"), "region": address.get("region"),
            "description_text": obj(ad.get("description")).get("text"),
            "occupation_classification": {k: ad.get(k) for k in ("occupation", "occupation_group", "occupation_field")},
            "number_of_vacancies": ad.get("number_of_vacancies"),
            "original_advertisement_url": ad.get("webpage_url"),
            "source_api": sorted({p["source_api"] for p in record["provenance"]}),
            "search_terms": sorted({p["search_term"] for p in record["provenance"]}),
            "provenance": record["provenance"], "needs_classification": True,
            "api_identifiers": [{"source_api": v["source_api"], "id": v["advertisement"].get("id"),
                                 "original_id": v["advertisement"].get("original_id")} for v in record["variants"]],
            "match_status": "uncertain_keyword_candidate", "geography_status": "sweden",
            "quality_flags": flags,
        }

    def save(self):
        for query in self.queries:
            if query["status"] == "running":
                query["status"] = "incomplete"
        rows, review = [], []
        with (self.run_dir / "candidates.jsonl").open("w", encoding="utf-8") as file:
            for key, record in self.records.items():
                row = self.cleaned(key, record)
                disposition = "included" if row else "excluded_or_requires_date_geography_review"
                file.write(json.dumps({"advertisement_id": key, "disposition": disposition, **record}, ensure_ascii=False) + "\n")
                if row:
                    rows.append(row)
                else:
                    review.append({"advertisement_id": key, "variants": [
                        {"source_api": v["source_api"],
                         "publication_date": v["advertisement"].get("publication_date"),
                         "parsed_publication_day": str(day) if (day := publication_day(v["advertisement"].get("publication_date"))) else None,
                         "geography_status": geography(v["advertisement"])}
                        for v in record["variants"]]})
        with (self.run_dir / "jobs.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                                 for k, v in row.items()})
        write_json(self.run_dir / "excluded_or_review.json", review)
        term_counts = []
        clean_keys = {r["advertisement_id"] for r in rows}
        for source in SOURCES:
            for term in self.settings.terms:
                keys = {key for key, record in self.records.items()
                        if {"source_api": source, "search_term": term} in record["provenance"]}
                term_counts.append({"source_api": source, "search_term": term,
                                    "downloaded": sum(q["downloaded"] for q in self.queries
                                                      if q["source_api"] == source and q["search_term"] == term),
                                    "unique_candidates": len(keys), "unique_in_period_sweden": len(keys & clean_keys)})
        days = [publication_day(row["publication_date"]) for row in rows]
        source_counts = []
        for source in SOURCES:
            keys = {key for key, record in self.records.items()
                    if any(p["source_api"] == source for p in record["provenance"])}
            source_counts.append({"source_api": source,
                                  "downloaded": sum(q["downloaded"] for q in self.queries if q["source_api"] == source),
                                  "unique_candidates": len(keys), "unique_in_period_sweden": len(keys & clean_keys)})
        unresolved = any(q["status"] == "incomplete" for q in self.queries) or bool(self.issues)
        summary = {
            "status": "incomplete" if unresolved else "queries_exhausted",
            "market_coverage_complete": False,
            "coverage_note": "Keyword candidates from Platsbanken only; query exhaustion does not establish complete market coverage. Active/historical data can change or lag. A run during the end date cannot cover the rest of that day.",
            "requested_start": str(self.settings.start), "requested_end_inclusive": str(self.settings.end),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "downloaded_occurrences": self.downloaded, "unique_candidates": len(self.records),
            "unique_in_period_sweden": len(rows), "excluded_or_review": len(review),
            "publication_date_min": str(min(days)) if days else None,
            "publication_date_max": str(max(days)) if days else None,
            "by_source": source_counts, "by_term_and_source": term_counts,
            "queries": self.queries, "issues": self.issues,
        }
        write_json(self.run_dir / "summary.json", summary)
        lines = [f"Status: {summary['status']}", f"Downloaded occurrences: {self.downloaded}",
                 f"Unique candidates: {len(self.records)}", f"In-period Swedish candidates: {len(rows)}",
                 f"Publication dates: {summary['publication_date_min']} to {summary['publication_date_max']}",
                 "", "Source | Search term | Downloaded | Unique candidates | In-period Sweden"]
        lines.extend(f"{r['source_api']} | {r['search_term']} | {r['downloaded']} | {r['unique_candidates']} | {r['unique_in_period_sweden']}" for r in term_counts)
        lines.extend(["", summary["coverage_note"], "", *self.issues])
        (self.run_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 4, 21))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 21), help="Inclusive Swedish calendar date")
    parser.add_argument("--terms-file", type=Path, default=PROJECT_ROOT / "config" / "search_terms.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "collections")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--delay", type=float, default=1.0, help="Minimum seconds between requests")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=3, help="Retries after the first attempt")
    parser.add_argument("--broad-query", action="store_true", help="Use unquoted multiword queries instead of phrase searches")
    args = parser.parse_args(argv)
    try:
        terms = json.loads(args.terms_file.read_text(encoding="utf-8"))
        if not isinstance(terms, list):
            raise ValueError("Terms file must contain a JSON list of strings.")
        settings = Settings(args.start, args.end, terms, args.page_size, args.delay, args.timeout, args.retries,
                            phrase_search=not args.broad_query)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output_dir.resolve() / run_id
    log_path = args.output_dir.resolve() / f"{run_id}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()], force=True)
    LOGGER.info("Starting %s through %s; output=%s", args.start, args.end, run_dir)
    summary = Collector(settings, run_dir).run()
    LOGGER.info("Status=%s downloaded=%s unique=%s in-period-Sweden=%s",
                summary["status"], summary["downloaded_occurrences"], summary["unique_candidates"], summary["unique_in_period_sweden"])
    print(f"Results: {run_dir}\nSummary: {run_dir / 'summary.txt'}\nLog: {log_path}")
    return 2 if summary["status"] == "incomplete" else 0


if __name__ == "__main__":
    raise SystemExit(main())
