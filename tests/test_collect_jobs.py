"""Offline checks: all network responses are synthetic mocks."""

import csv
import json
import shutil
import unittest
import uuid
from datetime import date
from pathlib import Path
from unittest.mock import Mock

import requests

from src.collect_jobs import ApiClient, Collector, Settings, SWEDEN, publication_day, stable_key


def ad(identifier="1", published="2026-06-15T12:00:00", **kwargs):
    return {"id": identifier, "headline": "Data Engineer", "publication_date": published,
            "workplace_address": {"country_concept_id": SWEDEN},
            "description": {"text": "Full text\nSvenska: åäö"}, **kwargs}


def spec(max_offset=2000, dates=False):
    parameters = [{"name": name, "in": "query", **extra} for name, extra in {
        "q": {"type": "string"}, "country": {"type": "array"},
        "limit": {"maximum": 100}, "offset": {"maximum": max_offset},
        "sort": {"enum": ["id", "relevance"]},
    }.items()]
    if dates:
        parameters.extend({"name": name, "in": "query", "type": "string", "description": "YYYY-mm-dd"}
                          for name in ("published-after", "published-before"))
    return {"paths": {"/search": {"get": {"parameters": parameters}}}}


def page(hits, total=None):
    return {"total": {"value": len(hits) if total is None else total}, "hits": hits}


class CollectionTests(unittest.TestCase):
    def setUp(self):
        # Use inherited Windows permissions (Python 3.14's tempfile mode 0700
        # restricts access to sandbox identities on some Windows installations).
        self.temp = Path(__file__).resolve().parent / (".test-" + uuid.uuid4().hex)
        self.temp.mkdir()
        self.addCleanup(self.cleanup)
        self.path = self.temp / "run"
        self.settings = Settings(date(2026, 4, 21), date(2026, 10, 21), ["data engineer"], delay=0)

    def cleanup(self):
        assert self.temp.resolve().is_relative_to(Path(__file__).resolve().parent)
        shutil.rmtree(self.temp)

    def collect(self, responder):
        client = Mock()
        client.get.side_effect = responder
        collector = Collector(self.settings, self.path, client)
        summary = collector.run()
        return collector, summary, client

    def rows(self):
        with (self.path / "jobs.csv").open(encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))

    def test_pagination_follows_total_even_when_page_is_short(self):
        self.settings.page_size = 2
        offsets = []
        def respond(url, params=None):
            if url.endswith("swagger.json"):
                return spec()
            if "historical" in url:
                return page([])
            offsets.append(params["offset"])
            return page([ad(str(params["offset"]))], total=3)
        _, summary, _ = self.collect(respond)
        self.assertEqual(offsets, [0, 1, 2])
        self.assertEqual(summary["status"], "queries_exhausted")
        self.assertEqual(summary["unique_in_period_sweden"], 3)

    def test_deduplication_preserves_source_term_pairs_and_raw_variants(self):
        self.settings.terms = ["data engineer", "dataingenjör"]
        def respond(url, params=None):
            if url.endswith("swagger.json"):
                return spec()
            return page([ad(employer={"name": "Historical" if "historical" in url else "Active"})])
        collector, summary, _ = self.collect(respond)
        self.assertEqual(summary["downloaded_occurrences"], 4)
        self.assertEqual(summary["unique_candidates"], 1)
        self.assertEqual(len(collector.records["1"]["provenance"]), 4)
        self.assertEqual(len(collector.records["1"]["variants"]), 2)
        self.assertEqual(self.rows()[0]["employer"], "Active")
        self.assertEqual(json.loads(self.rows()[0]["source_api"]), ["historical", "jobsearch"])
        self.assertEqual(len(list((self.path / "pages").glob("*.json"))), 4)
        self.assertTrue(json.loads(self.rows()[0]["needs_classification"].lower()))

    def test_publication_date_inclusive_boundaries_not_deadlines(self):
        hits = [ad("10", "2026-04-21T00:00:00"), ad("11", "2026-10-21T23:59:59"),
                ad("12", "2026-04-20T23:59:59", application_deadline="2026-07-01"),
                ad("13", "2026-10-22T00:00:00"), ad("14", None),
                ad("15", "2026-10-21T22:30:00Z")]
        _, summary, _ = self.collect(lambda url, params=None: spec() if url.endswith("swagger.json") else page(hits))
        self.assertEqual({r["advertisement_id"] for r in self.rows()}, {"10", "11"})
        self.assertEqual(summary["unique_candidates"], 6)
        self.assertEqual(summary["publication_date_min"], "2026-04-21")
        self.assertEqual(summary["publication_date_max"], "2026-10-21")

    def test_missing_optional_values_and_unknown_or_foreign_geography(self):
        hits = [ad(employer=None, occupation=None, number_of_vacancies=None, description=None),
                ad("2", workplace_address=None),
                ad("3", workplace_address={"country": "Norge"})]
        _, summary, _ = self.collect(lambda url, params=None: spec() if url.endswith("swagger.json") else page(hits))
        self.assertEqual(summary["unique_in_period_sweden"], 1)
        row = self.rows()[0]
        self.assertEqual(row["employer"], "")
        self.assertEqual(row["description_text"], "")
        self.assertIn("missing_plain_text_description", row["quality_flags"])
        self.assertEqual(len(json.loads((self.path / "excluded_or_review.json").read_text())), 2)
        self.assertEqual(len((self.path / "candidates.jsonl").read_text(encoding="utf-8").splitlines()), 3)

    def test_same_title_different_ids_are_not_merged(self):
        hits = [ad("1"), ad("2")]
        _, summary, _ = self.collect(lambda url, params=None: spec() if url.endswith("swagger.json") else page(hits))
        self.assertEqual(summary["unique_candidates"], 2)

    def test_historical_archive_id_links_to_active_original_id(self):
        def respond(url, params=None):
            if url.endswith("swagger.json"):
                return spec()
            return page([ad("archive-hash", original_id="1") if "historical" in url else ad("1")])
        collector, summary, _ = self.collect(respond)
        self.assertEqual(summary["unique_candidates"], 1)
        self.assertEqual(len(collector.records["1"]["variants"]), 2)
        self.assertEqual(self.rows()[0]["advertisement_id"], "1")

    def test_historical_versions_count_as_distinct_api_hits_but_one_candidate(self):
        def respond(url, params=None):
            if url.endswith("swagger.json"):
                return spec()
            if "jobsearch" in url:
                return page([])
            return page([ad(f"archive-{params['offset']}", original_id="1")], total=2)
        _, summary, _ = self.collect(respond)
        self.assertEqual(summary["status"], "queries_exhausted")
        self.assertEqual(summary["unique_candidates"], 1)
        self.assertEqual(summary["downloaded_occurrences"], 2)

    def test_duplicate_archive_ids_with_distinct_versions_do_not_break_paging(self):
        def respond(url, params=None):
            if url.endswith("swagger.json"):
                return spec()
            if "jobsearch" in url:
                return page([])
            return page([ad("same-hash", original_id="1", headline=f"Version {params['offset']}")], total=2)
        collector, summary, _ = self.collect(respond)
        self.assertEqual(summary["status"], "queries_exhausted")
        self.assertEqual(summary["unique_candidates"], 1)
        self.assertEqual(len(collector.records["1"]["variants"]), 2)

    def test_missing_id_is_preserved_and_flagged(self):
        hit = ad(None)
        _, summary, _ = self.collect(lambda url, params=None: spec() if url.endswith("swagger.json") else page([hit]))
        self.assertEqual(summary["unique_candidates"], 1)
        self.assertEqual(summary["status"], "incomplete")
        self.assertIn("missing_stable_id", self.rows()[0]["quality_flags"])
        self.assertEqual(stable_key(hit), stable_key(dict(reversed(list(hit.items())))))

    def test_pagination_limit_is_incomplete_not_success(self):
        _, summary, _ = self.collect(lambda url, params=None: spec(0) if url.endswith("swagger.json") else page([ad()], 10))
        self.assertEqual(summary["status"], "incomplete")
        self.assertTrue(all(q["status"] == "incomplete" for q in summary["queries"]))
        self.assertIn("Pagination limit", " ".join(summary["issues"]))

    def test_large_historical_window_splits_and_uses_supported_filters(self):
        self.settings.start, self.settings.end = date(2026, 6, 1), date(2026, 6, 10)
        hits = [ad("early", "2026-06-01T12:00:00"), ad("late", "2026-06-10T12:00:00")]
        def respond(url, params=None):
            if url.endswith("swagger.json"):
                return spec(0, dates="historical" in url)
            if "jobsearch" in url:
                self.assertNotIn("published-after", params)
                return page([])
            after, before = params["published-after"], params["published-before"]
            matches = [hit for hit in hits if after < hit["publication_date"][:10] < before]
            return page(matches[:1], len(matches))
        _, summary, _ = self.collect(respond)
        self.assertEqual(summary["status"], "queries_exhausted")
        self.assertEqual(summary["unique_in_period_sweden"], 2)
        self.assertEqual([q["status"] for q in summary["queries"]], ["exhausted", "split", "exhausted", "exhausted"])

    def test_repeated_page_before_total_is_incomplete(self):
        _, summary, _ = self.collect(lambda url, params=None: spec() if url.endswith("swagger.json") else page([ad()], 3))
        self.assertEqual(summary["status"], "incomplete")
        self.assertIn("progress", " ".join(summary["issues"]))

    def test_error_retains_partial_download(self):
        def respond(url, params=None):
            if url.endswith("swagger.json"):
                return spec()
            if params["offset"]:
                raise RuntimeError("HTTP 503 after retries")
            return page([ad()], 3)
        _, summary, _ = self.collect(respond)
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["unique_candidates"], 1)
        self.assertEqual(len(self.rows()), 1)

    def test_non_exact_total_preserves_ad_and_marks_incomplete(self):
        response = {"hits": [ad()], "total": {"value": 1, "relation": "gte"}}
        _, summary, _ = self.collect(lambda url, params=None: spec() if url.endswith("swagger.json") else response)
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["unique_candidates"], 1)

    def test_unavailable_schema_marks_each_source_and_term_incomplete(self):
        def respond(url, params=None):
            raise RuntimeError("Offline")
        _, summary, _ = self.collect(respond)
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(len(summary["queries"]), 2)
        self.assertEqual(summary["downloaded_occurrences"], 0)
        self.assertEqual(len(self.rows()), 0)

    def test_date_helpers_and_configuration(self):
        self.assertIsNone(publication_day("invalid"))
        self.assertIsNone(publication_day(None))
        self.assertEqual(publication_day("2026-04-20T22:30:00Z"), date(2026, 4, 21))
        with self.assertRaises(ValueError):
            Settings(date(2026, 10, 21), date(2026, 4, 21), ["test"])

    def test_phrase_search_keeps_description_only_candidates(self):
        def respond(url, params=None):
            if url.endswith("swagger.json"):
                return spec()
            self.assertEqual(params["q"], '"data engineer"')
            return page([ad(headline="Unrelated title", description={"text": "Work with a data engineer"})])
        _, summary, _ = self.collect(respond)
        self.assertEqual(summary["unique_in_period_sweden"], 1)
        self.assertEqual(self.rows()[0]["match_status"], "uncertain_keyword_candidate")


class RequestTests(unittest.TestCase):
    def test_retry_after_and_timeout_with_rate_limiting(self):
        session, sleep = Mock(), Mock()
        limited, success = Mock(), Mock()
        limited.status_code = 429
        limited.headers = {"Retry-After": "5"}
        limited.raise_for_status.side_effect = requests.HTTPError("429")
        success.raise_for_status.return_value = None
        success.json.return_value = page([])
        session.get.side_effect = [requests.Timeout("timeout"), limited, success]
        client = ApiClient(Settings(date(2026, 4, 21), date(2026, 10, 21), ["test"]), session, sleep, clock=lambda: 0)
        self.assertEqual(client.get("https://example.test"), page([]))
        self.assertEqual(session.get.call_count, 3)
        self.assertIn(((5.0,),), [(c.args,) for c in sleep.call_args_list])
        self.assertEqual(session.get.call_args.kwargs["timeout"], 90)

    def test_permanent_http_error_does_not_retry(self):
        session = Mock()
        response = session.get.return_value
        response.status_code = 400
        response.raise_for_status.side_effect = requests.HTTPError("400")
        client = ApiClient(Settings(date(2026, 4, 21), date(2026, 10, 21), ["test"]), session)
        with self.assertRaises(RuntimeError):
            client.get("https://example.test")
        self.assertEqual(session.get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
