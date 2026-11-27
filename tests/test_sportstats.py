"""Deterministic provider tests: no network, API key, real delays, or model files."""

import json
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from courtiq.sportstats import (
    BASE_URL, DAILY_LIMIT, MANUAL_REFRESH_MINIMUM, RECENT_TTL, REQUEST_SPACING,
    SportStatsResults, normalize_match, parse_completed_score,
)


STAMP = "2026-01-20T12:00:00Z"
TEST_CREDENTIAL = "test-only-not-a-real-credential"


def match(code="m1", day="2026-01-15"):
    return {
        "code": code, "match_id": 999, "category_name": "ATP - Singles: Test, Hard (Indoor)",
        "tour": "atp", "discipline": "singles", "stage": "main", "surface": "hard (indoor)",
        "sport": "tennis", "start_date": day, "status": "Finished", "status_code": "final",
        "competitor1": {"code": "p1", "id": 111, "name": "A. De Minaur", "sets_won": 0, "winner": False},
        "competitor2": {"code": "p2", "id": 222, "name": "C. Alcaraz", "sets_won": 2, "winner": True},
        "set_scores": {"s1": {"competitor1": "1", "competitor2": "6", "score": "1-6"},
                       "s2": {"competitor1": "4", "competitor2": "6", "score": "4-6"}},
    }


def response(rows, offset=0, total=None, headers=None):
    return (200, headers or {}, {"success": True, "data": rows,
                                "meta": {"offset": offset, "total": len(rows) if total is None else total}})


class FakeClock:
    def __init__(self, stamp=STAMP):
        self.value = datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
        self.waits = []

    def __call__(self):
        return self.value

    def wait(self, stop_event, seconds):
        self.waits.append(seconds)
        self.value += seconds
        return stop_event.is_set()


class ScriptedTransport:
    def __init__(self, pages, clock):
        self.pages = list(pages)
        self.calls = []
        self.clock = clock

    def __call__(self, url, headers, timeout):
        self.calls.append({"url": url, "headers": headers, "time": self.clock(), "timeout": timeout})
        if not self.pages:
            raise AssertionError("Unexpected extra provider request")
        value = self.pages.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class ParserTests(unittest.TestCase):
    def test_codes_abbreviations_winner_first_and_surface(self):
        result = normalize_match(match(), STAMP)
        self.assertEqual(result["match_id"], "sportstats:m1")
        self.assertEqual(result["winner"], {"provider_id": "p2", "name": "C. Alcaraz"})
        self.assertEqual(result["loser"]["name"], "A. De Minaur")
        self.assertEqual(result["score"], "6-1 6-4")
        self.assertEqual(result["score_kind"], "games")
        self.assertEqual(result["set_scores"], [
            {"winner_games": 6, "loser_games": 1, "tiebreak_loser_points": None, "is_match_tiebreak": False},
            {"winner_games": 6, "loser_games": 4, "tiebreak_loser_points": None, "is_match_tiebreak": False},
        ])
        self.assertEqual(result["surface"], "Hard")
        self.assertEqual(result["date_basis"], "match_date")

    def test_explicit_completion_is_required(self):
        for status, code in (("Set 2", "live"), ("Finished", "live"), ("Retired", "final"),
                             ("Walkover", "final"), ("Cancelled", "cancelled"), ("", "")):
            row = match()
            row.update(status=status, status_code=code)
            self.assertIsNone(normalize_match(row, STAMP), (status, code))

    def test_contradictory_winner_flags_or_scores_are_rejected(self):
        mutations = [lambda row: row["competitor1"].update(winner=True),
                     lambda row: row["competitor2"].update(winner=False),
                     lambda row: row["competitor2"].update(winner="true"),
                     lambda row: row["competitor1"].update(sets_won=3),
                     lambda row: row["competitor2"].update(sets_won=1),
                     lambda row: row["competitor1"].update(sets_won=True),
                     lambda row: row["set_scores"]["s1"].update(competitor1="6", competitor2="1")]
        for mutate in mutations:
            row = match()
            mutate(row)
            self.assertIsNone(normalize_match(row, STAMP))

    def test_filters_are_rechecked_locally(self):
        for key, value in (("tour", "wta"), ("tour", "challenger men"), ("discipline", "doubles"),
                           ("stage", "qualification"), ("retired", True), ("score", "6-0 6-1 RET")):
            row = match()
            row[key] = value
            self.assertIsNone(normalize_match(row, STAMP), (key, value))
        row = match()
        row["competitor1"]["member_ids"] = [1, 2]
        self.assertIsNone(normalize_match(row, STAMP))

    def test_partial_score_is_labelled_sets_and_unknown_fields_stay_null(self):
        row = match()
        row.pop("set_scores")
        row.pop("surface")
        row.pop("category_name")
        result = normalize_match(row, STAMP)
        self.assertEqual(result["score"], "2-0 sets")
        self.assertEqual(result["score_kind"], "sets")
        self.assertEqual(result["set_scores"], [])
        self.assertIsNone(result["surface"])
        self.assertIsNone(result["event"])

    def test_tiebreak_and_numeric_id_fallback(self):
        row = match()
        row.pop("code")
        for player in ("competitor1", "competitor2"):
            row[player].pop("code")
        row["set_scores"]["s2"].update(competitor1="6", competitor2="7", score="6-7(4)", tiebreak1=4, tiebreak2=7)
        result = normalize_match(row, STAMP)
        self.assertEqual(result["match_id"], "sportstats:999")
        self.assertEqual(result["score"], "6-1 7-6(4)")
        self.assertEqual(result["winner"]["provider_id"], "222")

    def test_observed_dotted_tiebreak_zero_points_and_winner_orientation(self):
        # The provider returned these two score payloads on 2026-11-27.
        first = match("ms84u", "2026-11-21")
        first["competitor1"].update(winner=True, sets_won=2)
        first["competitor2"].update(winner=False, sets_won=0)
        first["set_scores"] = {
            "s1": {"competitor1": "7.7", "competitor2": "6.0", "score": "7-6(0)", "tiebreak1": 7, "tiebreak2": 0},
            "s2": {"competitor1": "6", "competitor2": "4", "score": "6-4"},
        }
        second = match("mza0u", "2026-11-22")
        second["set_scores"] = {
            "s1": {"competitor1": "2", "competitor2": "6", "score": "2-6"},
            "s2": {"competitor1": "6.7", "competitor2": "7.9", "score": "6-7(7)", "tiebreak1": 7, "tiebreak2": 9},
        }
        for raw, expected, points in ((first, "7-6(0) 6-4", 0), (second, "6-2 7-6(7)", 7)):
            result = normalize_match(raw, STAMP)
            self.assertEqual(result["score"], expected)
            self.assertEqual(result["score_kind"], "games")
            self.assertEqual(next(s["tiebreak_loser_points"] for s in result["set_scores"]
                                  if s["tiebreak_loser_points"] is not None), points)
            self.assertEqual(result["set_scores"], parse_completed_score(expected))

    def test_dotted_notation_is_not_converted_to_a_float(self):
        row = match()
        row["set_scores"]["s2"] = {"competitor1": "6.10", "competitor2": "7.12"}
        result = normalize_match(row, STAMP)
        self.assertEqual(result["score"], "6-1 7-6(10)")
        self.assertEqual(result["set_scores"][1]["tiebreak_loser_points"], 10)

    def test_documented_score_string_can_fill_missing_game_cells(self):
        row = match()
        row["set_scores"] = {"s1": {"score": "1-6"}, "s2": {"score": "6-7(0)"}}
        result = normalize_match(row, STAMP)
        self.assertEqual(result["score"], "6-1 7-6(0)")
        row["set_scores"]["s2"]["competitor1"] = "7.7"
        self.assertIsNone(normalize_match(row, STAMP), "Conflicting cells may not be filled from score text")

    def test_invalid_optional_tiebreak_metadata_does_not_destroy_known_games(self):
        row = match()
        row["set_scores"]["s2"] = {
            "competitor1": "6", "competitor2": "7", "score": "6-7(7)", "tiebreak1": 7, "tiebreak2": 8,
        }
        result = normalize_match(row, STAMP)
        self.assertEqual(result["score"], "6-1 7-6")
        self.assertIsNone(result["set_scores"][1]["tiebreak_loser_points"])
        row["set_scores"]["s2"].update(tiebreak1=4, tiebreak2=7)
        result = normalize_match(row, STAMP)
        self.assertEqual(result["score"], "6-1 7-6", "Contradictory optional points remain unknown")

    def test_set_losses_are_kept_in_chronological_match_winner_order(self):
        row = match()
        row["competitor1"]["sets_won"] = 1
        row["set_scores"] = {
            "s1": {"competitor1": "7.7", "competitor2": "6.5", "score": "7-6(5)", "tiebreak1": 7, "tiebreak2": 5},
            "s2": {"competitor1": "3", "competitor2": "6", "score": "3-6"},
            "s3": {"competitor1": "4", "competitor2": "6", "score": "4-6"},
        }
        result = normalize_match(row, STAMP)
        self.assertEqual(result["score"], "6-7(5) 6-3 6-4")
        self.assertEqual(result["set_scores"][0]["winner_games"], 6)
        self.assertEqual(result["set_scores"][0]["tiebreak_loser_points"], 5)

    def test_partial_games_are_not_presented_as_a_complete_match_score(self):
        row = match()
        row["set_scores"].pop("s2")
        result = normalize_match(row, STAMP)
        self.assertEqual(result["score"], "2-0 sets")
        self.assertEqual(result["set_scores"], [])
        row["set_scores"]["s3"] = {"score": "6-0"}
        self.assertIsNone(normalize_match(row, STAMP))

    def test_legacy_score_parser_preserves_tiebreaks_without_inventing_format(self):
        accepted = ("6-4 7-6(0)", "6–7(5) 6–3 6–4", "6-4 6-2 6-3",
                    "6-4 4-6 6-3 3-6 8-6", "4-6 6-4 [10-8]", "4-6 6-4 10-8")
        for value in accepted:
            self.assertTrue(parse_completed_score(value), value)
        self.assertTrue(parse_completed_score("4-6 6-4 [10-8]")[-1]["is_match_tiebreak"])
        self.assertFalse(parse_completed_score("4-6 6-4 10-8")[-1]["is_match_tiebreak"])
        rejected = (None, "", "2-0 sets", "6-4", "6-4 4-6", "6-4 3-2", "6-4 6-4 RET",
                    "6-4 6-4 4-6", "6-4(3) 6-4", "6-4 7-7", "[10-8] 6-4", "4-6 6-4 [10-9]")
        for value in rejected:
            self.assertEqual(parse_completed_score(value), [], value)


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.cache_path = Path(self.directory.name) / "cache.json"
        self.clock = FakeClock()
        self.instances = []

    def tearDown(self):
        for provider in self.instances:
            provider.close()
        self.directory.cleanup()

    def provider(self, pages, key=TEST_CREDENTIAL, clock=None):
        clock = clock or self.clock
        transport = ScriptedTransport(pages, clock)
        provider = SportStatsResults(self.directory.name, api_key=key, clock=clock,
                                    transport=transport, cache_path=self.cache_path)
        self.instances.append(provider)
        return provider, transport

    def run_sync(self, provider):
        provider.refresh()
        if provider._thread:
            provider._thread.join(timeout=3)
            self.assertFalse(provider._thread.is_alive(), "Fake-clock worker did not finish")
        return provider.snapshot()

    def test_two_pages_are_required_before_range_is_covered(self):
        first_page = [match("m" + str(index)) for index in range(100)]
        provider, transport = self.provider([response(first_page, 0, 101), response([match("m100")], 100, 101)])
        result = self.run_sync(provider)
        self.assertEqual(result["status"]["state"], "ready")
        self.assertEqual(result["status"]["matches"], 101)
        self.assertTrue(result["status"]["coverage_complete"])
        self.assertEqual(result["status"]["covered_ranges"], [{"start": "2026-01-01", "end": "2026-01-20"}])
        self.assertEqual(result["status"]["progress"]["percent"], 100)
        query = parse_qs(urlsplit(transport.calls[1]["url"]).query)
        self.assertEqual(query["offset"], ["100"])
        self.assertEqual(query["limit"], ["100"])
        self.assertEqual(query["tour"], ["atp"])
        self.assertGreaterEqual(transport.calls[1]["time"] - transport.calls[0]["time"], REQUEST_SPACING)
        provider.refresh()
        provider._thread.join(timeout=3)
        self.assertEqual(len(transport.calls), 2, "Fresh recent window should not be requested twice")

    def test_interrupted_pagination_resumes_without_claiming_coverage(self):
        first_page = [match("m" + str(index)) for index in range(100)]
        provider, _ = self.provider([response(first_page, 0, 101), (500, {}, {})])
        result = self.run_sync(provider)
        self.assertFalse(result["status"]["coverage_complete"])
        self.assertEqual(result["status"]["covered_ranges"], [])
        self.assertEqual(result["status"]["state"], "error")
        provider.close()
        resumed, transport = self.provider([response([match("m100")], 100, 101)])
        result = self.run_sync(resumed)
        self.assertEqual(parse_qs(urlsplit(transport.calls[0]["url"]).query)["offset"], ["100"])
        self.assertEqual(len(result["records"]), 101)
        self.assertTrue(result["status"]["coverage_complete"])

    def test_latest_window_is_first_and_earlier_year_is_backfilled(self):
        clock = FakeClock("2026-03-10T12:00:00Z")
        provider, transport = self.provider([response([]), response([]), response([])], clock=clock)
        result = self.run_sync(provider)
        queries = [parse_qs(urlsplit(call["url"]).query) for call in transport.calls]
        self.assertEqual((queries[0]["date_from"], queries[0]["date_to"]), (["2026-02-11"], ["2026-03-10"]))
        self.assertEqual((queries[-1]["date_from"], queries[-1]["date_to"]), (["2026-01-01"], ["2026-01-13"]))
        self.assertTrue(result["status"]["coverage_complete"])

    def test_cache_is_available_without_a_key_and_contains_no_credential(self):
        provider, _ = self.provider([response([match()])])
        self.run_sync(provider)
        provider.close()
        cached, transport = self.provider([], key="")
        result = self.run_sync(cached)
        self.assertEqual(result["status"]["state"], "disabled")
        self.assertFalse(result["status"]["configured"])
        self.assertEqual(len(result["records"]), 1)
        self.assertFalse(transport.calls)
        self.assertNotIn(TEST_CREDENTIAL, self.cache_path.read_text())
        self.assertNotIn(TEST_CREDENTIAL, json.dumps(result))
        self.assertNotIn("authorization", self.cache_path.read_text().lower())

    def test_old_cached_game_scores_gain_validated_structured_sets(self):
        row = normalize_match(match(), STAMP)
        row.pop("set_scores")
        row.pop("score_observed_at")
        row["score"] = "6-1 7-6(0)"
        self.cache_path.write_text(json.dumps({"version": 1, "records": [row]}))
        provider, transport = self.provider([], key="")
        result = provider.snapshot()["records"][0]
        self.assertEqual(result["set_scores"], parse_completed_score("6-1 7-6(0)"))
        self.assertEqual(result["score_observed_at"], STAMP)
        self.assertEqual(transport.calls, [])

    def test_unverified_cached_arrays_cannot_turn_aggregate_into_game_scores(self):
        raw = match()
        raw.pop("set_scores")
        row = normalize_match(raw, STAMP)
        row["set_scores"] = parse_completed_score("6-1 6-4")
        self.cache_path.write_text(json.dumps({"version": 1, "records": [row]}))
        provider, _ = self.provider([], key="")
        result = provider.snapshot()["records"][0]
        self.assertEqual(result["score"], "2-0 sets")
        self.assertEqual(result["set_scores"], [])

    def test_legacy_aggregate_cache_refreshes_recent_window_once_to_recover_sets(self):
        incomplete = match()
        incomplete.pop("set_scores")
        provider, _ = self.provider([response([incomplete])])
        self.run_sync(provider)
        provider.close()
        old_cache = json.loads(self.cache_path.read_text())
        old_cache.pop("score_schema_version")
        for record in old_cache["records"]:
            record.pop("set_scores")
        self.cache_path.write_text(json.dumps(old_cache))
        refreshed, transport = self.provider([response([match()])])
        result = self.run_sync(refreshed)
        self.assertEqual(len(transport.calls), 1, "Migration should reuse the bounded results query")
        self.assertEqual(result["records"][0]["score"], "6-1 6-4")
        refreshed.close()
        restarted, no_calls = self.provider([])
        self.run_sync(restarted)
        self.assertEqual(no_calls.calls, [], "A migrated fresh cache must not refetch on each launch")

    def test_legacy_older_aggregate_revisits_only_affected_window_without_hiding_records(self):
        clock = FakeClock("2026-03-10T12:00:00Z")
        raw = match("older", "2026-01-15")
        raw.pop("set_scores")
        older = normalize_match(raw, STAMP)
        recent = normalize_match(match("recent", "2026-03-09"), "2026-03-10T12:00:00Z")
        self.cache_path.write_text(json.dumps({
            "version": 1, "records": [older, recent],
            "covered_ranges": [{"start": "2026-01-01", "end": "2026-03-10"}],
            "recent_refresh": clock(), "recent_range": "2026-02-11/2026-03-10",
            "checkpoints": [{"start": "2026-01-14", "end": "2026-02-10", "offset": 100,
                             "seen": ["sportstats:older"], "total": 101}],
        }))
        provider, transport = self.provider([response([match("older", "2026-01-15")])], clock=clock)
        pending = provider.snapshot()
        self.assertEqual(len(pending["records"]), 2, "Migration must keep existing records visible")
        self.assertFalse(pending["status"]["coverage_complete"])
        finished = self.run_sync(provider)
        self.assertEqual(len(transport.calls), 1)
        query = parse_qs(urlsplit(transport.calls[0]["url"]).query)
        self.assertEqual((query["date_from"], query["date_to"], query["offset"]),
                         (["2026-01-14"], ["2026-02-10"], ["0"]))
        recovered = next(row for row in finished["records"] if row["match_id"] == "sportstats:older")
        self.assertEqual(recovered["score"], "6-1 6-4")
        self.assertTrue(finished["status"]["coverage_complete"])
        self.assertEqual(len(finished["records"]), 2)
        self.assertEqual(provider._recent_refresh, clock(), "Older recovery should not disturb recent freshness")
        provider.close()
        restarted, transport = self.provider([], clock=clock)
        self.run_sync(restarted)
        self.assertEqual(transport.calls, [], "Completed migration should stay complete after restart")

    def test_transient_detail_omission_retains_verified_scores_with_original_timestamp(self):
        provider, transport = self.provider([response([match()])])
        initial = self.run_sync(provider)["records"][0]
        self.clock.value += RECENT_TTL + 1
        missing = match()
        missing.pop("set_scores")
        transport.pages.append(response([missing]))
        current = self.run_sync(provider)["records"][0]
        self.assertEqual(current["score"], initial["score"])
        self.assertEqual(current["set_scores"], initial["set_scores"])
        self.assertEqual(current["score_observed_at"], initial["observed_at"])
        self.assertNotEqual(current["observed_at"], initial["observed_at"])

    def test_corrected_complete_or_partial_scores_replace_old_details(self):
        provider, transport = self.provider([response([match()])])
        self.run_sync(provider)
        self.clock.value += RECENT_TTL + 1
        corrected = match()
        corrected["set_scores"]["s1"] = {"competitor1": "3", "competitor2": "6", "score": "3-6"}
        transport.pages.append(response([corrected]))
        self.assertEqual(self.run_sync(provider)["records"][0]["score"], "6-3 6-4")
        self.clock.value += RECENT_TTL + 1
        corrected["set_scores"] = {"s1": {"competitor1": "0", "competitor2": "6", "score": "0-6"}}
        transport.pages.append(response([corrected]))
        current = self.run_sync(provider)["records"][0]
        self.assertEqual(current["score"], "2-0 sets", "A conflicting partial correction may not reuse old games")
        self.assertEqual(current["set_scores"], [])

    def test_changed_winner_or_set_total_never_inherits_previous_game_scores(self):
        for correction in ("winner", "sets"):
            with self.subTest(correction=correction):
                self.cache_path.unlink(missing_ok=True)
                provider, transport = self.provider([response([match()])])
                self.run_sync(provider)
                self.clock.value += RECENT_TTL + 1
                corrected = match()
                corrected.pop("set_scores")
                if correction == "winner":
                    corrected["competitor1"].update(winner=True, sets_won=2)
                    corrected["competitor2"].update(winner=False, sets_won=0)
                else:
                    corrected["competitor1"]["sets_won"] = 1
                transport.pages.append(response([corrected]))
                result = self.run_sync(provider)["records"][0]
                self.assertEqual(result["score_kind"], "sets")
                self.assertEqual(result["set_scores"], [])
                provider.close()

    def test_tiebreak_points_survive_omission_but_authoritative_changes_replace_them(self):
        raw = match()
        raw["set_scores"]["s2"] = {"competitor1": "6.0", "competitor2": "7.7", "score": "6-7(0)"}
        provider, transport = self.provider([response([raw])])
        self.run_sync(provider)
        self.clock.value += RECENT_TTL + 1
        raw["set_scores"]["s2"] = {"competitor1": "6", "competitor2": "7", "score": "6-7"}
        transport.pages.append(response([raw]))
        self.assertEqual(self.run_sync(provider)["records"][0]["score"], "6-1 7-6(0)")
        self.clock.value += RECENT_TTL + 1
        raw["set_scores"]["s2"] = {"competitor1": "6.4", "competitor2": "7.7", "score": "6-7(4)"}
        transport.pages.append(response([raw]))
        current = self.run_sync(provider)["records"][0]
        self.assertEqual(current["score"], "6-1 7-6(4)")
        self.assertEqual(current["score_observed_at"], current["observed_at"])

    def test_429_honors_retry_after_and_does_not_loop(self):
        provider, transport = self.provider([(429, {"Retry-After": "120"}, {"secret": TEST_CREDENTIAL})])
        result = self.run_sync(provider)
        self.assertEqual(result["status"]["state"], "paused")
        self.assertEqual(result["status"]["delay_seconds"], 120)
        self.assertEqual(len(transport.calls), 1)
        provider.refresh()
        self.assertEqual(len(transport.calls), 1)
        self.assertNotIn(TEST_CREDENTIAL, json.dumps(result))
        self.assertNotIn(TEST_CREDENTIAL, self.cache_path.read_text())

    def test_remote_quota_iso_reset_stops_further_pages(self):
        rows = [match("m" + str(index)) for index in range(100)]
        headers = {"X-Quota-Remaining": "0", "X-Quota-Reset": "2026-01-21T00:00:00Z", "X-Data-Delay": "900"}
        provider, transport = self.provider([response(rows, 0, 101, headers)])
        result = self.run_sync(provider)
        self.assertEqual(result["status"]["state"], "paused")
        self.assertEqual(result["status"]["quota_remaining"], 0)
        self.assertEqual(result["status"]["quota_reset"], "2026-01-21T00:00:00Z")
        self.assertEqual(result["status"]["data_delay_seconds"], 900)
        self.assertEqual(len(transport.calls), 1)
        self.assertFalse(result["status"]["coverage_complete"])

    def test_quota_error_without_headers_pauses_until_utc_midnight(self):
        provider, transport = self.provider([(429, {}, {"error": {"code": "QUOTA_EXCEEDED"}})])
        result = self.run_sync(provider)
        self.assertEqual(result["status"]["quota_remaining"], 0)
        self.assertEqual(result["status"]["quota_reset"], "2026-01-21T00:00:00Z")
        self.assertEqual(result["status"]["delay_seconds"], 12 * 60 * 60)
        self.clock.value += 120
        provider.refresh(force=True)
        self.assertEqual(len(transport.calls), 1)

    def test_local_daily_budget_and_no_plain_error_echo(self):
        provider, transport = self.provider([])
        provider._requests_today = DAILY_LIMIT
        result = self.run_sync(provider)
        self.assertEqual(result["status"]["state"], "paused")
        self.assertEqual(len(transport.calls), 0)
        provider._requests_today = 0
        transport.pages.append(RuntimeError("Authorization: Bearer " + TEST_CREDENTIAL))
        result = self.run_sync(provider)
        self.assertEqual(result["status"]["state"], "error")
        self.assertNotIn(TEST_CREDENTIAL, json.dumps(result))

    def test_401_and_redirect_are_not_followed(self):
        for status in (401, 403, 302):
            with self.subTest(status=status):
                provider, transport = self.provider([(status, {"Location": "https://example.invalid/"}, {})])
                result = self.run_sync(provider)
                self.assertEqual(result["status"]["state"], "error")
                self.assertEqual(len(transport.calls), 1)
                self.assertTrue(transport.calls[0]["url"].startswith(BASE_URL + "?"))
                provider.close()

    def test_completed_refetch_removes_corrected_or_disappeared_records(self):
        provider, transport = self.provider([response([match("m1"), match("m2")])])
        first = self.run_sync(provider)
        self.clock.value += RECENT_TTL + 1
        changed = match("m1")
        changed["status"] = "Retired"
        # m1 explicitly corrected; m2 missing from a complete Finished refetch.
        transport.pages.append(response([changed]))
        second = self.run_sync(provider)
        self.assertEqual(second["records"], [])
        self.assertGreater(second["status"]["revision"], first["status"]["revision"])
        self.assertTrue(second["status"]["coverage_complete"])

    def test_manual_refresh_respects_fifteen_minute_cooldown(self):
        provider, transport = self.provider([response([match()])])
        self.run_sync(provider)
        provider.refresh(force=True)
        self.assertEqual(len(transport.calls), 1)
        self.clock.value += MANUAL_REFRESH_MINIMUM + 1
        provider.refresh()
        self.assertEqual(len(transport.calls), 1, "Normal refresh still uses six-hour freshness")
        transport.pages.append(response([match("updated")]))
        provider.refresh(force=True)
        provider._thread.join(timeout=3)
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(provider.snapshot()["records"][0]["match_id"], "sportstats:updated")

    def test_rate_limit_reset_delays_next_request(self):
        rows = [match("m" + str(index)) for index in range(100)]
        provider, transport = self.provider([
            response(rows, 0, 101, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "53"}),
            response([match("last")], 100, 101),
        ])
        result = self.run_sync(provider)
        self.assertEqual(result["status"]["state"], "ready")
        self.assertGreaterEqual(transport.calls[1]["time"] - transport.calls[0]["time"], 53)

    def test_query_warning_never_certifies_coverage(self):
        warned = response([match()])
        warned[2]["meta"]["warnings"] = ["Ignored from/to filter"]
        provider, _ = self.provider([warned])
        result = self.run_sync(provider)
        self.assertEqual(result["status"]["state"], "error")
        self.assertFalse(result["status"]["coverage_complete"])
        self.assertEqual(result["records"], [])

    def test_range_filter_violation_does_not_claim_coverage(self):
        outside = match(day="2025-12-20")
        provider, _ = self.provider([response([outside])])
        result = self.run_sync(provider)
        self.assertEqual(result["status"]["state"], "error")
        self.assertFalse(result["status"]["coverage_complete"])
        self.assertEqual(result["records"], [])

    def test_repeated_page_stops_and_resets_checkpoint(self):
        rows = [match("m" + str(index)) for index in range(100)]
        provider, transport = self.provider([response(rows, 0, 201), response(rows, 100, 201)])
        result = self.run_sync(provider)
        self.assertEqual(result["status"]["state"], "error")
        self.assertFalse(result["status"]["coverage_complete"])
        self.assertFalse(provider._checkpoints)
        self.assertEqual(len(transport.calls), 2)

    def test_only_one_worker_and_close_prevents_new_calls(self):
        started = threading.Event()
        release = threading.Event()
        calls = []

        def blocked(url, headers, timeout):
            calls.append(url)
            started.set()
            release.wait(timeout=2)
            return response([])

        provider = SportStatsResults(self.directory.name, api_key=TEST_CREDENTIAL, clock=self.clock,
                                    transport=blocked, cache_path=self.cache_path)
        self.instances.append(provider)
        provider.refresh()
        self.assertTrue(started.wait(timeout=1))
        first_thread = provider._thread
        provider.refresh()
        self.assertIs(provider._thread, first_thread)
        release.set()
        first_thread.join(timeout=2)
        self.assertEqual(len(calls), 1)
        provider.close()
        provider.refresh()
        self.assertEqual(len(calls), 1)

    def test_year_rollover_keeps_previous_year_results(self):
        clock = FakeClock("2025-12-31T12:00:00Z")
        provider, _ = self.provider([], clock=clock)
        previous = normalize_match(match(day="2025-12-20"), "2025-12-31T12:00:00Z")
        provider._records[previous["match_id"]] = previous
        provider._save_cache()
        provider.close()
        clock.value = datetime(2026, 1, 2, 12, tzinfo=timezone.utc).timestamp()
        new_provider, transport = self.provider([response([match("new", "2026-01-01")])], clock=clock)
        result = self.run_sync(new_provider)
        self.assertEqual(len(result["records"]), 2)
        self.assertTrue(result["status"]["coverage_complete"])
        self.assertEqual(parse_qs(urlsplit(transport.calls[0]["url"]).query)["date_from"], ["2026-01-01"])


if __name__ == "__main__":
    unittest.main()
