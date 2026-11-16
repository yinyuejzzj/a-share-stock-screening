"""Offline tests across actual producer/consumer boundaries, without quote requests."""
import json
import hashlib
import sys
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tail_screen"))
from wei_pan_core import config, data_fetcher, hard_filter, orchestrator, scorer, sector_ranking


def candidate():
    return pd.DataFrame([dict(
        code="000001", name="fixture", close=10.1, preclose=10., open=10.,
        high=10.5, low=9.8, volume=1000, amount_yi=1., pct_chg=.04,
        market_cap_yi=80., vol_ratio=2., turnover_rate=.08,
        MA5=10.2, MA10=10.1, MA20=10., MA45=9.8,
        has_limit_up_in_N_day={1: False, 20: True, 30: True, 45: True},
        new_stock=False, is_st=False, is_halt=False, list_date="20000101",
    )])


def scoring(frame):
    return scorer.run_rule_scoring(frame, pct_lo=.02, pct_hi=.07,
                                  vr_lo=1.1, vr_hi=5., tr_lo=.05, tr_hi=.15,
                                  limit_up_N=30)["scored"]


class TailDataFlowTests(unittest.TestCase):
    def setUp(self):
        config.select_profile("standard")

    def test_new_schema_does_not_reuse_published_old_cache_keys(self):
        old_keywords = ("daily_snapshot", "2026-11-16", ["000001"])
        old_digest = hashlib.md5("|".join(str(k) for k in old_keywords).encode("utf-8")).hexdigest()
        self.assertNotEqual(data_fetcher._cache_key(old_keywords), old_digest)

    def test_scoring_survives_json_cache_roundtrip(self):
        cold = candidate()
        cached = pd.read_json(StringIO(cold.to_json(orient="split")), orient="split")
        self.assertEqual(scoring(cold).iloc[0]["rule_bp_dict"],
                         scoring(cached).iloc[0]["rule_bp_dict"])

    def test_sector_requests_a_real_history_window(self):
        calls = []

        def history(**kwargs):
            calls.append(kwargs)
            # Mimic a server honoring the supplied range.
            count = 1 if kwargs["start_date"] == kwargs["end_date"] else 6
            return pd.DataFrame({"收盘": [100., 101., 102., 103., 104., 105.][:count]})

        fake = SimpleNamespace(
            stock_board_industry_name_em=lambda: pd.DataFrame({"板块名称": ["hot"], "涨跌幅": [1.]}),
            stock_board_industry_hist_em=history,
        )
        with patch.dict(sys.modules, {"akshare": fake}):
            result = sector_ranking.fetch_sector_rankings("2026-11-16")
        self.assertIsNotNone(result)
        self.assertLess(calls[0]["start_date"], calls[0]["end_date"])
        self.assertAlmostEqual(result.iloc[0]["ret_3d_pct"], 105 / 102 - 1)
        self.assertAlmostEqual(result.iloc[0]["ret_5d_pct"], .05)

    def test_five_closes_compute_three_day_but_not_five_day_return(self):
        fake = SimpleNamespace(
            stock_board_industry_name_em=lambda: pd.DataFrame({"板块名称": ["hot"], "涨跌幅": [1.]}),
            stock_board_industry_hist_em=lambda **_: pd.DataFrame({"收盘": [100., 101., 102., 103., 104.]}),
        )
        with patch.dict(sys.modules, {"akshare": fake}):
            result = sector_ranking.fetch_sector_rankings("2026-11-16")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.iloc[0]["ret_3d_pct"], 104 / 101 - 1)
        self.assertTrue(pd.isna(result.iloc[0]["ret_5d_pct"]))

    def test_fetched_intraday_schema_is_accepted_by_filter(self):
        times = ["14:00:00", "14:05:00", "14:10:00", "14:15:00", "14:20:00", "14:25:00", "14:30:00"]
        prices = [10., 10.02, 10.04, 10.06, 10.08, 10.10, 10.12]
        payload = [dict(day="2026-11-16 " + t, open="10", close=str(p), high=str(p), low="10", volume="100")
                   for t, p in zip(times, prices)]
        with patch.object(data_fetcher, "_cache_read", return_value=None), \
             patch.object(data_fetcher, "_cache_write"), \
             patch.object(data_fetcher, "_new_session", return_value=SimpleNamespace(headers={})), \
             patch.object(data_fetcher, "_try_request_with_retries", return_value=(SimpleNamespace(text=json.dumps(payload)), None)):
            result = data_fetcher.fetch_intraday_minute("000001", "2026-11-16", "14:30")
        self.assertTrue(result["ok"])
        passed, reason = hard_filter._cond9_intraday_shape(result["data"], 1.2, mode="A", close_price=10.12)
        self.assertTrue(passed, reason)
        scored = scorer.run_rule_scoring(candidate(), pct_lo=.02, pct_hi=.07,
                                         vr_lo=1.1, vr_hi=5., tr_lo=.05, tr_hi=.15,
                                         limit_up_N=30, intraday_map={"000001": result["data"]})
        self.assertGreater(scored["scored"].iloc[0]["rule_bp_dict"]["minute_shape"], 2.5)

    def test_individual_stock_routes_do_not_use_index_overrides(self):
        self.assertEqual(data_fetcher._tencent_market_prefix("000001"), "sz000001")
        urls = []

        def request(_session, _method, url):
            urls.append(url)
            return SimpleNamespace(text=json.dumps([dict(day="2026-11-16", open="10", high="10", low="10", close="10", volume="100")])), None

        with patch.object(data_fetcher, "_cache_read", return_value=None), \
             patch.object(data_fetcher, "_cache_write"), \
             patch.object(data_fetcher, "_new_session", return_value=SimpleNamespace(headers={})), \
             patch.object(data_fetcher, "_try_request_with_retries", side_effect=request):
            data_fetcher._fetch_single_daily_k("000001")
        self.assertIn("symbol=sz000001", urls[0])

    def test_index_daily_route_is_explicit(self):
        urls = []

        def request(_session, _method, url):
            urls.append(url)
            return SimpleNamespace(text=json.dumps([dict(day="2026-11-16", open="10", high="10", low="10", close="10", volume="100")])), None

        with patch.object(data_fetcher, "_cache_read", return_value=None), \
             patch.object(data_fetcher, "_cache_write"), \
             patch.object(data_fetcher, "_new_session", return_value=SimpleNamespace(headers={})), \
             patch.object(data_fetcher, "_try_request_with_retries", side_effect=request):
            data_fetcher._fetch_single_daily_k("000300", is_index=True)
        self.assertIn("symbol=sh000300", urls[0])

    def test_failed_universe_lookup_does_not_invent_a_fixed_stock_list(self):
        fake = SimpleNamespace(stock_info_a_code_name=lambda: pd.DataFrame(columns=["code", "name"]))
        with patch.dict(sys.modules, {"akshare": fake}), \
             patch.object(data_fetcher, "_new_session", return_value=SimpleNamespace(headers={})), \
             patch.object(data_fetcher, "_try_request_with_retries", return_value=(None, "offline")):
            codes, info = data_fetcher._fetch_tencent_code_list_and_info()
        self.assertEqual(codes, [])
        self.assertEqual(info, {})

    def test_primary_hs300_source_accepts_actual_akshare_english_columns(self):
        fake = SimpleNamespace(stock_zh_index_daily_em=lambda **_: pd.DataFrame({"close": [100., 101., 102., 103., 104., 105.]}))
        with patch.dict(sys.modules, {"akshare": fake}), \
             patch.object(data_fetcher, "_cache_read", return_value=None), \
             patch.object(data_fetcher, "_cache_write"), \
             patch.object(data_fetcher, "_fetch_single_daily_k", return_value=None) as fallback, \
             patch.object(data_fetcher, "_fetch_sina_all_spot", return_value=pd.DataFrame()), \
             patch.object(data_fetcher, "_new_session", return_value=SimpleNamespace(headers={})), \
             patch.object(data_fetcher, "_try_request_with_retries", return_value=(None, "offline")):
            result = data_fetcher.fetch_hs300_regime_signals("2026-11-16")
        self.assertAlmostEqual(result["data"]["hs300_5d_return"], .05)
        fallback.assert_not_called()

    def test_old_intraday_cache_schema_is_restored(self):
        frame = pd.DataFrame({"time_str": ["14:00", "14:05", "14:10", "14:15", "14:20", "14:25"],
                              "open": [10.] * 6, "close": [10., 10.02, 10.04, 10.06, 10.08, 10.1],
                              "high": [10., 10.02, 10.04, 10.06, 10.08, 10.1], "low": [10.] * 6})
        with patch.object(data_fetcher, "_cache_read", return_value=frame.to_json(orient="split")):
            result = data_fetcher.fetch_intraday_minute("000001", "2026-11-16", "14:30")
        passed, reason = hard_filter._cond9_intraday_shape(result["data"], 1.2, mode="A", close_price=10.1)
        self.assertTrue(passed, reason)

    def test_sector_mapping_reaches_scorer_in_actual_pipeline(self):
        rows = candidate()
        gate = dict(gate_action="normal", data_incomplete=False, global_tightness_factor=1., loosen_limitup_extra_days=0)
        regime = dict(regime="neutral", bull_hits=0, weak_hits=0, ml_weight=0.)
        intraday = pd.DataFrame({"time": ["2026-11-16 " + t for t in ["14:00", "14:10", "14:20", "14:30", "14:44"]],
                                 "close": [10., 10.02, 10.04, 10.05, 10.06],
                                 "high": [10., 10.02, 10.04, 10.05, 10.06], "low": [10.] * 5})
        ranks = pd.DataFrame([dict(sector_name="hot", ret_3d_pct=.1, ret_5d_pct=.18, count_stocks=50)])
        with patch.object(orchestrator, "compute_weighted_gate", return_value=gate), \
             patch.object(orchestrator, "detect_regime", return_value=regime), \
             patch.object(orchestrator, "fetch_all_stocks_daily_snapshot", return_value={"ok": True, "data": rows}), \
             patch.object(orchestrator, "build_candidate_pool", return_value={"ok": True, "data": rows, "stats": {}}), \
             patch.object(data_fetcher, "fetch_intraday_minute", return_value={"ok": True, "data": intraday}), \
             patch.object(orchestrator, "fetch_sector_rankings", return_value=ranks), \
             patch.object(orchestrator, "fetch_stock_sector_map", return_value={"000001": "hot"}):
            result = orchestrator.run_wei_pan(mode="B")
        self.assertEqual(result["final_action"], "SELECTED")
        self.assertGreater(result["final_top5"].iloc[0]["rule_sector_30"], 25.)


if __name__ == "__main__":
    unittest.main()
