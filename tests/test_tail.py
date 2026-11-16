import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tail_screen"))
from wei_pan_core import config, data_fetcher, hard_filter
from wei_pan_core import orchestrator


def candidate():
    return pd.DataFrame([dict(
        code="600000", pct_chg=.04, market_cap_yi=80., vol_ratio=2.,
        turnover_rate=.08, has_limit_up_in_N_day={30: False, 45: True},
        close=10.1, MA5=10.2, MA10=10.1, MA20=10., MA45=9.8,
        preclose=10., high=10.5, low=9.8, new_stock=False,
    )])


class TailRegressionTests(unittest.TestCase):
    def setUp(self):
        config.select_profile("standard")

    def tearDown(self):
        config.select_profile("standard")

    def test_longer_window_hit_cannot_override_exact_false(self):
        _, report = hard_filter.run_hard_filter(candidate(), mode="B", skip_cond9=True)
        self.assertEqual(report.dropped_by_cond.get("cond5_gene"), 1)

    def test_zero_centered_ma_band_tightens_on_both_sides(self):
        a = hard_filter._apply_thresholds_for_mode("A", 1., 0)
        b = hard_filter._apply_thresholds_for_mode("B", 1., 0)
        self.assertGreater(a["dev_lo"], b["dev_lo"])
        self.assertLess(a["dev_hi"], b["dev_hi"])

    def test_missing_intraday_never_passes(self):
        for mode in "ABC":
            ok, _ = hard_filter._cond9_intraday_shape(None, 1., mode=mode, close_price=10.)
            self.assertFalse(ok)

    def test_augment_computes_exact_window_hits(self):
        # Last session jumps 10%; it must be seen in even the one-session window.
        closes = np.r_[np.full(60, 10.), 11.]
        history = pd.DataFrame({"close": closes})
        with patch.object(data_fetcher, "_fetch_single_daily_k", return_value=history):
            out = data_fetcher._augment_with_historical_columns(pd.DataFrame({"code": ["600000"]}), "2026-11-16")
        self.assertTrue(out.iloc[0]["has_limit_up_in_N_day"][30])
        self.assertTrue(out.iloc[0]["has_limit_up_in_N_day"].get(1))

    def test_all_intraday_requests_failing_does_not_skip_second_phase(self):
        rows = candidate()
        rows.at[0, "has_limit_up_in_N_day"] = {30: True}
        gate = {"gate_action": "normal", "data_incomplete": False,
                "global_tightness_factor": 1., "loosen_limitup_extra_days": 0}
        regime = {"regime": "neutral", "bull_hits": 0, "weak_hits": 0, "ml_weight": 0.}
        with patch.object(orchestrator, "compute_weighted_gate", return_value=gate), \
             patch.object(orchestrator, "detect_regime", return_value=regime), \
             patch.object(orchestrator, "fetch_all_stocks_daily_snapshot", return_value={"ok": True, "data": rows}), \
             patch.object(orchestrator, "build_candidate_pool", return_value={"ok": True, "data": rows, "stats": {}}), \
             patch.object(data_fetcher, "fetch_intraday_minute", return_value={"ok": False}), \
             patch.object(orchestrator, "run_rule_scoring") as score:
            result = orchestrator.run_wei_pan(mode="B")
        self.assertEqual(result["final_action"], "NO_PASS")
        score.assert_not_called()

    def test_profiles_tighten_numeric_thresholds_under_same_mode(self):
        effective = {}
        for profile in ("strict", "standard", "relaxed"):
            config.select_profile(profile)
            effective[profile] = hard_filter._apply_thresholds_for_mode("B", 1., 0)
        for tighter, looser in (("strict", "standard"), ("standard", "relaxed")):
            a, b = effective[tighter], effective[looser]
            for field in ("pct_lo", "cap_lo", "vr_lo", "tr_lo", "dev_lo", "amplitude_min"):
                self.assertGreaterEqual(a[field], b[field])
            for field in ("pct_hi", "cap_hi", "vr_hi", "tr_hi", "dev_hi", "limit_up_N"):
                self.assertLessEqual(a[field], b[field])

    def test_offline_pipeline_reaches_report_candidates(self):
        rows = candidate()
        rows.at[0, "has_limit_up_in_N_day"] = {30: True}
        gate = {"gate_action": "normal", "data_incomplete": False,
                "global_tightness_factor": 1., "loosen_limitup_extra_days": 0}
        regime = {"regime": "neutral", "bull_hits": 0, "weak_hits": 0, "ml_weight": 0.}
        intraday = pd.DataFrame({
            "time": ["2026-11-16 14:00", "2026-11-16 14:10", "2026-11-16 14:20", "2026-11-16 14:30", "2026-11-16 14:44"],
            "close": [10., 10.02, 10.04, 10.05, 10.06],
            "high": [10., 10.02, 10.04, 10.05, 10.06], "low": [10.] * 5,
        })
        with patch.object(orchestrator, "compute_weighted_gate", return_value=gate), \
             patch.object(orchestrator, "detect_regime", return_value=regime), \
             patch.object(orchestrator, "fetch_all_stocks_daily_snapshot", return_value={"ok": True, "data": rows}), \
             patch.object(orchestrator, "build_candidate_pool", return_value={"ok": True, "data": rows, "stats": {}}), \
             patch.object(data_fetcher, "fetch_intraday_minute", return_value={"ok": True, "data": intraday}), \
             patch.object(orchestrator, "fetch_sector_rankings", return_value=None), \
             patch.object(orchestrator, "fetch_stock_sector_map", return_value={}):
            result = orchestrator.run_wei_pan(mode="B")
        self.assertEqual(result["final_action"], "SELECTED")
        self.assertEqual(len(result["final_top5"]), 1)
        self.assertFalse(result["ml_meta"]["ml_used"])


if __name__ == "__main__":
    unittest.main()
