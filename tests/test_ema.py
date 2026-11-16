"""Offline tests for cutoff handling and nested primary-signal profiles."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd


spec = importlib.util.spec_from_file_location(
    "ema_judge", Path(__file__).resolve().parents[1] / "ema" / "ema20_judge.py")
ema = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ema)


class EMATests(unittest.TestCase):
    def tearDown(self):
        ema.apply_profile("standard")

    def test_primary_signal_profiles_are_nested_on_same_data(self):
        for low in (99.0, 101.0, 101.5, 102.5, 104.0):
            for volume_ratio in (0.50, 0.70, 0.85, 1.0):
                df = pd.DataFrame({
                    "close": np.full(30, 103.0), "ema20": np.full(30, 100.0),
                    "low": np.full(30, low), "high": np.full(30, 105.0),
                    "volume": np.full(30, volume_ratio * 100),
                    "vol_ma5": np.full(30, 100.0), "bias": np.full(30, 3.0),
                    "vol_ratio": np.full(30, volume_ratio),
                })
                matches = []
                for profile in ("strict", "standard", "relaxed"):
                    ema.apply_profile(profile)
                    matches.append(ema.primary_signals(df)["规则一 缩量回踩定买点"]["triggered"])
                self.assertLessEqual(matches[0], matches[1])
                self.assertLessEqual(matches[1], matches[2])

    def test_as_of_filters_future_rows_and_disables_realtime(self):
        source = pd.DataFrame({
            "日期": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "开盘": [10., 11., 12.], "收盘": [10., 11., 12.],
            "最高": [10., 11., 12.], "最低": [10., 11., 12.],
            "成交量": [100., 200., 300.], "成交额": [1000., 2200., 3600.],
        })
        fake = Mock()
        fake.stock_zh_a_hist.return_value = source
        with patch.object(ema, "ak", fake), patch.object(ema, "get_realtime_quote") as rt:
            result = ema.fetch_data("000001", days=60, realtime=True, as_of="2020-01-02")
        self.assertEqual(result["date"].max(), pd.Timestamp("2020-01-02"))
        self.assertEqual(len(result), 2)
        rt.assert_not_called()
        self.assertEqual(fake.stock_zh_a_hist.call_args.kwargs["end_date"], "20200102")

    def test_as_of_also_filters_fallback_source(self):
        fake = Mock()
        fake.stock_zh_a_hist.side_effect = RuntimeError("offline")
        fake.stock_zh_a_daily.return_value = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "open": [10., 11.], "close": [10., 11.], "high": [10., 11.],
            "low": [10., 11.], "volume": [100., 100.], "amount": [1000., 1100.],
        })
        with patch.object(ema, "ak", fake), patch.object(ema.time, "sleep"):
            result = ema.fetch_data("000001", as_of="2020-01-02")
        self.assertEqual(result["date"].max(), pd.Timestamp("2020-01-02"))

    def test_invalid_window_and_future_date_rejected(self):
        with patch.object(ema, "ak", Mock()):
            with self.assertRaises(ValueError):
                ema.fetch_data("000001", days=10)
            with self.assertRaises(ValueError):
                ema.fetch_data("000001", as_of="2999-01-01")


if __name__ == "__main__":
    unittest.main()
