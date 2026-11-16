"""Offline tests for cutoff handling and nested primary-signal profiles."""
import importlib.util
from pathlib import Path
import tempfile
import sys
import io
from contextlib import redirect_stdout
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
        self.assertEqual(fake.stock_zh_a_hist.call_args.kwargs["adjust"], "qfq")

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

    def market_frame(self):
        return pd.DataFrame({"date": pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=40),
            "open": 10., "high": 10.1, "low": 9.9, "close": 10.,
            "volume": 100., "amount": 100000.})

    def test_default_does_not_fetch_realtime(self):
        fake = Mock()
        fake.stock_zh_a_hist.return_value = self.market_frame().iloc[:-1]
        with patch.object(ema, "ak", fake), patch.object(ema, "get_realtime_quote") as rt:
            ema.fetch_data("000001")
        rt.assert_not_called()

    def test_malformed_daily_values_and_duplicates_are_rejected(self):
        for column, value in (("volume", -1), ("close", np.inf), ("low", 11)):
            frame = self.market_frame()
            frame.loc[frame.index[-1], column] = value
            fake = Mock()
            fake.stock_zh_a_hist.return_value = frame
            with self.subTest(column=column), patch.object(ema, "ak", fake):
                with self.assertRaises((ValueError, RuntimeError)):
                    ema.fetch_data("000001", realtime=False)
        frame = self.market_frame()
        fake.stock_zh_a_hist.return_value = pd.concat([frame, frame.tail(1)])
        with patch.object(ema, "ak", fake), self.assertRaises((ValueError, RuntimeError)):
            ema.fetch_data("000001", realtime=False)

    def test_stale_realtime_is_rejected_and_same_day_not_duplicated(self):
        frame = self.market_frame()
        fake = Mock()
        fake.stock_zh_a_hist.return_value = frame
        rt = {"date": str(pd.Timestamp.today().date()), "name": "synthetic", "open": 10.,
              "price": 10., "high": 10.1, "low": 9.9, "volume": 100., "amount": 100000.,
              "pct_change": 0.}
        with patch.object(ema, "ak", fake), patch.object(ema, "get_realtime_quote", return_value=rt):
            result = ema.fetch_data("000001", realtime=True)
        self.assertEqual(result["date"].duplicated().sum(), 0)
        self.assertEqual(ema.compute_volume_features(result).iloc[-1]["vol_ratio"], 1.)
        self.assertEqual(fake.stock_zh_a_hist.call_args.kwargs["adjust"], "")
        for delta in (-3, 1):
            rt["date"] = str((pd.Timestamp.today() + pd.Timedelta(days=delta)).date())
            with patch.object(ema, "ak", fake), patch.object(ema, "get_realtime_quote", return_value=rt):
                with self.assertRaises((ValueError, RuntimeError)):
                    ema.fetch_data("000001", realtime=True)

    def test_sina_realtime_shares_converted_to_hands(self):
        fields = ["synthetic", "10", "10", "10", "10.1", "9.9", "0", "0", "10000", "100000"] + ["0"] * 20
        fields += [str(pd.Timestamp.today().date()), "14:55:00"]
        response = Mock(text='var hq_str_sz000001="' + ",".join(fields) + '";')
        with patch.object(ema.requests, "get", return_value=response):
            quote = ema.get_realtime_quote("000001")
        self.assertEqual(quote["volume"], 100.)

    def test_sina_daily_shares_converted_to_hands(self):
        fake = Mock()
        fake.stock_zh_a_hist.return_value = pd.DataFrame()
        frame = self.market_frame()
        frame["volume"] = 10000.
        fake.stock_zh_a_daily.return_value = frame
        with patch.object(ema, "ak", fake):
            result = ema.fetch_data("000001", realtime=False)
        self.assertEqual(result.iloc[-1]["volume"], 100.)

    def test_report_creates_requested_output_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "new-directory" / "report.md"
            with patch.object(ema, "fetch_data", return_value=self.market_frame()), \
                 patch.object(sys, "argv", ["ema20_judge.py", "000001", "--out", str(path)]), \
                 redirect_stdout(io.StringIO()):
                ema.main()
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
