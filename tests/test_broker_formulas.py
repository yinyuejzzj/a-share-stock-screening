"""Offline semantic checks; these do not compile formulas in a broker client."""
import importlib.util
from pathlib import Path
import re
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = load("broker_reference", ROOT / "broker_formulas/reference.py")
ema = load("broker_ema", ROOT / "ema/ema20_judge.py")
gap = load("broker_gap", ROOT / "gap_screen/screen_gapup.py")


def frame(rows=100):
    rng = np.random.default_rng(27)
    close = 10 * np.exp(np.cumsum(rng.normal(0, .018, rows)))
    opening = close * .999
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=rows),
        "open": opening, "close": close, "high": close * 1.012, "low": close * .988,
        "volume": rng.integers(300, 1000, rows).astype(float)})


class BrokerFormulaTests(unittest.TestCase):
    def tearDown(self):
        ema.apply_profile("standard")

    def test_each_tdx_has_one_output_and_no_future_or_unverified_fields(self):
        formulas = list((ROOT / "broker_formulas").glob("*.tdx"))
        self.assertEqual(len(formulas), 3)
        for path in formulas:
            source = re.sub(r"\{[^}]*\}", "", path.read_text(encoding="utf-8"))
            self.assertEqual(re.findall(r"\b(\w+)\s*:(?!=)", source), ["XG"])
            self.assertNotRegex(source.upper(), r"\b(REFX|REFXV|BACKSET|ZIG|PEAK|TROUGH|BARSNEXT|FINANCE|DYNAINFO)\s*\(")
            self.assertNotRegex(source, r"REF\([^;]*,\s*-\d")
            self.assertIn("PROFILE:=2;", source)

    def test_ema_primary_matches_existing_python_on_same_history(self):
        data = frame()
        for name in ref.PROFILES:
            ema.apply_profile(name)
            result = ref.evaluate(data, "ema", name)
            for end in range(25, len(data) + 1):
                prefix = data.iloc[:end]
                enriched = ema.compute_volume_features(ema.compute_macd(ema.compute_ema20(prefix)))
                original = ema.primary_signals(enriched)["规则一 缩量回踩定买点"]["triggered"]
                self.assertEqual(bool(result.iloc[end - 1]["selected"]), original)

    def test_gap_matches_existing_core_and_strict_percent_unit(self):
        data = pd.DataFrame([
            {"date": "2020-01-03", "open": 9.8, "high": 10., "low": 9.5, "close": 9.9, "volume": 100.},
            {"date": "2020-01-06", "open": 10.04, "high": 10.2, "low": 10., "close": 10.1, "volume": 100.},
        ])
        for name, parameters in ref.PROFILES.items():
            actual = bool(ref.evaluate(data, "gap", name).iloc[-1]["selected"])
            existing = gap.gap_values(data, "2020-01-06", parameters["gap_pct"]) is not None
            self.assertEqual(actual, existing)
        self.assertFalse(ref.evaluate(data, "gap", "strict").iloc[-1]["selected"])
        self.assertTrue(ref.evaluate(data, "gap", "standard").iloc[-1]["selected"])
        # Exactly 0.5% above prior high is excluded by the strict > comparison.
        data.loc[1, "open"] = 10. * (1 + .5 / 100)
        self.assertFalse(ref.evaluate(data, "gap", "strict").iloc[-1]["selected"])

    def test_future_append_does_not_rewrite_past_results_and_as_of(self):
        data = frame(120)
        for kind in ("ema", "gap", "tail"):
            full = ref.evaluate(data, kind)
            prefix = ref.evaluate(data.iloc[:100], kind)
            pd.testing.assert_series_equal(prefix["selected"], full["selected"].iloc[:100])
            cut = ref.evaluate(data, kind, as_of=str(data.iloc[99]["date"].date()))
            pd.testing.assert_frame_equal(cut, prefix)

    def test_profiles_are_nested_on_identical_data(self):
        data = frame(300)
        for kind in ("ema", "gap", "tail"):
            sets = [set(ref.evaluate(data, kind, name).query("selected").index)
                    for name in ("strict", "standard", "relaxed")]
            self.assertTrue(sets[0] <= sets[1] <= sets[2])

    def test_tail_uses_subtract_then_divide_at_return_endpoints(self):
        data = frame()
        data["close"] = 10.
        data["open"] = 10.
        data["high"] = 10.8
        data["low"] = 9.8
        for closing in (10.25, 10.6):
            data.loc[data.index[-1], "close"] = closing
            result = ref.evaluate(data, "tail", "strict")
            ret = result.iloc[-1]["return_1d"]
            self.assertGreaterEqual(ret, .025)
            self.assertLessEqual(ret, .06)

    def test_warmup_and_zero_volume_do_not_signal(self):
        data = frame()
        for kind, minimum in (("ema", 25), ("gap", 2), ("tail", 60)):
            self.assertFalse(ref.evaluate(data.iloc[:minimum - 1], kind)["selected"].any())
            no_volume = data.copy()
            no_volume.loc[no_volume.index[-1], "volume"] = 0
            self.assertFalse(ref.evaluate(no_volume, kind).iloc[-1]["selected"])

    def test_malformed_or_multi_security_daily_csv_rejected(self):
        data = frame()
        data["code"] = "000001"
        data.loc[0, "code"] = "000002"
        with self.assertRaises(ValueError):
            ref.evaluate(data, "ema")
        data = frame()
        data.loc[0, "volume"] = -1
        with self.assertRaises(ValueError):
            ref.evaluate(data, "gap")

    def test_timezone_and_symbol_alias_mixed_securities_rejected(self):
        for alias in ("symbol", "股票代码"):
            data = frame()
            data[alias] = "000001"
            data.loc[0, alias] = "000002"
            with self.assertRaisesRegex(ValueError, "一只股票"):
                ref.evaluate(data, "ema")
        data = frame()
        data["date"] = data["date"].dt.tz_localize("Asia/Shanghai")
        with self.assertRaisesRegex(ValueError, "时区"):
            ref.evaluate(data, "gap")

    def test_insufficient_history_is_explicit(self):
        result = ref.evaluate(frame(24), "ema")
        self.assertFalse(result.iloc[-1]["history_ready"])
        self.assertFalse(result.iloc[-1]["selected"])


if __name__ == "__main__":
    unittest.main()
