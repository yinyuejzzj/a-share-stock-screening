import importlib.util
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("screen_gapup", ROOT / "gap_screen/screen_gapup.py")
gap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gap)


class GapTests(unittest.TestCase):
    def setUp(self):
        # 合成数据：周五到周一，验证上一交易日而非上一自然日。
        self.bars = pd.DataFrame([
            {"date": "2026-09-25", "open": 9.5, "high": 10, "close": 9.8},
            {"date": "2026-09-28", "open": 10.1, "high": 10.5, "close": 10.2},
            {"date": "2026-09-29", "open": 99, "high": 100, "close": 99},
        ])

    def test_previous_trading_day_and_future_rows_ignored(self):
        match = gap.gap_values(self.bars, "2026-09-28")
        self.assertEqual(match["previous_date"], "2026-09-25")
        self.assertAlmostEqual(match["gap_above_high_pct"], 1)

    def test_no_silent_date_fallback(self):
        self.assertIsNone(gap.gap_values(self.bars, "2026-09-27"))

    def test_strict_gap_is_not_simple_high_open(self):
        self.bars.loc[1, "open"] = 9.9  # 高于昨收9.8，低于昨高10
        self.assertIsNone(gap.gap_values(self.bars, "2026-09-28"))

    def test_minimum_gap_threshold(self):
        self.assertIsNone(gap.gap_values(self.bars, "2026-09-28", min_gap_pct=2))

    def test_infinite_prices_do_not_create_signals(self):
        self.bars.loc[1, "open"] = float("inf")
        self.assertIsNone(gap.gap_values(self.bars, "2026-09-28"))

    def test_snapshot_date_must_match(self):
        snapshot = pd.DataFrame([{"code": "000001", "name": "示例", "market_cap": 5e9,
                                  "pb": 2, "volume": 100, "snapshot_date": "2026-09-25"}])
        with self.assertRaisesRegex(ValueError, "快照日期"):
            gap.filter_snapshot(snapshot, "2026-09-28", gap.PROFILES["standard"])

    def test_profiles_nested_and_unsafe_names_rejected(self):
        snapshot = pd.DataFrame([
            {"code": str(i).zfill(6), "name": name, "market_cap": cap * 1e8,
             "pb": pb, "volume": vol, "snapshot_date": "2026-09-28"}
            for i, (name, cap, pb, vol) in enumerate([
                ("样本甲", 40, 2, 100), ("样本乙", 25, 8, 100),
                ("样本丙", 150, 12, 100), ("ST样本", 40, 2, 100), ("停牌样本", 40, 2, 0)])])
        sets = [set(gap.filter_snapshot(snapshot, "2026-09-28", gap.PROFILES[p])["code"])
                for p in ("strict", "standard", "relaxed")]
        self.assertTrue(sets[0] < sets[1] < sets[2])
        self.assertNotIn("000003", sets[2])
        self.assertNotIn("000004", sets[2])


if __name__ == "__main__":
    unittest.main()
