"""Synthetic checks only; no market-data downloads or claimed trading performance."""
import unittest
import numpy as np
import pandas as pd

from .cli import FEATURES, PROFILES, evaluate_and_predict, normalize_data, parser
from .indicators import add_all_indicators, add_label


def fixture(rows=650):
    rng = np.random.default_rng(42)
    close = 30 * np.exp(np.cumsum(rng.normal(0, 0.018, rows)))
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=rows),
        "open": close * 0.997, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.integers(10000, 100000, rows), "amount": close * 10000})


class OfflineTests(unittest.TestCase):
    def test_unknown_future_labels_remain_missing(self):
        labeled = add_label(fixture(), forward_days=5)
        self.assertTrue(labeled["label"].iloc[-5:].isna().all())
        self.assertTrue(labeled["label"].iloc[:-5].notna().all())

    def test_features_do_not_change_when_future_is_appended(self):
        frame = fixture()
        prefix = add_all_indicators(frame.iloc[:400])
        full = add_all_indicators(frame)
        columns = [column for column in FEATURES if column in full]
        pd.testing.assert_frame_equal(prefix[columns], full[columns].iloc[:400])

    def test_cutoff_drops_future_bars(self):
        frame = fixture()
        result = normalize_data(frame, "2021-01-01", 1500)
        self.assertLessEqual(result["date"].max(), pd.Timestamp("2021-01-01"))

    def test_optional_amount_accepts_missing_markers(self):
        frame = fixture()
        frame["amount"] = "--"
        normalized = normalize_data(frame, "2023-01-01", 1500)
        self.assertTrue(normalized["amount"].eq(0).all())
        self.assertIn("vol_ratio", add_all_indicators(normalized))

    def test_model_reaches_latest_unlabeled_row(self):
        frame = normalize_data(fixture(), "2023-01-01", 1500)
        result = evaluate_and_predict(frame, forward_days=3)
        self.assertEqual(result["as_of"], str(frame["date"].max().date()))
        self.assertGreaterEqual(result["model_probability"], 0)
        self.assertLessEqual(result["model_probability"], 1)

    def test_profile_nesting_and_cli(self):
        self.assertGreater(PROFILES["strict"], PROFILES["standard"])
        self.assertGreater(PROFILES["standard"], PROFILES["relaxed"])
        args = parser().parse_args(["--code", "000001", "--profile", "strict", "--forward-days", "3"])
        self.assertEqual(args.forward_days, 3)


if __name__ == "__main__":
    unittest.main()
