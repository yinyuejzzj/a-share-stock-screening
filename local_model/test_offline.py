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
    def test_large_horizon_cannot_wrap_purge_slice(self):
        frame = fixture(1500)
        close = 30 + 5 * np.sin(np.arange(len(frame)) * 2 * np.pi / 500)
        frame['close'] = close
        frame['open'], frame['high'], frame['low'] = close * .997, close * 1.01, close * .99
        with self.assertRaisesRegex(ValueError, '训练|步长'):
            evaluate_and_predict(frame, forward_days=650)

    def test_rejects_nonfinite_and_inconsistent_ohlc(self):
        for column, value in [('close', np.inf), ('volume', np.inf), ('high', .01)]:
            with self.subTest(column=column):
                frame = fixture()
                frame.loc[300, column] = value
                with self.assertRaises(ValueError):
                    normalize_data(frame, '2023-01-01', 1500)

    def test_csv_symbol_must_match_requested_code(self):
        frame = fixture()
        frame['code'] = '600519'
        with self.assertRaisesRegex(ValueError, '代码'):
            normalize_data(frame, '2023-01-01', 1500, expected_code='000001')
        frame['code'] = '000001'
        frame.loc[300, 'code'] = '600519'
        with self.assertRaisesRegex(ValueError, '代码'):
            normalize_data(frame, '2023-01-01', 1500, expected_code='000001')

    def test_dates_reject_missing_and_timezone_with_clear_error(self):
        frame = fixture()
        frame.loc[300, 'date'] = pd.NaT
        with self.assertRaisesRegex(ValueError, '日期'):
            normalize_data(frame, '2023-01-01', 1500)
        frame = fixture()
        frame['date'] = frame['date'].dt.tz_localize('UTC')
        with self.assertRaisesRegex(ValueError, '时区'):
            normalize_data(frame, '2023-01-01', 1500)

    def test_experimental_scaler_matches_actual_prediction_input(self):
        from .experimental.predictor import StockPredictor
        data = add_label(add_all_indicators(fixture()), forward_days=1)
        predictor = StockPredictor(forward_days=1, model_type='rf')
        predictor.feature_cols = ['rsi']
        a, b, c, ya, yb, yc = predictor.prepare_data(data)
        original_mean, original_scale = predictor.scaler.mean_.copy(), predictor.scaler.scale_.copy()
        predictor._select_and_rescale(a, b, c, ya)
        class CaptureModel:
            def predict_proba(self, values):
                self.input = np.asarray(values)
                return np.array([[.4, .6]])
            def predict(self, values):
                return np.array([1])
        predictor.model = CaptureModel()
        predictor.predict_latest(data)
        expected = (data[['rsi']].iloc[-1].values - original_mean) / original_scale
        np.testing.assert_allclose(predictor.model.input[0], expected)

    def test_experimental_scaler_tracks_selected_feature_columns(self):
        from .experimental.predictor import StockPredictor
        rng = np.random.default_rng(7)
        data = fixture(400)
        names = [f'feature_{number}' for number in range(35)]
        for number, name in enumerate(names):
            data[name] = rng.normal(number * 10, number + 1, len(data))
        data['label'] = rng.integers(0, 2, len(data))
        predictor = StockPredictor(forward_days=1, model_type='rf')
        predictor.feature_cols = names.copy()
        a, b, c, ya, yb, yc = predictor.prepare_data(data)
        means, scales = predictor.scaler.mean_.copy(), predictor.scaler.scale_.copy()
        selected_train, _, _ = predictor._select_and_rescale(a, b, c, ya)
        positions = [names.index(name) for name in predictor.feature_cols]
        np.testing.assert_allclose(predictor.scaler.mean_, means[positions])
        np.testing.assert_allclose(predictor.scaler.scale_, scales[positions])
        np.testing.assert_allclose(selected_train, a[:, positions])
        actual = predictor.scaler.transform(data[predictor.feature_cols].iloc[[-1]])
        expected = (data[predictor.feature_cols].iloc[-1].values - means[positions]) / scales[positions]
        np.testing.assert_allclose(actual[0], expected)

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
