import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.predict import load_saved_model, predict_latest_price
from src.return_calibration import ReturnCalibrator


class DummyModel:
    def __init__(self, predictions):
        self.predictions = np.array(predictions)

    def predict(self, X):
        return self.predictions[: len(X)]


class SeriesCalibrator:
    def predict(self, raw_returns):
        return pd.Series([0.10, 0.20])


class ShortCalibrator:
    def predict(self, raw_returns):
        return np.array([0.10])


class Model1PredictTests(unittest.TestCase):
    def test_load_saved_model_can_return_metadata_for_new_artifacts(self):
        artifact = {
            "model": DummyModel([0.01]),
            "features": ["feature"],
            "horizon": 5,
            "target_type": "future_return",
            "return_calibrator": ReturnCalibrator(slope=0.5),
        }

        with patch("src.predict.joblib.load", return_value=artifact) as load:
            model, features, metadata = load_saved_model(
                "model.pkl",
                include_metadata=True,
            )

        load.assert_called_once_with("model.pkl")
        self.assertIs(artifact["model"], model)
        self.assertEqual(["feature"], features)
        self.assertEqual("future_return", metadata["target_type"])
        self.assertIs(artifact["return_calibrator"], metadata["return_calibrator"])

    def test_load_saved_model_keeps_old_return_tuple_by_default(self):
        artifact = {
            "model": DummyModel([0.01]),
            "features": ["feature"],
            "horizon": 5,
            "target_type": "future_return",
            "return_calibrator": ReturnCalibrator(slope=0.5),
        }

        with patch("src.predict.joblib.load", return_value=artifact):
            result = load_saved_model("model.pkl")

        self.assertEqual(2, len(result))
        self.assertIs(artifact["model"], result[0])
        self.assertEqual(["feature"], result[1])

    def test_predict_latest_price_uses_return_target_artifact_metadata(self):
        df = pd.DataFrame(
            {
                "trading_date": ["2024-01-01", "2024-01-02"],
                "symbol": ["AAA", "AAA"],
                "close": [100.0, 120.0],
                "feature": [1.0, 2.0],
            }
        )
        model = DummyModel([0.20])
        calibrator = ReturnCalibrator(slope=0.5, intercept=0.0, min_abs_signal=0.0)

        latest_df = predict_latest_price(
            df=df,
            model=model,
            features=["feature"],
            target_type="future_return",
            return_calibrator=calibrator,
        )

        self.assertAlmostEqual(0.20, latest_df.iloc[0]["raw_predicted_return"])
        self.assertAlmostEqual(0.10, latest_df.iloc[0]["predicted_return"])
        self.assertAlmostEqual(132.0, latest_df.iloc[0]["predicted_close"])

    def test_predict_latest_price_keeps_price_target_compatibility(self):
        df = pd.DataFrame(
            {
                "trading_date": ["2024-01-01"],
                "symbol": ["AAA"],
                "close": [100.0],
                "feature": [1.0],
            }
        )
        model = DummyModel([110.0])

        latest_df = predict_latest_price(
            df=df,
            model=model,
            features=["feature"],
        )

        self.assertAlmostEqual(110.0, latest_df.iloc[0]["predicted_close"])
        self.assertAlmostEqual(0.10, latest_df.iloc[0]["predicted_return"])

    def test_predict_latest_price_assigns_series_calibration_positionally(self):
        df = pd.DataFrame(
            {
                "trading_date": [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                ],
                "symbol": ["BBB", "AAA", "BBB", "AAA"],
                "close": [50.0, 100.0, 80.0, 120.0],
                "feature": [1.0, 2.0, 3.0, 4.0],
            },
            index=[7, 3, 11, 5],
        )
        model = DummyModel([0.30, 0.40])

        latest_df = predict_latest_price(
            df=df,
            model=model,
            features=["feature"],
            target_type="future_return",
            return_calibrator=SeriesCalibrator(),
        )

        self.assertEqual([5, 11], latest_df.index.tolist())
        self.assertFalse(latest_df["predicted_return"].isna().any())
        self.assertAlmostEqual(0.10, latest_df.loc[5, "predicted_return"])
        self.assertAlmostEqual(132.0, latest_df.loc[5, "predicted_close"])
        self.assertAlmostEqual(0.20, latest_df.loc[11, "predicted_return"])
        self.assertAlmostEqual(96.0, latest_df.loc[11, "predicted_close"])

    def test_predict_latest_price_rejects_mismatched_calibrated_prediction_length(self):
        df = pd.DataFrame(
            {
                "trading_date": ["2024-01-01", "2024-01-01"],
                "symbol": ["AAA", "BBB"],
                "close": [100.0, 80.0],
                "feature": [1.0, 2.0],
            }
        )
        model = DummyModel([0.30, 0.40])

        with self.assertRaisesRegex(
            ValueError,
            "calibrated predictions must match latest rows",
        ):
            predict_latest_price(
                df=df,
                model=model,
                features=["feature"],
                target_type="future_return",
                return_calibrator=ShortCalibrator(),
            )


if __name__ == "__main__":
    unittest.main()
