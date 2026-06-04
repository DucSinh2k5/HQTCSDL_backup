import unittest

import pandas as pd

from src.config import DATA_PATH, FEATURES


def project_relative(path):
    from src import config

    return path.relative_to(config.PROJECT_ROOT).as_posix()


class ConfigFeatureTests(unittest.TestCase):
    @unittest.skipUnless(DATA_PATH.exists(), "features_all.csv is not available")
    def test_configured_features_exist_in_raw_dataset(self):
        columns = set(pd.read_csv(DATA_PATH, nrows=0).columns.str.strip())

        missing_features = [feature for feature in FEATURES if feature not in columns]

        self.assertEqual([], missing_features)

    def test_backtest_config_is_available(self):
        from src import config

        self.assertEqual(5, config.HORIZON)
        self.assertEqual("reports/backtest.csv", project_relative(config.BACKTEST_PATH))
        self.assertEqual(
            "reports/backtest_metrics.json",
            project_relative(config.BACKTEST_METRICS_PATH),
        )
        self.assertEqual(
            "reports/backtest_sweep.csv",
            project_relative(config.BACKTEST_SWEEP_PATH),
        )
        self.assertGreater(config.BACKTEST_TOP_K, 0)
        self.assertGreater(config.BACKTEST_MIN_VOLUME, 0)
        self.assertGreater(config.BACKTEST_MIN_CLOSE, 0)
        self.assertGreaterEqual(config.BACKTEST_MIN_PREDICTED_RETURN, 0)
        self.assertGreaterEqual(config.TRANSACTION_COST_RATE, 0)
        self.assertGreaterEqual(config.SLIPPAGE_RATE, 0)
        self.assertIn(config.BACKTEST_TOP_K, config.BACKTEST_TOP_K_VALUES)
        self.assertIn(config.BACKTEST_MIN_VOLUME, config.BACKTEST_MIN_VOLUME_VALUES)
        self.assertIn(config.BACKTEST_MIN_CLOSE, config.BACKTEST_MIN_CLOSE_VALUES)
        self.assertIn(
            config.BACKTEST_MIN_PREDICTED_RETURN,
            config.BACKTEST_MIN_PREDICTED_RETURN_VALUES,
        )
        self.assertEqual("future_return", config.TARGET_TYPE)
        self.assertEqual(0.001, config.RETURN_CALIBRATION_MIN_ABS_SIGNAL)
        self.assertEqual("mae", config.XGB_PARAMS["eval_metric"])

    def test_walk_forward_config_is_available(self):
        from src import config

        self.assertEqual(
            "reports/walk_forward_predictions.csv",
            project_relative(config.WALK_FORWARD_PREDICTION_PATH),
        )
        self.assertEqual(
            "reports/walk_forward_fold_metrics.csv",
            project_relative(config.WALK_FORWARD_FOLD_METRICS_PATH),
        )
        self.assertEqual(
            "reports/walk_forward_backtest.csv",
            project_relative(config.WALK_FORWARD_BACKTEST_PATH),
        )
        self.assertEqual(
            "reports/walk_forward_backtest_metrics.json",
            project_relative(config.WALK_FORWARD_BACKTEST_METRICS_PATH),
        )
        self.assertGreater(config.WALK_FORWARD_INITIAL_TRAIN_RATIO, 0)
        self.assertGreater(config.WALK_FORWARD_VALIDATION_RATIO, 0)
        self.assertGreater(config.WALK_FORWARD_TEST_RATIO, 0)
        self.assertGreater(config.WALK_FORWARD_STEP_RATIO, 0)


if __name__ == "__main__":
    unittest.main()
