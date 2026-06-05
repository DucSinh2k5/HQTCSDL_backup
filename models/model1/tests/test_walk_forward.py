import importlib.util
from pathlib import Path
import unittest
import warnings
from unittest.mock import call, patch

import numpy as np
import pandas as pd

from src.walk_forward import create_walk_forward_folds, run_walk_forward_backtest


class ConstantReturnModel:
    def __init__(self, predicted_return):
        self.predicted_return = predicted_return
        self.best_iteration = 3

    def predict(self, X):
        return np.full(len(X), self.predicted_return)


class WalkForwardTests(unittest.TestCase):
    def test_cli_create_folders_creates_configured_output_parent_dirs(self):
        script_path = Path(__file__).resolve().parents[1] / "walk_forward.py"
        spec = importlib.util.spec_from_file_location(
            "walk_forward_cli_for_folder_test",
            script_path,
        )
        walk_forward_script = importlib.util.module_from_spec(spec)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The 'u' type code is deprecated.*",
                category=DeprecationWarning,
            )
            spec.loader.exec_module(walk_forward_script)

        configured_paths = [
            walk_forward_script.WALK_FORWARD_PREDICTION_PATH,
            walk_forward_script.WALK_FORWARD_FOLD_METRICS_PATH,
            walk_forward_script.WALK_FORWARD_BACKTEST_PATH,
            walk_forward_script.WALK_FORWARD_BACKTEST_METRICS_PATH,
        ]
        expected_dirs = sorted({str(Path(path).parent) for path in configured_paths})

        with patch.object(walk_forward_script.os, "makedirs") as makedirs:
            walk_forward_script.create_folders()

        self.assertEqual(
            [call(directory, exist_ok=True) for directory in expected_dirs],
            makedirs.call_args_list,
        )
        self.assertNotIn(call("reports", exist_ok=True), makedirs.call_args_list)

    def test_create_walk_forward_folds_expands_train_window_chronologically(self):
        df = pd.DataFrame(
            {
                "trading_date": pd.date_range("2024-01-01", periods=10, freq="D"),
                "symbol": ["AAA"] * 10,
                "close": range(10),
            }
        )

        folds = create_walk_forward_folds(
            df=df,
            initial_train_ratio=0.4,
            validation_ratio=0.2,
            test_ratio=0.2,
            step_ratio=0.2,
        )

        self.assertEqual(2, len(folds))
        self.assertEqual(4, len(folds[0]["train_df"]))
        self.assertEqual(2, len(folds[0]["validation_df"]))
        self.assertEqual(2, len(folds[0]["test_df"]))
        self.assertEqual(6, len(folds[1]["train_df"]))
        self.assertLess(
            folds[0]["train_df"]["trading_date"].max(),
            folds[0]["validation_df"]["trading_date"].min(),
        )
        self.assertLess(
            folds[0]["validation_df"]["trading_date"].max(),
            folds[0]["test_df"]["trading_date"].min(),
        )
        self.assertLess(
            folds[0]["test_df"]["trading_date"].max(),
            folds[1]["test_df"]["trading_date"].min(),
        )

    def test_run_walk_forward_backtest_combines_predictions_metrics_and_backtest(self):
        df = pd.DataFrame(
            {
                "trading_date": pd.date_range("2024-01-01", periods=10, freq="D"),
                "symbol": ["AAA"] * 10,
                "feature": np.arange(10),
                "close": [100.0] * 10,
                "future_close": [101.0] * 10,
                "target_close": [101.0] * 10,
                "target_return": [0.01] * 10,
            }
        )
        train_calls = []

        def train_model_fn(
            X_train,
            y_train,
            params,
            X_val,
            y_val,
            early_stopping_rounds,
            verbose,
        ):
            train_calls.append((X_train, y_train, X_val, y_val))
            return ConstantReturnModel(0.01)

        predictions_df, fold_metrics_df, backtest_df, backtest_metrics = (
            run_walk_forward_backtest(
                df=df,
                features=["feature"],
                params={"n_estimators": 10},
                initial_train_ratio=0.4,
                validation_ratio=0.2,
                test_ratio=0.2,
                step_ratio=0.2,
                early_stopping_rounds=5,
                backtest_kwargs={"top_k": 1},
                train_model_fn=train_model_fn,
                target_type="future_return",
                calibration_min_abs_signal=0.0,
            )
        )

        self.assertEqual(2, len(train_calls))
        self.assertTrue((train_calls[0][1] == 0.01).all())
        self.assertEqual(4, len(predictions_df))
        self.assertEqual([1, 2], fold_metrics_df["fold_id"].tolist())
        self.assertIn("MAE", fold_metrics_df.columns)
        self.assertIn("Return_MAE_Improvement", fold_metrics_df.columns)
        self.assertIn("Beats_Baseline_Return_MAE", fold_metrics_df.columns)
        self.assertIn("predicted_close", predictions_df.columns)
        self.assertEqual(4, len(backtest_df))
        self.assertEqual(2, backtest_metrics["Walk_Forward_Folds"])
        self.assertEqual(4, backtest_metrics["Backtest_Days"])


if __name__ == "__main__":
    unittest.main()
