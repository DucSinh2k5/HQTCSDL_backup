import unittest
from unittest.mock import Mock, patch

import pandas as pd

import main as model1_main


class MainPipelineTests(unittest.TestCase):
    def test_create_folders_creates_configured_output_parent_directories(self):
        output_paths = [
            model1_main.MODEL_PATH,
            model1_main.METRICS_PATH,
            model1_main.PREDICTION_PATH,
            model1_main.PREDICTION_ACCURACY_PATH,
            model1_main.FEATURE_IMPORTANCE_PATH,
            model1_main.BACKTEST_PATH,
            model1_main.BACKTEST_METRICS_PATH,
            model1_main.BACKTEST_SWEEP_PATH,
        ]
        expected_parent_dirs = {path.parent for path in output_paths}

        with patch.object(model1_main.os, "makedirs") as makedirs:
            model1_main.create_folders()

        created_dirs = {call.args[0] for call in makedirs.call_args_list}

        self.assertEqual(expected_parent_dirs, created_dirs)
        self.assertTrue(
            all(call.kwargs == {"exist_ok": True} for call in makedirs.call_args_list)
        )
        self.assertNotIn("models", created_dirs)
        self.assertNotIn("reports", created_dirs)

    def test_main_trains_on_target_return_and_saves_return_artifact(self):
        raw_df = pd.DataFrame({"raw": [1]})
        processed_df = pd.DataFrame(
            {
                "trading_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"]
                ),
                "symbol": ["AAA", "AAA", "AAA"],
                "feature": [1.0, 2.0, 3.0],
                "close": [100.0, 100.0, 100.0],
                "target_close": [101.0, 102.0, 103.0],
                "target_return": [0.01, 0.02, 0.03],
            }
        )
        train_df = processed_df.iloc[[0]]
        validation_df = processed_df.iloc[[1]]
        test_df = processed_df.iloc[[2]]
        model = Mock()
        model.predict.return_value = [0.02]

        with patch.object(model1_main, "FEATURES", ["feature"]), \
            patch.object(model1_main, "create_folders"), \
            patch.object(model1_main, "load_data", return_value=raw_df), \
            patch.object(
                model1_main,
                "preprocess_data",
                return_value=(processed_df, ["feature"]),
            ), \
            patch.object(
                model1_main,
                "split_train_validation_test_by_time",
                return_value=(
                    train_df,
                    validation_df,
                    test_df,
                    pd.Timestamp("2024-01-02"),
                    pd.Timestamp("2024-01-03"),
                ),
            ), \
            patch.object(model1_main, "train_xgboost_model", return_value=model) as train_fn, \
            patch.object(model1_main, "fit_return_calibrator") as fit_calibrator, \
            patch.object(
                model1_main,
                "evaluate_model",
                return_value=({"Return_MAE": 0.01}, test_df.copy()),
            ) as evaluate_fn, \
            patch.object(
                model1_main,
                "compute_top_k_backtest",
                return_value=(pd.DataFrame(), {}),
            ), \
            patch.object(model1_main, "run_backtest_sweep", return_value=pd.DataFrame()), \
            patch.object(model1_main, "save_model") as save_model_fn, \
            patch.object(model1_main, "save_metrics"), \
            patch.object(
                model1_main,
                "build_prediction_accuracy_table",
                return_value=pd.DataFrame(),
            ), \
            patch.object(model1_main, "save_backtest_metrics"), \
            patch.object(model1_main, "save_feature_importance"), \
            patch.object(pd.DataFrame, "to_csv"):

            calibrator = Mock()
            fit_calibrator.return_value = calibrator

            model1_main.main()

        self.assertEqual([0.01], train_fn.call_args.kwargs["y_train"].tolist())
        self.assertEqual([0.02], train_fn.call_args.kwargs["y_val"].tolist())
        fit_calibrator.assert_called_once()
        self.assertEqual(
            [0.02],
            list(fit_calibrator.call_args.args[0]),
        )
        self.assertEqual(
            [0.02],
            fit_calibrator.call_args.args[1].tolist(),
        )
        self.assertEqual("future_return", evaluate_fn.call_args.kwargs["target_type"])
        self.assertIs(calibrator, evaluate_fn.call_args.kwargs["return_calibrator"])
        self.assertEqual("future_return", save_model_fn.call_args.kwargs["target_type"])
        self.assertIs(calibrator, save_model_fn.call_args.kwargs["return_calibrator"])


if __name__ == "__main__":
    unittest.main()
