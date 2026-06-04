import importlib.util
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import pandas as pd


def load_generate_marts_script():
    script_path = Path(__file__).resolve().parents[1] / "generate_marts.py"
    spec = importlib.util.spec_from_file_location("generate_marts_cli", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GenerateMartsCliTests(unittest.TestCase):
    def setUp(self):
        self.script = load_generate_marts_script()
        self.work_dir = Path.cwd() / "reports" / "marts" / "test_generate_marts"
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True)
        self.input_dir = self.work_dir / "input"
        self.output_dir = self.work_dir / "output"
        self.input_dir.mkdir(parents=True, exist_ok=True)

        predictions_df = pd.DataFrame(
            {
                "trading_date": ["2026-06-01"],
                "future_trading_date": ["2026-06-08"],
                "symbol": ["AAA"],
                "close": [100.0],
                "future_close": [103.0],
                "target_return": [0.03],
                "predicted_return": [0.05],
                "predicted_close": [105.0],
                "actual_direction": [1],
                "predicted_direction": [1],
            }
        )
        backtest_df = pd.DataFrame(
            {
                "trading_date": ["2026-06-01"],
                "selected_symbols": ["AAA"],
                "selected_count": [1],
                "daily_return": [0.02],
                "benchmark_return": [0.01],
                "avg_predicted_return": [0.05],
                "daily_cost_rate": [0.004],
                "daily_return_net": [0.016],
                "cumulative_return": [0.02],
                "cumulative_return_net": [0.016],
                "benchmark_cumulative_return": [0.01],
            }
        )

        predictions_df.to_csv(self.input_dir / "predictions.csv", index=False)
        backtest_df.to_csv(self.input_dir / "backtest.csv", index=False)
        (self.input_dir / "metrics.json").write_text(
            json.dumps({"Return_MAE": 0.036}),
            encoding="utf-8",
        )
        (self.input_dir / "backtest_metrics.json").write_text(
            json.dumps(
                {
                    "Cumulative_Return_Net": 0.016,
                    "Benchmark_Cumulative_Return": 0.01,
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_generate_model1_marts_writes_local_outputs_without_upload(self):
        with patch.object(self.script, "upload_marts_to_clickhouse") as upload:
            written = self.script.generate_model1_marts(
                predictions_path=self.input_dir / "predictions.csv",
                backtest_path=self.input_dir / "backtest.csv",
                metrics_path=self.input_dir / "metrics.json",
                backtest_metrics_path=self.input_dir / "backtest_metrics.json",
                output_dir=self.output_dir,
                model_run_id="11111111-1111-1111-1111-111111111111",
                upload_clickhouse=False,
            )

        self.assertFalse(upload.called)
        self.assertEqual(
            {
                "mart_model1_price_forecast",
                "mart_model1_top_expected_return",
                "mart_model1_backtest_daily",
                "mart_model1_metrics",
                "daily_insights_model1",
            },
            set(written),
        )
        for output_path in written.values():
            self.assertTrue(output_path.exists())

    def test_generate_model1_marts_uploads_only_when_requested(self):
        with patch.object(self.script, "upload_marts_to_clickhouse") as upload:
            self.script.generate_model1_marts(
                predictions_path=self.input_dir / "predictions.csv",
                backtest_path=self.input_dir / "backtest.csv",
                metrics_path=self.input_dir / "metrics.json",
                backtest_metrics_path=self.input_dir / "backtest_metrics.json",
                output_dir=self.output_dir,
                model_run_id="11111111-1111-1111-1111-111111111111",
                upload_clickhouse=True,
                clickhouse_database="stock_mart_test",
            )

        upload.assert_called_once()
        self.assertEqual("stock_mart_test", upload.call_args.kwargs["database"])

    def test_display_output_path_uses_model1_relative_path(self):
        path = self.script.MODEL1_DIR / "reports" / "marts" / "example.csv"

        display_path = self.script.display_output_path(path)

        self.assertEqual("reports/marts/example.csv", display_path)


if __name__ == "__main__":
    unittest.main()
