import unittest
from pathlib import Path
import shutil

import pandas as pd

from src.marts import (
    build_all_model1_marts,
    build_backtest_daily_mart,
    build_daily_insights_mart,
    build_metrics_mart,
    build_price_forecast_mart,
    build_top_expected_return_mart,
    create_model1_mart_tables,
    write_marts_to_csv,
)


class Model1MartsTests(unittest.TestCase):
    def setUp(self):
        self.created_at = pd.Timestamp("2026-06-05 01:00:00")
        self.model_run_id = "11111111-1111-1111-1111-111111111111"
        self.predictions_df = pd.DataFrame(
            {
                "trading_date": ["2026-06-01", "2026-06-01", "2026-06-02"],
                "future_trading_date": [
                    "2026-06-08",
                    "2026-06-08",
                    "2026-06-09",
                ],
                "symbol": ["AAA", "BBB", "AAA"],
                "close": [100.0, 50.0, 101.0],
                "future_close": [103.0, 49.0, 104.0],
                "target_return": [0.03, -0.02, 0.0297029703],
                "predicted_return": [0.05, 0.01, 0.04],
                "predicted_close": [105.0, 50.5, 105.04],
                "actual_direction": [1, 0, 1],
                "predicted_direction": [1, 1, 1],
            }
        )
        self.backtest_df = pd.DataFrame(
            {
                "trading_date": ["2026-06-01", "2026-06-02"],
                "selected_symbols": ["AAA,BBB", "AAA"],
                "selected_count": [2, 1],
                "daily_return": [0.02, 0.01],
                "benchmark_return": [0.01, -0.01],
                "avg_predicted_return": [0.03, 0.04],
                "daily_cost_rate": [0.004, 0.004],
                "daily_return_net": [0.016, 0.006],
                "cumulative_return": [0.02, 0.0302],
                "cumulative_return_net": [0.016, 0.022096],
                "benchmark_cumulative_return": [0.01, -0.0001],
            }
        )
        self.metrics = {
            "Return_MAE": 0.036,
            "Baseline_Return_MAE": 0.038,
            "Directional_Accuracy": 54.0,
        }
        self.backtest_metrics = {
            "Cumulative_Return_Net": 0.22,
            "Benchmark_Cumulative_Return": 0.10,
            "Hit_Rate": 60.0,
        }

    def test_build_price_forecast_mart_normalizes_prediction_rows(self):
        mart = build_price_forecast_mart(
            self.predictions_df,
            model_run_id=self.model_run_id,
            created_at=self.created_at,
        )

        self.assertEqual(3, len(mart))
        self.assertEqual(
            [
                "model_run_id",
                "prediction_date",
                "target_date",
                "symbol",
                "current_close",
                "real_close",
                "predicted_close",
                "actual_return",
                "predicted_return",
                "direction_correct",
                "model_name",
                "created_at",
            ],
            mart.columns.tolist(),
        )
        self.assertEqual(pd.Timestamp("2026-06-01").date(), mart.loc[0, "prediction_date"])
        self.assertEqual(pd.Timestamp("2026-06-08").date(), mart.loc[0, "target_date"])
        self.assertEqual("model1", mart.loc[0, "model_name"])
        self.assertEqual(1, mart.loc[0, "direction_correct"])
        self.assertEqual(0, mart.loc[1, "direction_correct"])
        self.assertAlmostEqual(103.0, mart.loc[0, "real_close"])

    def test_build_top_expected_return_mart_ranks_latest_prediction_date(self):
        price_mart = build_price_forecast_mart(
            self.predictions_df,
            model_run_id=self.model_run_id,
            created_at=self.created_at,
        )

        top_mart = build_top_expected_return_mart(price_mart, top_n=2)

        self.assertEqual(1, len(top_mart))
        self.assertEqual(pd.Timestamp("2026-06-02").date(), top_mart.loc[0, "prediction_date"])
        self.assertEqual(1, top_mart.loc[0, "rank"])
        self.assertEqual("AAA", top_mart.loc[0, "symbol"])
        self.assertAlmostEqual(0.04, top_mart.loc[0, "predicted_return"])

    def test_build_backtest_daily_mart_preserves_strategy_columns(self):
        mart = build_backtest_daily_mart(
            self.backtest_df,
            model_run_id=self.model_run_id,
            created_at=self.created_at,
        )

        self.assertEqual(2, len(mart))
        self.assertEqual(pd.Timestamp("2026-06-02").date(), mart.loc[1, "trading_date"])
        self.assertEqual("AAA", mart.loc[1, "selected_symbols"])
        self.assertAlmostEqual(0.022096, mart.loc[1, "cumulative_return_net"])

    def test_build_metrics_mart_flattens_model_and_backtest_metrics(self):
        mart = build_metrics_mart(
            self.metrics,
            self.backtest_metrics,
            model_run_id=self.model_run_id,
            created_at=self.created_at,
        )

        rows = {
            (row.metric_scope, row.metric_name): row.metric_value
            for row in mart.itertuples()
        }
        self.assertAlmostEqual(0.036, rows[("holdout", "Return_MAE")])
        self.assertAlmostEqual(0.22, rows[("backtest", "Cumulative_Return_Net")])
        self.assertTrue((mart["model_name"] == "model1").all())

    def test_build_daily_insights_mart_creates_top_and_backtest_messages(self):
        price_mart = build_price_forecast_mart(
            self.predictions_df,
            model_run_id=self.model_run_id,
            created_at=self.created_at,
        )
        top_mart = build_top_expected_return_mart(price_mart, top_n=2)
        insights = build_daily_insights_mart(
            top_expected_return_mart=top_mart,
            backtest_metrics=self.backtest_metrics,
            model_run_id=self.model_run_id,
            created_at=self.created_at,
        )

        self.assertEqual(2, len(insights))
        self.assertEqual(["model1", "model1"], insights["source_model"].tolist())
        self.assertIn("Top expected return", insights.loc[0, "title"])
        self.assertIn("AAA", insights.loc[0, "message"])
        self.assertEqual("success", insights.loc[1, "severity"])
        self.assertAlmostEqual(0.12, insights.loc[1, "metric_value"])

    def test_build_all_model1_marts_returns_named_marts(self):
        marts = build_all_model1_marts(
            predictions_df=self.predictions_df,
            backtest_df=self.backtest_df,
            metrics=self.metrics,
            backtest_metrics=self.backtest_metrics,
            model_run_id=self.model_run_id,
            created_at=self.created_at,
            top_n=2,
        )

        self.assertEqual(
            {
                "mart_model1_price_forecast",
                "mart_model1_top_expected_return",
                "mart_model1_backtest_daily",
                "mart_model1_metrics",
                "daily_insights_model1",
            },
            set(marts),
        )

    def test_write_marts_to_csv_writes_each_named_mart(self):
        marts = build_all_model1_marts(
            predictions_df=self.predictions_df,
            backtest_df=self.backtest_df,
            metrics=self.metrics,
            backtest_metrics=self.backtest_metrics,
            model_run_id=self.model_run_id,
            created_at=self.created_at,
            top_n=2,
        )

        output_dir = Path.cwd() / "reports" / "marts" / "test_marts_output"
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)

        try:
            written = write_marts_to_csv(marts, output_dir)

            self.assertEqual(set(marts), set(written))
            for path in written.values():
                self.assertTrue(path.exists())
                self.assertEqual(".csv", path.suffix)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

    def test_create_model1_mart_tables_keeps_nullable_columns_out_of_sorting_key(self):
        class FakeClient:
            def __init__(self):
                self.commands = []

            def command(self, sql):
                self.commands.append(sql)

        client = FakeClient()

        create_model1_mart_tables(client, database="stock_mart")

        daily_insights_sql = next(
            sql for sql in client.commands if "daily_insights" in sql
        )
        order_by_clause = daily_insights_sql.split("ORDER BY", 1)[1]
        self.assertNotIn("source_model", order_by_clause)
        self.assertNotIn("symbol", order_by_clause)
        self.assertNotIn("sector", order_by_clause)


if __name__ == "__main__":
    unittest.main()
