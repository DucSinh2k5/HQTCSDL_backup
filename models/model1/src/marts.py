from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


MODEL_NAME = "model1"

PRICE_FORECAST_COLUMNS = [
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
]

TOP_EXPECTED_RETURN_COLUMNS = [
    "model_run_id",
    "prediction_date",
    "rank",
    "symbol",
    "current_close",
    "predicted_close",
    "actual_return",
    "predicted_return",
    "direction_correct",
    "model_name",
    "created_at",
]

BACKTEST_DAILY_COLUMNS = [
    "model_run_id",
    "trading_date",
    "selected_symbols",
    "selected_count",
    "daily_return",
    "benchmark_return",
    "avg_predicted_return",
    "daily_cost_rate",
    "daily_return_net",
    "cumulative_return",
    "cumulative_return_net",
    "benchmark_cumulative_return",
    "model_name",
    "created_at",
]

METRICS_COLUMNS = [
    "model_run_id",
    "run_date",
    "metric_scope",
    "metric_name",
    "metric_value",
    "model_name",
    "created_at",
]

DAILY_INSIGHTS_COLUMNS = [
    "model_run_id",
    "insight_date",
    "insight_type",
    "source_model",
    "symbol",
    "sector",
    "severity",
    "metric_name",
    "metric_value",
    "title",
    "message",
    "created_at",
]

MART_TABLE_NAMES = {
    "mart_model1_price_forecast": "mart_model1_price_forecast",
    "mart_model1_top_expected_return": "mart_model1_top_expected_return",
    "mart_model1_backtest_daily": "mart_model1_backtest_daily",
    "mart_model1_metrics": "mart_model1_metrics",
    "daily_insights_model1": "daily_insights",
}


def _created_at(created_at=None) -> pd.Timestamp:
    if created_at is None:
        return pd.Timestamp.now().floor("s")
    return pd.Timestamp(created_at)


def _date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _require_columns(df: pd.DataFrame, columns: list[str], frame_name: str) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{frame_name} is missing required columns: {', '.join(missing_columns)}"
        )


def _direction_correct(predictions_df: pd.DataFrame) -> pd.Series:
    if {"actual_direction", "predicted_direction"}.issubset(predictions_df.columns):
        actual = _numeric_series(predictions_df["actual_direction"])
        predicted = _numeric_series(predictions_df["predicted_direction"])
        correct = actual.eq(predicted)
        return correct.where(actual.notna() & predicted.notna()).astype("Int64")

    actual_return = _numeric_series(predictions_df["target_return"])
    predicted_return = _numeric_series(predictions_df["predicted_return"])
    actual_direction = actual_return.ge(0)
    predicted_direction = predicted_return.ge(0)
    correct = actual_direction.eq(predicted_direction)
    return correct.where(actual_return.notna() & predicted_return.notna()).astype("Int64")


def build_price_forecast_mart(
    predictions_df: pd.DataFrame,
    model_run_id: str,
    created_at=None,
) -> pd.DataFrame:
    _require_columns(
        predictions_df,
        [
            "trading_date",
            "future_trading_date",
            "symbol",
            "close",
            "future_close",
            "target_return",
            "predicted_return",
            "predicted_close",
        ],
        "predictions_df",
    )

    mart = pd.DataFrame(
        {
            "model_run_id": model_run_id,
            "prediction_date": _date_series(predictions_df["trading_date"]),
            "target_date": _date_series(predictions_df["future_trading_date"]),
            "symbol": predictions_df["symbol"].astype(str).str.upper().str.strip(),
            "current_close": _numeric_series(predictions_df["close"]),
            "real_close": _numeric_series(predictions_df["future_close"]),
            "predicted_close": _numeric_series(predictions_df["predicted_close"]),
            "actual_return": _numeric_series(predictions_df["target_return"]),
            "predicted_return": _numeric_series(predictions_df["predicted_return"]),
            "direction_correct": _direction_correct(predictions_df),
            "model_name": MODEL_NAME,
            "created_at": _created_at(created_at),
        }
    )
    return mart[PRICE_FORECAST_COLUMNS]


def build_top_expected_return_mart(
    price_forecast_mart: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    _require_columns(
        price_forecast_mart,
        [
            "model_run_id",
            "prediction_date",
            "symbol",
            "current_close",
            "predicted_close",
            "actual_return",
            "predicted_return",
            "direction_correct",
            "model_name",
            "created_at",
        ],
        "price_forecast_mart",
    )

    valid_rows = price_forecast_mart.dropna(
        subset=["prediction_date", "predicted_return"]
    ).copy()
    if valid_rows.empty:
        return pd.DataFrame(columns=TOP_EXPECTED_RETURN_COLUMNS)

    latest_prediction_date = valid_rows["prediction_date"].max()
    latest_rows = valid_rows[
        valid_rows["prediction_date"] == latest_prediction_date
    ].copy()
    latest_rows = latest_rows.sort_values(
        ["predicted_return", "symbol"],
        ascending=[False, True],
    ).head(top_n)
    latest_rows["rank"] = range(1, len(latest_rows) + 1)

    return latest_rows[TOP_EXPECTED_RETURN_COLUMNS].reset_index(drop=True)


def build_backtest_daily_mart(
    backtest_df: pd.DataFrame,
    model_run_id: str,
    created_at=None,
) -> pd.DataFrame:
    _require_columns(
        backtest_df,
        [
            "trading_date",
            "selected_symbols",
            "selected_count",
            "daily_return",
            "benchmark_return",
            "avg_predicted_return",
            "daily_cost_rate",
            "daily_return_net",
            "cumulative_return",
            "cumulative_return_net",
            "benchmark_cumulative_return",
        ],
        "backtest_df",
    )

    mart = pd.DataFrame(
        {
            "model_run_id": model_run_id,
            "trading_date": _date_series(backtest_df["trading_date"]),
            "selected_symbols": backtest_df["selected_symbols"].astype(str),
            "selected_count": _numeric_series(backtest_df["selected_count"]).astype(
                "Int64"
            ),
            "daily_return": _numeric_series(backtest_df["daily_return"]),
            "benchmark_return": _numeric_series(backtest_df["benchmark_return"]),
            "avg_predicted_return": _numeric_series(
                backtest_df["avg_predicted_return"]
            ),
            "daily_cost_rate": _numeric_series(backtest_df["daily_cost_rate"]),
            "daily_return_net": _numeric_series(backtest_df["daily_return_net"]),
            "cumulative_return": _numeric_series(backtest_df["cumulative_return"]),
            "cumulative_return_net": _numeric_series(
                backtest_df["cumulative_return_net"]
            ),
            "benchmark_cumulative_return": _numeric_series(
                backtest_df["benchmark_cumulative_return"]
            ),
            "model_name": MODEL_NAME,
            "created_at": _created_at(created_at),
        }
    )
    return mart[BACKTEST_DAILY_COLUMNS]


def build_metrics_mart(
    metrics: Mapping[str, object],
    backtest_metrics: Mapping[str, object],
    model_run_id: str,
    created_at=None,
) -> pd.DataFrame:
    created_at_ts = _created_at(created_at)
    run_date = created_at_ts.date()
    rows = []

    for scope, metric_mapping in [
        ("holdout", metrics),
        ("backtest", backtest_metrics),
    ]:
        for metric_name, metric_value in metric_mapping.items():
            numeric_value = pd.to_numeric(pd.Series([metric_value]), errors="coerce").iloc[0]
            if pd.isna(numeric_value):
                continue

            rows.append(
                {
                    "model_run_id": model_run_id,
                    "run_date": run_date,
                    "metric_scope": scope,
                    "metric_name": str(metric_name),
                    "metric_value": float(numeric_value),
                    "model_name": MODEL_NAME,
                    "created_at": created_at_ts,
                }
            )

    return pd.DataFrame(rows, columns=METRICS_COLUMNS)


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_daily_insights_mart(
    top_expected_return_mart: pd.DataFrame,
    backtest_metrics: Mapping[str, object],
    model_run_id: str,
    created_at=None,
) -> pd.DataFrame:
    created_at_ts = _created_at(created_at)
    insight_date = created_at_ts.date()
    rows = []

    if not top_expected_return_mart.empty:
        latest_date = top_expected_return_mart["prediction_date"].max()
        latest_top = top_expected_return_mart[
            top_expected_return_mart["prediction_date"] == latest_date
        ].copy()
        latest_top = latest_top.sort_values("rank")
        top_symbol = str(latest_top.iloc[0]["symbol"])
        top_return = float(latest_top.iloc[0]["predicted_return"])
        symbol_summary = ", ".join(
            f"{row.symbol} {_format_pct(float(row.predicted_return))}"
            for row in latest_top.itertuples()
        )
        insight_date = latest_date
        rows.append(
            {
                "model_run_id": model_run_id,
                "insight_date": insight_date,
                "insight_type": "model1",
                "source_model": MODEL_NAME,
                "symbol": top_symbol,
                "sector": None,
                "severity": "success" if top_return > 0 else "info",
                "metric_name": "predicted_return",
                "metric_value": top_return,
                "title": "Model1 Top expected return",
                "message": (
                    f"Top expected return 5D on {latest_date}: {symbol_summary}."
                ),
                "created_at": created_at_ts,
            }
        )

    cumulative_return_net = pd.to_numeric(
        pd.Series([backtest_metrics.get("Cumulative_Return_Net")]),
        errors="coerce",
    ).iloc[0]
    benchmark_cumulative_return = pd.to_numeric(
        pd.Series([backtest_metrics.get("Benchmark_Cumulative_Return")]),
        errors="coerce",
    ).iloc[0]

    if pd.notna(cumulative_return_net) and pd.notna(benchmark_cumulative_return):
        outperformance = float(cumulative_return_net - benchmark_cumulative_return)
        if outperformance > 0:
            severity = "success"
        elif outperformance < 0:
            severity = "danger"
        else:
            severity = "warning"

        rows.append(
            {
                "model_run_id": model_run_id,
                "insight_date": insight_date,
                "insight_type": "model1",
                "source_model": MODEL_NAME,
                "symbol": None,
                "sector": None,
                "severity": severity,
                "metric_name": "cumulative_return_net_over_benchmark",
                "metric_value": outperformance,
                "title": "Model1 Backtest outperformance",
                "message": (
                    "Model1 net cumulative return is "
                    f"{_format_pct(outperformance)} over benchmark "
                    f"({_format_pct(float(cumulative_return_net))} vs "
                    f"{_format_pct(float(benchmark_cumulative_return))})."
                ),
                "created_at": created_at_ts,
            }
        )

    return pd.DataFrame(rows, columns=DAILY_INSIGHTS_COLUMNS)


def build_all_model1_marts(
    predictions_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    metrics: Mapping[str, object],
    backtest_metrics: Mapping[str, object],
    model_run_id: str,
    created_at=None,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    created_at_ts = _created_at(created_at)
    price_forecast = build_price_forecast_mart(
        predictions_df=predictions_df,
        model_run_id=model_run_id,
        created_at=created_at_ts,
    )
    top_expected_return = build_top_expected_return_mart(
        price_forecast,
        top_n=top_n,
    )
    backtest_daily = build_backtest_daily_mart(
        backtest_df=backtest_df,
        model_run_id=model_run_id,
        created_at=created_at_ts,
    )
    metrics_mart = build_metrics_mart(
        metrics=metrics,
        backtest_metrics=backtest_metrics,
        model_run_id=model_run_id,
        created_at=created_at_ts,
    )
    daily_insights = build_daily_insights_mart(
        top_expected_return_mart=top_expected_return,
        backtest_metrics=backtest_metrics,
        model_run_id=model_run_id,
        created_at=created_at_ts,
    )

    return {
        "mart_model1_price_forecast": price_forecast,
        "mart_model1_top_expected_return": top_expected_return,
        "mart_model1_backtest_daily": backtest_daily,
        "mart_model1_metrics": metrics_mart,
        "daily_insights_model1": daily_insights,
    }


def write_marts_to_csv(
    marts: Mapping[str, pd.DataFrame],
    output_dir: Path | str,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_paths = {}
    for mart_name, mart_df in marts.items():
        path = output_path / f"{mart_name}.csv"
        mart_df.to_csv(path, index=False)
        written_paths[mart_name] = path

    return written_paths


def _quote_identifier(identifier: str) -> str:
    return "`" + str(identifier).replace("`", "``") + "`"


def _qualified_table(database: str, table: str) -> str:
    return f"{_quote_identifier(database)}.{_quote_identifier(table)}"


def create_model1_mart_tables(client, database: str = "stock_mart") -> None:
    client.command(f"CREATE DATABASE IF NOT EXISTS {_quote_identifier(database)}")

    table_sql = {
        "mart_model1_price_forecast": f"""
            CREATE TABLE IF NOT EXISTS {_qualified_table(database, "mart_model1_price_forecast")}
            (
                model_run_id UUID,
                prediction_date Date,
                target_date Nullable(Date),
                symbol String,
                current_close Nullable(Float64),
                real_close Nullable(Float64),
                predicted_close Nullable(Float64),
                actual_return Nullable(Float64),
                predicted_return Nullable(Float64),
                direction_correct Nullable(UInt8),
                model_name String,
                created_at DateTime
            )
            ENGINE = MergeTree
            ORDER BY (symbol, prediction_date, model_run_id)
        """,
        "mart_model1_top_expected_return": f"""
            CREATE TABLE IF NOT EXISTS {_qualified_table(database, "mart_model1_top_expected_return")}
            (
                model_run_id UUID,
                prediction_date Date,
                rank UInt16,
                symbol String,
                current_close Nullable(Float64),
                predicted_close Nullable(Float64),
                actual_return Nullable(Float64),
                predicted_return Nullable(Float64),
                direction_correct Nullable(UInt8),
                model_name String,
                created_at DateTime
            )
            ENGINE = MergeTree
            ORDER BY (prediction_date, rank, symbol, model_run_id)
        """,
        "mart_model1_backtest_daily": f"""
            CREATE TABLE IF NOT EXISTS {_qualified_table(database, "mart_model1_backtest_daily")}
            (
                model_run_id UUID,
                trading_date Date,
                selected_symbols String,
                selected_count UInt16,
                daily_return Nullable(Float64),
                benchmark_return Nullable(Float64),
                avg_predicted_return Nullable(Float64),
                daily_cost_rate Nullable(Float64),
                daily_return_net Nullable(Float64),
                cumulative_return Nullable(Float64),
                cumulative_return_net Nullable(Float64),
                benchmark_cumulative_return Nullable(Float64),
                model_name String,
                created_at DateTime
            )
            ENGINE = MergeTree
            ORDER BY (trading_date, model_run_id)
        """,
        "mart_model1_metrics": f"""
            CREATE TABLE IF NOT EXISTS {_qualified_table(database, "mart_model1_metrics")}
            (
                model_run_id UUID,
                run_date Date,
                metric_scope String,
                metric_name String,
                metric_value Nullable(Float64),
                model_name String,
                created_at DateTime
            )
            ENGINE = MergeTree
            ORDER BY (model_name, run_date, metric_scope, metric_name, model_run_id)
        """,
        "daily_insights": f"""
            CREATE TABLE IF NOT EXISTS {_qualified_table(database, "daily_insights")}
            (
                model_run_id UUID,
                insight_date Date,
                insight_type String,
                source_model Nullable(String),
                symbol Nullable(String),
                sector Nullable(String),
                severity String,
                metric_name String,
                metric_value Nullable(Float64),
                title String,
                message String,
                created_at DateTime
            )
            ENGINE = MergeTree
            ORDER BY (insight_date, insight_type, model_run_id)
        """,
    }

    for create_sql in table_sql.values():
        client.command(create_sql)


def _clean_for_clickhouse(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(object).where(pd.notna(df), None)


def upload_marts_to_clickhouse(
    marts: Mapping[str, pd.DataFrame],
    client=None,
    database: str = "stock_mart",
) -> None:
    if client is None:
        import clickhouse_connect

        from src.config import (
            CLICKHOUSE_DATABASE,
            CLICKHOUSE_HOST,
            CLICKHOUSE_PASSWORD,
            CLICKHOUSE_PORT,
            CLICKHOUSE_SECURE,
            CLICKHOUSE_USER,
        )

        client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            username=CLICKHOUSE_USER,
            password=CLICKHOUSE_PASSWORD,
            database=CLICKHOUSE_DATABASE,
            secure=CLICKHOUSE_SECURE,
        )

    create_model1_mart_tables(client, database=database)

    for mart_name, mart_df in marts.items():
        if mart_df.empty:
            continue
        table_name = MART_TABLE_NAMES[mart_name]
        client.insert_df(
            _qualified_table(database, table_name),
            _clean_for_clickhouse(mart_df),
        )
