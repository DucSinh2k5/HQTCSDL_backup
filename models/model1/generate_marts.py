from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import uuid

import pandas as pd


MODEL1_DIR = Path(__file__).resolve().parent
if str(MODEL1_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL1_DIR))

from src.config import (  # noqa: E402
    BACKTEST_METRICS_PATH,
    BACKTEST_PATH,
    METRICS_PATH,
    PREDICTION_PATH,
)
from src.marts import (  # noqa: E402
    build_all_model1_marts,
    upload_marts_to_clickhouse,
    write_marts_to_csv,
)


MART_OUTPUT_DIR = MODEL1_DIR / "reports" / "marts"


def load_json(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_model1_marts(
    predictions_path: Path | str = PREDICTION_PATH,
    backtest_path: Path | str = BACKTEST_PATH,
    metrics_path: Path | str = METRICS_PATH,
    backtest_metrics_path: Path | str = BACKTEST_METRICS_PATH,
    output_dir: Path | str = MART_OUTPUT_DIR,
    model_run_id: str | None = None,
    top_n: int = 10,
    upload_clickhouse: bool = False,
    clickhouse_database: str = "stock_mart",
) -> dict[str, Path]:
    run_id = model_run_id or str(uuid.uuid4())
    created_at = pd.Timestamp.now().floor("s")

    predictions_df = pd.read_csv(predictions_path)
    backtest_df = pd.read_csv(backtest_path)
    metrics = load_json(metrics_path)
    backtest_metrics = load_json(backtest_metrics_path)

    marts = build_all_model1_marts(
        predictions_df=predictions_df,
        backtest_df=backtest_df,
        metrics=metrics,
        backtest_metrics=backtest_metrics,
        model_run_id=run_id,
        created_at=created_at,
        top_n=top_n,
    )
    written_paths = write_marts_to_csv(marts, output_dir)

    if upload_clickhouse:
        upload_marts_to_clickhouse(
            marts,
            database=clickhouse_database,
        )

    return written_paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate dashboard-ready Model 1 marts and insights."
    )
    parser.add_argument("--predictions-path", default=str(PREDICTION_PATH))
    parser.add_argument("--backtest-path", default=str(BACKTEST_PATH))
    parser.add_argument("--metrics-path", default=str(METRICS_PATH))
    parser.add_argument("--backtest-metrics-path", default=str(BACKTEST_METRICS_PATH))
    parser.add_argument("--output-dir", default=str(MART_OUTPUT_DIR))
    parser.add_argument("--model-run-id", default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--upload-clickhouse", action="store_true")
    parser.add_argument("--clickhouse-database", default="stock_mart")
    return parser.parse_args(argv)


def display_output_path(path: Path | str) -> str:
    output_path = Path(path)
    try:
        return output_path.relative_to(MODEL1_DIR).as_posix()
    except ValueError:
        return output_path.name


def main(argv=None) -> None:
    args = parse_args(argv)
    written_paths = generate_model1_marts(
        predictions_path=args.predictions_path,
        backtest_path=args.backtest_path,
        metrics_path=args.metrics_path,
        backtest_metrics_path=args.backtest_metrics_path,
        output_dir=args.output_dir,
        model_run_id=args.model_run_id,
        top_n=args.top_n,
        upload_clickhouse=args.upload_clickhouse,
        clickhouse_database=args.clickhouse_database,
    )

    print("Generated Model 1 marts:")
    for mart_name, path in written_paths.items():
        print(f"- {mart_name}: {display_output_path(path)}")


if __name__ == "__main__":
    main()
