from __future__ import annotations

import argparse
from pathlib import Path

from model5.create_risk_mart import create_local_risk_mart
from model5.risk_features import (
    DEFAULT_INPUT_CSV,
    create_risk_features,
    load_stock_prices_csv,
    save_risk_features_csv,
)
from model5.save_risk_predictions import (
    prepare_prediction_df,
    save_predictions_csv,
    save_test_evaluation_csv,
)
from model5.train_risk_model import save_metrics_json, train_models


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output_model5"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the local CSV stock risk-alert ML pipeline."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input cleaned stock price CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated CSV/JSON outputs.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Time-based train ratio. No shuffle is used.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="HIGH_RISK threshold for class-1 probability.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[pipeline] Starting local CSV risk-alert pipeline")
    print(f"[pipeline] Input CSV: {args.input_csv}")
    print(f"[pipeline] Output dir: {output_dir}")

    prices_df = load_stock_prices_csv(args.input_csv)
    features_df = create_risk_features(prices_df)

    features_path = save_risk_features_csv(
        features_df,
        output_dir / "risk_features.csv",
    )

    _, prediction_df, metrics = train_models(
        features_df,
        train_ratio=args.train_ratio,
        threshold=args.threshold,
    )

    predictions_path = save_predictions_csv(
        prediction_df,
        output_dir / "risk_predictions.csv",
    )
    evaluation_path = save_test_evaluation_csv(
        prediction_df,
        output_dir / "risk_test_evaluation.csv",
    )
    metrics_path = save_metrics_json(metrics, output_dir / "risk_metrics.json")

    prepared_predictions = prepare_prediction_df(prediction_df)
    mart_path = create_local_risk_mart(
        predictions_df=prepared_predictions,
        features_df=features_df,
        output_path=output_dir / "mart_risk_alerts.csv",
    )

    high_risk_count = int((prepared_predictions["risk_label"] == "HIGH_RISK").sum())
    print("\n[pipeline] Done")
    print(f"  Features:    {features_path}")
    print(f"  Predictions: {predictions_path}")
    print(f"  Evaluation:  {evaluation_path}")
    print(f"  Metrics:     {metrics_path}")
    print(f"  Mart:        {mart_path}")
    print(f"  HIGH_RISK rows: {high_risk_count:,}/{len(prepared_predictions):,}")


if __name__ == "__main__":
    main()
