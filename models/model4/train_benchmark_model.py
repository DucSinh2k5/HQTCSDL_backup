from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import clickhouse_connect
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent
MODEL_OUTPUT = MODEL_DIR / "output"
MODEL_SAVE_DIR = MODEL_DIR / "models"
MODEL_PATH = MODEL_SAVE_DIR / "benchmark_outperformance_lgbm.pkl"

ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

CLICKHOUSE_CONNECT_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "default")
FEATURES_DATABASE = os.getenv("CLICKHOUSE_SOURCE_DATABASE", "stock")
FEATURES_TABLE = os.getenv("MODEL4_CLICKHOUSE_FEATURES_TABLE", "features_all")

HORIZON = 5
TRAIN_RATIO = 0.8

FEATURE_COLUMNS = [
    "encode_sector",
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "ma_5",
    "ma_20",
    "ma_50",
    "price_vs_ma20",
    "ma5_vs_ma20",
    "volatility_5d",
    "volatility_20d",
    "volatility_change",
    "rolling_max_20d",
    "drawdown_20d",
    "volume_ma_5",
    "volume_ma_20",
    "volume_ratio_5_20",
    "volume_change_1d",
    "daily_range",
    "body_ratio",
    "close_position",
]


def quote_identifier(name: str) -> str:
    return "`" + str(name).replace("`", "``") + "`"


def get_client():
    return clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST"),
        port=int(os.getenv("CLICKHOUSE_PORT") or "8443"),
        username=os.getenv("CLICKHOUSE_USER"),
        password=os.getenv("CLICKHOUSE_PASSWORD"),
        database=CLICKHOUSE_CONNECT_DATABASE,
        secure=os.getenv("CLICKHOUSE_SECURE", "true").strip().lower()
        in {"1", "true", "yes"},
    )


def create_benchmark_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "trading_date"]).reset_index(drop=True)
    grouped = df.groupby("symbol", sort=False)

    df["future_close"] = grouped["close"].shift(-HORIZON)
    df["future_return"] = df["future_close"] / df["close"] - 1

    benchmark = (
        df.groupby("trading_date")["future_return"]
        .mean()
        .rename("benchmark_return")
        .reset_index()
    )

    df = df.merge(benchmark, on="trading_date", how="left")
    df["label"] = (df["future_return"] > df["benchmark_return"]).astype("Int64")
    return df.replace([np.inf, -np.inf], np.nan)


def load_features() -> pd.DataFrame:
    print(f"[model4] Loading features from ClickHouse: {FEATURES_DATABASE}.{FEATURES_TABLE}")
    query = f"""
        SELECT *
        FROM {quote_identifier(FEATURES_DATABASE)}.{quote_identifier(FEATURES_TABLE)}
        ORDER BY symbol, trading_date
    """
    df = get_client().query_df(query)
    df.columns = [str(column).strip() for column in df.columns]

    required_columns = set(FEATURE_COLUMNS + ["trading_date", "symbol", "close"])
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Missing required columns from ClickHouse features_all: "
            + ", ".join(missing_columns)
        )

    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")

    for column in FEATURE_COLUMNS + ["close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = create_benchmark_labels(df)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label", "trading_date", "symbol"])
    df["label"] = df["label"].astype(int)
    df = df.sort_values(["trading_date", "symbol"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No trainable rows after creating benchmark labels.")

    print(
        f"[model4] Trainable rows: {len(df):,}, "
        f"symbols={df['symbol'].nunique():,}, "
        f"{df['trading_date'].min().date()} -> {df['trading_date'].max().date()}"
    )
    return df


def time_split(df: pd.DataFrame):
    unique_dates = sorted(df["trading_date"].dropna().unique())
    if len(unique_dates) < 3:
        raise ValueError("Need at least 3 unique trading dates for time split.")

    cutoff_idx = int(len(unique_dates) * TRAIN_RATIO)
    cutoff_idx = min(max(cutoff_idx, 1), len(unique_dates) - 1)
    cutoff_date = pd.Timestamp(unique_dates[cutoff_idx])

    train_df = df[df["trading_date"] < cutoff_date].copy()
    test_df = df[df["trading_date"] >= cutoff_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train/Test split produced an empty dataset.")

    print(
        f"[model4] Train: {len(train_df):,} rows "
        f"({train_df['trading_date'].min().date()} -> "
        f"{train_df['trading_date'].max().date()})"
    )
    print(
        f"[model4] Test:  {len(test_df):,} rows "
        f"({test_df['trading_date'].min().date()} -> "
        f"{test_df['trading_date'].max().date()})"
    )
    print(f"[model4] Cutoff date: {cutoff_date.date()}")
    return train_df, test_df


def train_model(train_df: pd.DataFrame):
    print("[model4] Training LightGBM classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"])
    print("[model4] Training done.")
    return model


def save_model(model: Any, metrics: dict[str, Any]) -> Path:
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS,
            "horizon": HORIZON,
            "target_type": "benchmark_outperformance",
            "train_ratio": TRAIN_RATIO,
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    print(f"[model4] Saved model: {MODEL_PATH}")
    return MODEL_PATH


def evaluate_model(model, test_df: pd.DataFrame):
    print("[model4] Evaluating model...")
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"]

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
        if y_test.nunique() == 2
        else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_ratio": TRAIN_RATIO,
        "test_rows": int(len(test_df)),
        "feature_columns": FEATURE_COLUMNS,
    }

    print("\n" + "=" * 40)
    print("  MODEL4 EVALUATION")
    print("=" * 40)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']}")
    print(f"  Confusion Matrix: {metrics['confusion_matrix']}")
    print("=" * 40 + "\n")
    return metrics, y_pred, y_prob


def save_results(model, metrics, test_df, y_pred, y_prob):
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)

    metrics_path = MODEL_OUTPUT / "benchmark_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[model4] Saved metrics: {metrics_path}")

    predictions = test_df[["symbol", "trading_date", "close", "label"]].copy()
    predictions["predicted_label"] = y_pred
    predictions["outperform_probability"] = y_prob
    predictions["prediction_correct"] = (
        predictions["label"] == predictions["predicted_label"]
    )

    pred_path = MODEL_OUTPUT / "benchmark_predictions.csv"
    predictions.to_csv(pred_path, index=False)
    print(f"[model4] Saved predictions: {pred_path}")

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_path = MODEL_OUTPUT / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    print(f"[model4] Saved feature importance: {importance_path}")
    return pred_path


def main() -> None:
    df = load_features()
    train_df, test_df = time_split(df)
    model = train_model(train_df)
    metrics, y_pred, y_prob = evaluate_model(model, test_df)
    save_model(model, metrics)
    save_results(model, metrics, test_df, y_pred, y_prob)
    print("[model4] Done.")


if __name__ == "__main__":
    main()
