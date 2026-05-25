import os
from pathlib import Path

import clickhouse_connect
import numpy as np
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

MODEL_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = MODEL_DIR / "output" / "benchmark_features.csv"
CLICKHOUSE_DATABASE = "stock"
CLICKHOUSE_TABLE = "features_all"
HORIZON = 5

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
        port=int(os.getenv("CLICKHOUSE_PORT", "8443")),
        username=os.getenv("CLICKHOUSE_USER"),
        password=os.getenv("CLICKHOUSE_PASSWORD"),
        database=os.getenv("CLICKHOUSE_DATABASE", "stock"),
        secure=os.getenv("CLICKHOUSE_SECURE", "true").lower() == "true",
    )


def load_features_all(client) -> pd.DataFrame:
    query = f"""
        SELECT *
        FROM {quote_identifier(CLICKHOUSE_DATABASE)}.{quote_identifier(CLICKHOUSE_TABLE)}
        ORDER BY symbol, trading_date
    """
    print("[model4] Loading features from ClickHouse stock.features_all...")
    df = client.query_df(query)
    df.columns = [str(column).strip() for column in df.columns]
    print(f"[model4] Loaded {len(df):,} rows, {df['symbol'].nunique()} symbols")
    return df


def create_benchmark_labels(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = set(FEATURE_COLUMNS + ["trading_date", "symbol", "close"])
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing columns from ClickHouse features_all: {missing_columns}")

    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")
    for column in FEATURE_COLUMNS + ["close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values(["symbol", "trading_date"]).reset_index(drop=True)
    group = df.groupby("symbol", sort=False)
    df["future_close"] = group["close"].shift(-HORIZON)
    df["future_return"] = df["future_close"] / df["close"] - 1
    benchmark = (
        df.groupby("trading_date")["future_return"]
        .mean()
        .rename("benchmark_return")
        .reset_index()
    )
    df = df.merge(benchmark, on="trading_date", how="left")
    df["label"] = (df["future_return"] > df["benchmark_return"]).astype(int)
    return df.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    client = get_client()
    df = load_features_all(client)
    df = create_benchmark_labels(df)
    df_clean = df.dropna(subset=FEATURE_COLUMNS + ["label"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"[model4] Saved {len(df_clean):,} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
