from pathlib import Path

import numpy as np
import pandas as pd

try:
    from connect_clickhouse.clickhouse_client import get_clickhouse_client
except ModuleNotFoundError:
    from clickhouse_client import get_clickhouse_client


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FEATURES_CSV = PROJECT_DIR / "output_model5" / "risk_features.csv"

FEATURES_ALL_COLUMNS = [
    "trading_date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
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
    "created_at",
]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
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


def create_features_all_table(client, database: str = "stock") -> None:
    client.command(f"CREATE DATABASE IF NOT EXISTS {database}")
    client.command(f"DROP TABLE IF EXISTS {database}.features_all")
    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {database}.features_all
        (
            trading_date Date,
            symbol String,
            open Nullable(Float64),
            high Nullable(Float64),
            low Nullable(Float64),
            close Nullable(Float64),
            volume Nullable(Float64),
            return_1d Nullable(Float64),
            return_3d Nullable(Float64),
            return_5d Nullable(Float64),
            return_10d Nullable(Float64),
            return_20d Nullable(Float64),
            ma_5 Nullable(Float64),
            ma_20 Nullable(Float64),
            ma_50 Nullable(Float64),
            price_vs_ma20 Nullable(Float64),
            ma5_vs_ma20 Nullable(Float64),
            volatility_5d Nullable(Float64),
            volatility_20d Nullable(Float64),
            volatility_change Nullable(Float64),
            rolling_max_20d Nullable(Float64),
            drawdown_20d Nullable(Float64),
            volume_ma_5 Nullable(Float64),
            volume_ma_20 Nullable(Float64),
            volume_ratio_5_20 Nullable(Float64),
            volume_change_1d Nullable(Float64),
            daily_range Nullable(Float64),
            body_ratio Nullable(Float64),
            close_position Nullable(Float64),
            created_at DateTime
        )
        ENGINE = MergeTree
        ORDER BY (symbol, trading_date)
        """
    )
    print(f"[clickhouse] Recreated table: {database}.features_all")


def load_features_csv() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {FEATURES_CSV}")

    df = pd.read_csv(FEATURES_CSV)
    df = df.drop(columns=["future_return_5d", "risk_drop_label"], errors="ignore")

    for column in FEATURES_ALL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    df = df[FEATURES_ALL_COLUMNS].copy()

    df["symbol"] = df["symbol"].astype("string").str.strip().str.upper()
    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce").dt.date
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["created_at"] = df["created_at"].fillna(pd.Timestamp.now().floor("s"))

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.where(pd.notna(df), None)


def main() -> None:
    client = get_clickhouse_client()
    if client is None:
        print("[clickhouse] Could not create ClickHouse client. Upload stopped.")
        return

    create_features_all_table(client, database="stock")
    df = load_features_csv()

    client.insert_df(
        table="stock.features_all",
        df=df,
    )
    print(f"Uploaded {len(df):,} rows to stock.features_all")


if __name__ == "__main__":
    main()
