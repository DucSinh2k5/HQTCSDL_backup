import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, TARGET_COL


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["symbol", "trading_date"])

    group = df.groupby("symbol")

    # df["return_1d"] = group["close"].pct_change(1)
    # df["return_3d"] = group["close"].pct_change(3)
    # df["return_5d"] = group["close"].pct_change(5)
    # df["return_10d"] = group["close"].pct_change(10)
    # df["return_20d"] = group["close"].pct_change(20)

    # df["ma_5"] = group["close"].transform(lambda x: x.rolling(5).mean())
    # df["ma_20"] = group["close"].transform(lambda x: x.rolling(20).mean())
    # df["ma_50"] = group["close"].transform(lambda x: x.rolling(50).mean())

    # df["price_vs_ma20"] = df["close"] / df["ma_20"] - 1
    # df["ma5_vs_ma20"] = df["ma_5"] / df["ma_20"] - 1

    # df["volatility_5d"] = group["return_1d"].transform(lambda x: x.rolling(5).std())
    # df["volatility_20d"] = group["return_1d"].transform(lambda x: x.rolling(20).std())
    # df["volatility_change"] = df["volatility_5d"] / df["volatility_20d"] - 1

    # df["rolling_max_20d"] = group["close"].transform(lambda x: x.rolling(20).max())
    # df["drawdown_20d"] = df["close"] / df["rolling_max_20d"] - 1

    # df["volume_ma_5"] = group["volume"].transform(lambda x: x.rolling(5).mean())
    # df["volume_ma_20"] = group["volume"].transform(lambda x: x.rolling(20).mean())
    # df["volume_ratio_5_20"] = df["volume_ma_5"] / df["volume_ma_20"]

    # df["volume_change_1d"] = group["volume"].pct_change(1)

    # price_range = df["high"] - df["low"]
    # df["daily_range"] = price_range / df["close"]
    # df["body_ratio"] = (df["close"] - df["open"]).abs() / price_range
    # df["close_position"] = (df["close"] - df["low"]) / price_range

    df[TARGET_COL] = group["close"].shift(-5) / df["close"] - 1

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COL])

    return df


def get_train_test_data(df: pd.DataFrame, test_start_date: str):
    df = df.copy()
    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")
    cutoff_date = pd.to_datetime(test_start_date)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COL, "trading_date"])

    train_df = df[df["trading_date"] < cutoff_date].copy()
    test_df = df[df["trading_date"] >= cutoff_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Train/Test split produced an empty dataset. "
            f"Check TEST_START_DATE={test_start_date}."
        )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COL]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COL]

    return X_train, X_test, y_train, y_test, train_df, test_df
