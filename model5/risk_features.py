from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT / "data" / "clean" / "Data_500_stocks_clean_ver2.csv"
)
DEFAULT_FEATURE_CSV = PROJECT_ROOT / "output_model5" / "risk_features.csv"
# ghi chú ý nghĩa từng feature:
# return_1d: lợi suất 1 ngày, được tính bằng (giá đóng cửa hôm nay / giá đóng cửa hôm qua) - 1
# return_3d: lợi suất 3 ngày, được tính bằng (giá đóng cửa hôm nay / giá đóng cửa 3 ngày trước) - 1
# return_5d: lợi suất 5 ngày, được tính bằng (giá đóng cửa hôm nay / giá đóng cửa 5 ngày trước) - 1
# return_10d: lợi suất 10 ngày, được tính bằng (giá đóng cửa hôm nay / giá đóng cửa 10 ngày trước) - 1
# return_20d: lợi suất 20 ngày, được tính bằng (giá đóng cửa hôm nay / giá đóng cửa 20 ngày trước) - 1

# ma_5: đường trung bình động 5 ngày của giá đóng cửa
# ma_20: đường trung bình động 20 ngày của giá đóng cửa
# ma_50: đường trung bình động 50 ngày của giá đóng cửa

# price_vs_ma20: so sánh giá đóng cửa hôm nay với đường trung bình động 20 ngày,
# được tính bằng (giá đóng cửa hôm nay / ma_20) - 1

# ma5_vs_ma20: so sánh đường trung bình động 5 ngày với đường trung bình động 20 ngày,
# được tính bằng (ma_5 / ma_20) - 1

# volatility_5d: độ biến động 5 ngày, được tính bằng độ lệch chuẩn rolling 5 ngày của lợi suất hằng ngày
# volatility_20d: độ biến động 20 ngày, được tính bằng độ lệch chuẩn rolling 20 ngày của lợi suất hằng ngày

# volatility_change: mức thay đổi độ biến động,
# được tính bằng (volatility_5d / volatility_20d) - 1

# rolling_max_20d: giá đóng cửa cao nhất trong cửa sổ rolling 20 ngày gần nhất

# drawdown_20d: mức sụt giảm so với đỉnh rolling 20 ngày,
# được tính bằng (giá đóng cửa hôm nay / rolling_max_20d) - 1

# volume_ma_5: trung bình động 5 ngày của khối lượng giao dịch
# volume_ma_20: trung bình động 20 ngày của khối lượng giao dịch

# volume_ratio_5_20: tỷ lệ giữa trung bình động khối lượng 5 ngày và 20 ngày,
# được tính bằng (volume_ma_5 / volume_ma_20)

# volume_change_1d: mức thay đổi khối lượng giao dịch trong 1 ngày,
# được tính bằng (khối lượng hôm nay / khối lượng hôm qua) - 1

# daily_range: biên độ dao động giá trong ngày được chuẩn hóa theo giá đóng cửa,
# được tính bằng (giá cao nhất - giá thấp nhất) / giá đóng cửa

# body_ratio: tỷ lệ thân nến so với biên độ dao động trong ngày,
# được tính bằng abs(giá đóng cửa - giá mở cửa) / (giá cao nhất - giá thấp nhất)

# close_position: vị trí của giá đóng cửa trong biên độ giá trong ngày,
# được tính bằng (giá đóng cửa - giá thấp nhất) / (giá cao nhất - giá thấp nhất)    

FEATURE_COLUMNS = [
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

FEATURE_TABLE_COLUMNS = [
    "trading_date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    *FEATURE_COLUMNS,
    "future_return_5d",
    "risk_drop_label",
    "created_at",
]

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


def _safe_divide(numerator: pd.Series, denominator: pd.Series):
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan)


def normalize_stock_prices(df: pd.DataFrame):
    """Normalize CSV/ClickHouse price data to the pipeline schema."""
    if df.empty:
        print("[risk_features] Input stock price dataframe is empty.")
        return pd.DataFrame(columns=["trading_date", "symbol", *PRICE_COLUMNS])

    normalized = df.copy()
    normalized.columns = [str(col).strip().lower() for col in normalized.columns]

    if "trading_date" not in normalized.columns and "date" in normalized.columns:
        normalized = normalized.rename(columns={"date": "trading_date"})

    required_columns = {"trading_date", "symbol", *PRICE_COLUMNS}
    missing_columns = sorted(required_columns - set(normalized.columns))
    if missing_columns:
        print(f"[risk_features] Missing required price columns: {missing_columns}")
        return pd.DataFrame(columns=["trading_date", "symbol", *PRICE_COLUMNS])

    normalized["symbol"] = (
        normalized["symbol"].astype(str).str.strip().str.upper()
    )
    normalized["trading_date"] = pd.to_datetime(
        normalized["trading_date"], errors="coerce"
    ).dt.normalize()

    for column in PRICE_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna(subset=["symbol", "trading_date", *PRICE_COLUMNS])
    normalized = normalized[normalized["symbol"] != ""]
    normalized = normalized.drop_duplicates(
        subset=["symbol", "trading_date"], keep="last"
    )
    normalized = normalized.sort_values(["symbol", "trading_date"]).reset_index(
        drop=True
    )
    return normalized


def load_stock_prices_csv(csv_path: Path | str = DEFAULT_INPUT_CSV):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[risk_features] Input CSV not found: {csv_path}")
        return pd.DataFrame(columns=["trading_date", "symbol", *PRICE_COLUMNS])

    print(f"[risk_features] Loading stock prices from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    normalized = normalize_stock_prices(df)
    print(
        "[risk_features] Loaded "
        f"{len(normalized):,} rows, "
        f"{normalized['symbol'].nunique():,} symbols, "
        f"{normalized['trading_date'].min().date()} -> "
        f"{normalized['trading_date'].max().date()}"
    )
    return normalized


def load_stock_prices(
    client: Any | None = None,
    csv_path: Path | str = DEFAULT_INPUT_CSV,
    database: str = "stock",
    table: str = "dw_stock_prices",
):
    """Load prices from CSV by default, or from ClickHouse when client is given."""
    if client is None:
        return load_stock_prices_csv(csv_path)

    query = f"""
        SELECT
            trading_date,
            symbol,
            open,
            high,
            low,
            close,
            volume
        FROM {database}.{table}
        ORDER BY symbol, trading_date
    """
    print(f"[risk_features] Loading stock prices from ClickHouse: {database}.{table}")
    return normalize_stock_prices(client.query_df(query))


def create_risk_features(df: pd.DataFrame):
    """Create historical-only features and the future 5-session risk target."""
    features = normalize_stock_prices(df)
    if features.empty:
        print("[risk_features] No rows available to create risk features.")
        for column in [
            *FEATURE_COLUMNS,
            "future_close_5d",
            "target_date",
            "future_return_5d",
            "risk_drop_label",
            "created_at",
        ]:
            if column not in features.columns:
                features[column] = pd.NA
        return features

    group = features.groupby("symbol", group_keys=False)

    for window in [1, 3, 5, 10, 20]:
        features[f"return_{window}d"] = group["close"].pct_change(window)

    for window in [5, 20, 50]:
        features[f"ma_{window}"] = group["close"].transform(
            lambda series, w=window: series.rolling(window=w, min_periods=w).mean()
        )

    features["price_vs_ma20"] = _safe_divide(
        features["close"], features["ma_20"]
    ) - 1
    features["ma5_vs_ma20"] = _safe_divide(features["ma_5"], features["ma_20"]) - 1

    features["volatility_5d"] = group["return_1d"].transform(
        lambda series: series.rolling(window=5, min_periods=5).std()
    )
    features["volatility_20d"] = group["return_1d"].transform(
        lambda series: series.rolling(window=20, min_periods=20).std()
    )
    features["volatility_change"] = _safe_divide(
        features["volatility_5d"], features["volatility_20d"]
    ) - 1

    features["rolling_max_20d"] = group["close"].transform(
        lambda series: series.rolling(window=20, min_periods=20).max()
    )
    features["drawdown_20d"] = _safe_divide(
        features["close"], features["rolling_max_20d"]
    ) - 1

    features["volume_ma_5"] = group["volume"].transform(
        lambda series: series.rolling(window=5, min_periods=5).mean()
    )
    features["volume_ma_20"] = group["volume"].transform(
        lambda series: series.rolling(window=20, min_periods=20).mean()
    )
    features["volume_ratio_5_20"] = _safe_divide(
        features["volume_ma_5"], features["volume_ma_20"]
    )
    features["volume_change_1d"] = group["volume"].pct_change(1)

    high_low_range = features["high"] - features["low"]
    features["daily_range"] = _safe_divide(high_low_range, features["close"])
    features["body_ratio"] = np.where(
        high_low_range.ne(0),
        (features["close"] - features["open"]).abs() / high_low_range,
        0.0,
    )
    features["close_position"] = np.where(
        high_low_range.ne(0),
        (features["close"] - features["low"]) / high_low_range,
        0.5,
    )

    features["future_close_5d"] = group["close"].shift(-5)
    features["target_date"] = group["trading_date"].shift(-5)
    features["future_return_5d"] = _safe_divide(
        features["future_close_5d"], features["close"]
    ) - 1
    features["risk_drop_label"] = (
        features["future_return_5d"] <= -0.05
    ).astype("Int64")
    features.loc[features["future_return_5d"].isna(), "risk_drop_label"] = pd.NA

    features = features.replace([np.inf, -np.inf], np.nan)
    features["created_at"] = pd.Timestamp.now().floor("s")

    print(
        "[risk_features] Created features. "
        f"Rows={len(features):,}, "
        f"trainable_rows={features.dropna(subset=FEATURE_COLUMNS + ['risk_drop_label']).shape[0]:,}"
    )
    return features


def save_risk_features_csv(
    df: pd.DataFrame,
    output_path: Path | str = DEFAULT_FEATURE_CSV,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing_columns = sorted(set(FEATURE_TABLE_COLUMNS) - set(df.columns))
    if missing_columns:
        print(f"[risk_features] Missing feature columns, filling nulls: {missing_columns}")
        for column in missing_columns:
            df[column] = pd.NA

    output = df[FEATURE_TABLE_COLUMNS].copy()
    output["trading_date"] = pd.to_datetime(output["trading_date"]).dt.date
    output.to_csv(output_path, index=False)
    print(f"[risk_features] Saved feature CSV: {output_path} ({len(output):,} rows)")
    return output_path


def create_clickhouse_feature_table(
    client: Any,
    database: str = "stock",
    table: str = "dw_stock_risk_features",
):
    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {database}.{table}
        (
            trading_date Date,
            symbol String,
            open Float64,
            high Float64,
            low Float64,
            close Float64,
            volume Float64,
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
            future_return_5d Nullable(Float64),
            risk_drop_label Nullable(UInt8),
            created_at DateTime
        )
        ENGINE = MergeTree
        ORDER BY (symbol, trading_date)
        """
    )


def save_risk_features(
    client: Any,
    df: pd.DataFrame,
    database: str = "stock",
    table: str = "dw_stock_risk_features",
):
    """Future ClickHouse writer. Not used by the current local CSV pipeline."""
    create_clickhouse_feature_table(client, database=database, table=table)
    output = df[FEATURE_TABLE_COLUMNS].copy()
    output["trading_date"] = pd.to_datetime(output["trading_date"]).dt.date
    output["created_at"] = pd.to_datetime(output["created_at"])
    output = output.where(pd.notna(output), None)
    client.insert_df(table=f"{database}.{table}", df=output)
    print(f"[risk_features] Inserted {len(output):,} rows into {database}.{table}")
