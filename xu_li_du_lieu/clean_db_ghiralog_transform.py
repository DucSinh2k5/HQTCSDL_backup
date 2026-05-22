import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# Require at least one letter to exclude numeric-only symbols like "123".
VALID_SYMBOL_RE = re.compile(r"^(?=.*[A-Z])[A-Z0-9]+$")
INPUT_PATH = Path(
    r"F:\Documents\CODE\Python\cv_project\stock\HQTCSDL_stocks\data\dirty\Data_500_stocks_dirty.csv"
)
OUTPUT_PATH = Path(
    r"F:\Documents\CODE\Python\cv_project\stock\HQTCSDL_stocks\data\clean\Data_500_stocks_clean_ver2.csv"
)
LOG_DIR = Path(
    r"F:\Documents\CODE\Python\cv_project\stock\HQTCSDL_stocks\data\clean_log"
)

COLUMN_ORDER = ["symbol", "date", "open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]
NUMERIC_COLUMNS = PRICE_COLUMNS + ["volume"]
DROP_OUTLIERS = True


def clean_symbol(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip().str.upper()
    cleaned = cleaned.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "NULL": pd.NA})
    invalid = ~cleaned.str.match(VALID_SYMBOL_RE, na=False)
    return cleaned.mask(invalid, pd.NA)


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "date" not in df.columns:
        if "time" in df.columns:
            df = df.rename(columns={"time": "date"})
        else:
            raise ValueError("Missing 'date' or 'time' column in input CSV")

    missing = [c for c in COLUMN_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df[COLUMN_ORDER]


def coerce_numeric(df: pd.DataFrame, columns: List[str]) -> None:
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def fill_missing_numeric(df: pd.DataFrame, columns: List[str]) -> None:
    group_means = df.groupby("symbol")[columns].transform("mean")
    df[columns] = df[columns].fillna(group_means)
    df[columns] = df[columns].fillna(df[columns].mean())


def round_numeric(df: pd.DataFrame, columns: List[str], decimals: int) -> None:
    for col in columns:
        series = df[col]
        non_null = series.notna()
        if not non_null.any():
            continue
        frac = np.modf(series[non_null].to_numpy(dtype=float))[0]
        round_idx = series[non_null].index[~np.isclose(frac, 0.0)]
        df.loc[round_idx, col] = series.loc[round_idx].round(decimals)


def save_log_df(df: pd.DataFrame, filename: str) -> None:
    if df.empty:
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_DIR / filename, index=False)


def write_summary(stats: Dict[str, int], rows_in: int, rows_out: int) -> None:
    lines = [
        "clean_summary",
        f"input_path: {INPUT_PATH}",
        f"output_path: {OUTPUT_PATH}",
        f"rows_in: {rows_in}",
        f"rows_out: {rows_out}",
        f"rows_removed_total: {rows_in - rows_out}",
    ]
    for key, value in stats.items():
        lines.append(f"{key}: {value}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "clean_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def drop_rows(
    df: pd.DataFrame, mask: pd.Series, stats: Dict[str, int], key: str
) -> pd.DataFrame:
    count = int(mask.sum())
    stats[key] = count
    if count:
        return df.loc[~mask].copy()
    return df


def drop_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> pd.Series:
    outlier_mask = pd.Series(False, index=df.index)
    for col in columns:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask |= (df[col] < lower) | (df[col] > upper)
    return outlier_mask


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    rows_in = len(df)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    df = ensure_columns(df)

    stats: Dict[str, int] = {}

    df["symbol"] = clean_symbol(df["symbol"])
    invalid_symbol_mask = df["symbol"].isna()
    stats["invalid_symbol"] = int(invalid_symbol_mask.sum())
    save_log_df(df.loc[invalid_symbol_mask], "removed_invalid_symbol.csv")
    df = drop_rows(df, invalid_symbol_mask, stats, "invalid_symbol")

    parsed_date = pd.to_datetime(df["date"], errors="coerce")
    invalid_date_mask = parsed_date.isna()
    stats["invalid_date"] = int(invalid_date_mask.sum())
    save_log_df(df.loc[invalid_date_mask], "removed_invalid_date.csv")
    df = drop_rows(df, invalid_date_mask, stats, "invalid_date")
    df["date"] = parsed_date.loc[df.index]

    raw_numeric = df[NUMERIC_COLUMNS].copy()
    coerce_numeric(df, NUMERIC_COLUMNS)

    invalid_numeric_rows = []
    invalid_numeric_row_mask = pd.Series(False, index=df.index)
    for col in NUMERIC_COLUMNS:
        mask = raw_numeric[col].notna() & df[col].isna()
        if mask.any():
            temp = pd.DataFrame(
                {
                    "symbol": df.loc[mask, "symbol"],
                    "date": df.loc[mask, "date"],
                    "column": col,
                    "raw_value": raw_numeric.loc[mask, col].astype(str),
                }
            )
            invalid_numeric_rows.append(temp)
            invalid_numeric_row_mask |= mask
    if invalid_numeric_rows:
        save_log_df(
            pd.concat(invalid_numeric_rows, ignore_index=True),
            "invalid_numeric_values.csv",
        )
    stats["invalid_numeric_rows"] = int(invalid_numeric_row_mask.sum())

    missing_numeric_before_mask = df[NUMERIC_COLUMNS].isna().any(axis=1)
    stats["missing_numeric_before_fill"] = int(missing_numeric_before_mask.sum())
    save_log_df(df.loc[missing_numeric_before_mask], "missing_numeric_before_fill.csv")

    fill_missing_numeric(df, NUMERIC_COLUMNS)
    missing_numeric_after_mask = df[NUMERIC_COLUMNS].isna().any(axis=1)
    stats["missing_numeric_after_fill"] = int(missing_numeric_after_mask.sum())
    save_log_df(df.loc[missing_numeric_after_mask], "removed_missing_numeric.csv")
    df = drop_rows(df, missing_numeric_after_mask, stats, "missing_numeric")

    non_positive_mask = (df[PRICE_COLUMNS] <= 0).any(axis=1)
    stats["non_positive_prices"] = int(non_positive_mask.sum())
    save_log_df(df.loc[non_positive_mask], "removed_non_positive_prices.csv")
    df = drop_rows(df, non_positive_mask, stats, "non_positive_prices")

    ohlc_invalid_mask = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )
    stats["invalid_ohlc"] = int(ohlc_invalid_mask.sum())
    save_log_df(df.loc[ohlc_invalid_mask], "removed_invalid_ohlc.csv")
    df = drop_rows(df, ohlc_invalid_mask, stats, "invalid_ohlc")

    negative_volume_mask = df["volume"].notna() & (df["volume"] < 0)
    stats["negative_volume"] = int(negative_volume_mask.sum())
    save_log_df(df.loc[negative_volume_mask], "removed_negative_volume.csv")
    df = drop_rows(df, negative_volume_mask, stats, "negative_volume")

    non_integer_volume_mask = df["volume"].notna() & (
        np.abs(df["volume"] - df["volume"].round()) > 1e-6
    )
    stats["non_integer_volume"] = int(non_integer_volume_mask.sum())
    if stats["non_integer_volume"]:
        rounded_volume = df.loc[non_integer_volume_mask, "volume"].round(0)
        log_df = df.loc[non_integer_volume_mask, ["symbol", "date", "volume"]].copy()
        log_df["volume_rounded"] = rounded_volume
        save_log_df(log_df, "volume_rounded.csv")
        df.loc[non_integer_volume_mask, "volume"] = rounded_volume

    dup_full_mask = df.duplicated(keep=False)
    stats["duplicate_full_rows_total"] = int(dup_full_mask.sum())
    save_log_df(df.loc[dup_full_mask], "duplicate_full_rows.csv")
    pre_len = len(df)
    df = df.drop_duplicates(keep="first")
    stats["duplicate_full_rows_removed"] = pre_len - len(df)

    dup_key_mask = df.duplicated(subset=["symbol", "date"], keep=False)
    stats["duplicate_symbol_date_total"] = int(dup_key_mask.sum())
    save_log_df(df.loc[dup_key_mask], "duplicate_symbol_date.csv")
    pre_len = len(df)
    df = df.drop_duplicates(subset=["symbol", "date"], keep="first")
    stats["duplicate_symbol_date_removed"] = pre_len - len(df)

    if DROP_OUTLIERS:
        outlier_mask = drop_outliers_iqr(df, NUMERIC_COLUMNS)
        stats["outliers_iqr"] = int(outlier_mask.sum())
        save_log_df(df.loc[outlier_mask], "removed_outliers_iqr.csv")
        df = drop_rows(df, outlier_mask, stats, "outliers_iqr")

    round_numeric(df, PRICE_COLUMNS, 2)
    df["volume"] = df["volume"].round(0)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    df = df[COLUMN_ORDER]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    write_summary(stats, rows_in, len(df))

    print(f"Saved clean file: {OUTPUT_PATH}")
    print(f"Saved clean logs: {LOG_DIR}")
    print("Removed rows summary:")
    for key, value in stats.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()