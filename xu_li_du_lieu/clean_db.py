import re
from pathlib import Path

import numpy as np
import pandas as pd


VALID_SYMBOL_RE = re.compile(r"^[A-Z]{1,5}$")
INPUT_PATH = Path("data/clean/Data_500_stocks_cleaned.csv")
OUTPUT_PATH = Path("data/clean/Data_500_stocks_cleaned.csv")


def clean_symbol(series):
    cleaned = series.astype("string").str.strip().str.upper()
    cleaned = cleaned.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "NULL": pd.NA})
    invalid = ~cleaned.str.match(VALID_SYMBOL_RE, na=False)
    cleaned = cleaned.mask(invalid, pd.NA)
    return cleaned.ffill()


def trim_float_columns(df, numeric_cols):
    for col in numeric_cols:
        series = df[col]
        non_null = series.notna()
        if not non_null.any():
            continue
        frac = np.modf(series[non_null].to_numpy(dtype=float))[0]
        round_idx = series[non_null].index[~np.isclose(frac, 0.0)]
        df.loc[round_idx, col] = series.loc[round_idx].round(2)
    return df


def main():
    df = pd.read_csv(INPUT_PATH)

    if "symbol" not in df.columns:
        raise ValueError("Missing 'symbol' column in input CSV")

    df["symbol"] = clean_symbol(df["symbol"])

    numeric_cols = [c for c in df.columns if c not in {"time", "date", "symbol"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    group_means = df.groupby("symbol")[numeric_cols].transform("mean")
    df[numeric_cols] = df[numeric_cols].fillna(group_means)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    df = trim_float_columns(df, numeric_cols)

    if "date" not in df.columns:
        raise ValueError("Missing 'date' column in input CSV")

    df = df.drop_duplicates(subset=["symbol", "date"], keep="first")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)
    


if __name__ == "__main__":
    main()
