import clickhouse_connect
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from config1 import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_SECURE,
    CLICKHOUSE_TABLE,
    CLICKHOUSE_USER,
    DATE_COL,
    FEATURE_COLUMNS,
    INTERACTION_FEATURES,
    LAG_DAYS,
    LAG_FEATURE_COLUMNS,
    LAG_SOURCE_COLUMNS,
    SYMBOL_COL,
    TARGET_COL,
    TARGET_MAX,
    TARGET_MIN,
)


GENERATED_COLUMNS = set(["symbol_encoded"] + INTERACTION_FEATURES + LAG_FEATURE_COLUMNS)
LOAD_START_YEAR = 2015
LOAD_END_YEAR = 2027


def quote_identifier(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def unique_columns(columns: list[str]) -> list[str]:
    result = []
    seen = set()
    for column in columns:
        if column not in seen:
            result.append(column)
            seen.add(column)
    return result


def source_columns() -> list[str]:
    table_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in GENERATED_COLUMNS
    ]
    return unique_columns([DATE_COL, SYMBOL_COL, "close"] + table_features)


def load_data() -> pd.DataFrame:
    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE,
        secure=CLICKHOUSE_SECURE,
    )

    columns = ", ".join(quote_identifier(column) for column in source_columns())
    table = f"{quote_identifier(CLICKHOUSE_DATABASE)}.{quote_identifier(CLICKHOUSE_TABLE)}"
    date_col = quote_identifier(DATE_COL)
    symbol_col = quote_identifier(SYMBOL_COL)

    print("[model2.1] Loading features from ClickHouse...")
    frames = []

    for year in range(LOAD_START_YEAR, LOAD_END_YEAR):
        query = f"""
        SELECT {columns}
        FROM {table}
        WHERE {date_col} >= toDate('{year}-01-01')
          AND {date_col} < toDate('{year + 1}-01-01')
        ORDER BY {symbol_col}, {date_col}
        """

        part = client.query_df(query)
        print(f"[model2.1] {year}: {len(part):,} rows")

        if not part.empty:
            frames.append(part)

    if not frames:
        raise ValueError(f"No rows loaded from {table}")

    df = pd.concat(frames, ignore_index=True)
    print(f"[model2.1] Total loaded rows: {len(df):,}")
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(set(source_columns()) - set(df.columns))
    if missing_columns:
        raise KeyError(f"Missing columns in {CLICKHOUSE_TABLE}: {missing_columns}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, SYMBOL_COL, "close"])
    df = df.sort_values([SYMBOL_COL, DATE_COL]).reset_index(drop=True)

    df[TARGET_COL] = df.groupby(SYMBOL_COL)["close"].shift(-5) / df["close"] - 1
    df["symbol_encoded"] = pd.factorize(df[SYMBOL_COL].astype(str), sort=True)[0].astype(
        np.int32
    )

    for column in LAG_SOURCE_COLUMNS:
        for lag in LAG_DAYS:
            df[f"{column}_lag{lag}"] = df.groupby(SYMBOL_COL)[column].shift(lag)

    df["volume_volatility_interaction"] = (
        df["volume_ratio_5_20"] * df["volatility_5d"]
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[df[TARGET_COL].between(TARGET_MIN, TARGET_MAX)].copy()

    for column in FEATURE_COLUMNS + [TARGET_COL]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COL]).reset_index(drop=True)

    print(f"[model2.1] Prepared rows: {len(df):,}")
    print(f"[model2.1] Feature count: {len(FEATURE_COLUMNS)}")
    return df


def train_and_evaluate(train: pd.DataFrame) -> None:
    folds = [
        ("2020-01-01", "2021-01-01"),
        ("2021-01-01", "2022-01-01"),
        ("2022-01-01", "2023-01-01"),
        ("2023-01-01", "2024-01-01"),
        ("2024-01-01", "2025-01-01"),
    ]

    for fold, (start_date, end_date) in enumerate(folds, start=1):
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        train_mask = train[DATE_COL] < start_dt
        val_mask = (train[DATE_COL] >= start_dt) & (train[DATE_COL] < end_dt)

        if not train_mask.any() or not val_mask.any():
            print(f"Fold {fold} - skipped (no data in range)")
            continue

        X_train = train.loc[train_mask, FEATURE_COLUMNS].copy()
        y_train = train.loc[train_mask, TARGET_COL]
        X_val = train.loc[val_mask, FEATURE_COLUMNS].copy()
        y_val = train.loc[val_mask, TARGET_COL]

        medians = X_train.median(numeric_only=True)
        X_train = X_train.fillna(medians)
        X_val = X_val.fillna(medians)

        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            booster="gbtree",
            n_estimators=700,
            learning_rate=0.03,
            max_depth=6,
            max_leaves=31,
            grow_policy="lossguide",
            tree_method="hist",
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )

        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False,
            )
        except TypeError:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        pred = model.predict(X_val)
        mse = mean_squared_error(y_val, pred)
        r2 = r2_score(y_val, pred)
        mae = mean_absolute_error(y_val, pred)

        print(f"Fold {fold} - MSE: {mse:.6f}, R2: {r2:.6f}, MAE: {mae:.6f}")


def main() -> None:
    df = load_data()
    train = prepare_data(df)
    train_and_evaluate(train)


if __name__ == "__main__":
    main()
