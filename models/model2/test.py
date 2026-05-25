import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from load_data import StockDataLoader


MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_PATH = MODEL_DIR / "future_return_lgbm.pkl"


# =========================
# LOAD DATA
# =========================

df = StockDataLoader().load_data()

df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")

df = df.dropna(subset=["trading_date", "symbol", "close"])

df = df.sort_values(["symbol", "trading_date"]).reset_index(drop=True)


# =========================
# CREATE TARGET BY SYMBOL
# =========================

df["future_close_5d"] = (
    df.groupby("symbol")["close"]
    .shift(-5)
)

df["future_return_5d"] = (
    df["future_close_5d"] / df["close"] - 1
)


# =========================
# SYMBOL ENCODING
# =========================

df["symbol_encoded"] = df["symbol"].astype("category").cat.codes

symbol_mapping = dict(
    zip(
        df["symbol"].astype("category").cat.categories,
        range(len(df["symbol"].astype("category").cat.categories))
    )
)


# =========================
# FEATURE COLUMNS
# =========================

feature_cols = [
    "encode_sector",
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

    "symbol_encoded"
]

target_col = "future_return_5d"


# =========================
# CLEAN DATA
# =========================

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)


# =========================
# WALK-FORWARD VALIDATION
# =========================

folds = [
    ("2020-01-01", "2021-01-01"),
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
]

results = []

for fold, (start_date, end_date) in enumerate(folds, start=1):
    print("=" * 60)
    print(f"FOLD {fold}: train < {start_date}, test {start_date} -> {end_date}")

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    train_mask = df["trading_date"] < start_dt
    test_mask = (
        (df["trading_date"] >= start_dt) &
        (df["trading_date"] < end_dt)
    )

    X_train = df.loc[train_mask, feature_cols].copy()
    y_train = df.loc[train_mask, target_col].copy()

    X_test = df.loc[test_mask, feature_cols].copy()
    y_test = df.loc[test_mask, target_col].copy()

    if len(X_train) == 0 or len(X_test) == 0:
        print("Skip fold because no data")
        continue

    medians = X_train.median(numeric_only=True)

    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    model = LGBMRegressor(
        objective="regression",
        n_estimators=800,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    direction_accuracy = np.mean(
        np.sign(y_test) == np.sign(y_pred)
    )

    result = {
        "fold": fold,
        "test_start": start_date,
        "test_end": end_date,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "direction_accuracy": direction_accuracy,
        "train_size": len(X_train),
        "test_size": len(X_test)
    }

    results.append(result)

    print(result)


# =========================
# VALIDATION SUMMARY
# =========================

print("=" * 60)
print("WALK-FORWARD RESULT")

result_df = pd.DataFrame(results)
print(result_df)

print("=" * 60)
print("MEAN RESULT")
print(result_df[["mae", "rmse", "r2", "direction_accuracy"]].mean())


# =========================
# FINAL TRAIN
# =========================

print("=" * 60)
print("TRAIN FINAL MODEL")

train_df = df[df["trading_date"] < "2024-01-01"].copy()
test_df = df[df["trading_date"] >= "2024-01-01"].copy()

X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

X_test = test_df[feature_cols].copy()
y_test = test_df[target_col].copy()

medians = X_train.median(numeric_only=True)

X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

final_model = LGBMRegressor(
    objective="regression",
    n_estimators=800,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

direction_accuracy = np.mean(
    np.sign(y_test) == np.sign(y_pred)
)

print("=" * 60)
print("FINAL MODEL EVALUATION")
print(f"MAE                : {mae:.6f}")
print(f"RMSE               : {rmse:.6f}")
print(f"R2                 : {r2:.6f}")
print(f"DIRECTION ACCURACY : {direction_accuracy:.4f}")


# =========================
# FEATURE IMPORTANCE
# =========================

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": final_model.feature_importances_
}).sort_values("importance", ascending=False)

print("=" * 60)
print("TOP 20 FEATURE IMPORTANCE")
print(importance_df.head(20))


# =========================
# SAVE MODEL
# =========================

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": final_model,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "medians": medians,
        "symbol_mapping": symbol_mapping,
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "direction_accuracy": direction_accuracy
        }
    },
    MODEL_PATH
)

print("=" * 60)
print(f"MODEL SAVED: {MODEL_PATH}")


# =========================
# PREDICT LATEST EACH SYMBOL
# =========================

print("=" * 60)
print("PREDICT LATEST EACH SYMBOL")

latest_df = (
    df.sort_values(["symbol", "trading_date"])
    .groupby("symbol")
    .tail(1)
    .copy()
)

X_latest = latest_df[feature_cols].copy()
X_latest = X_latest.fillna(medians)

latest_df["predicted_future_return_5d"] = final_model.predict(X_latest)

latest_df["predicted_future_return_5d_percent"] = (
    latest_df["predicted_future_return_5d"] * 100
)

latest_df["direction"] = np.where(
    latest_df["predicted_future_return_5d"] > 0,
    "INCREASE",
    "DECREASE"
)

print(
    latest_df[
        [
            "symbol",
            "trading_date",
            "close",
            "predicted_future_return_5d_percent",
            "direction"
        ]
    ].head(20)
)
