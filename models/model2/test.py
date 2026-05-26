import pandas as pd
import numpy as np
import joblib
import clickhouse_connect

from pathlib import Path
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. CONFIG CLICKHOUSE
# ============================================================

CLICKHOUSE_HOST = "zmbwqe05t3.ap-southeast-1.aws.clickhouse.cloud"
CLICKHOUSE_PORT = 8443
CLICKHOUSE_USER = "default"
CLICKHOUSE_PASSWORD = "BiHI92y_rbkgT"
CLICKHOUSE_DATABASE = "stock"
CLICKHOUSE_TABLE = "features_all"
CLICKHOUSE_SECURE = True


# ============================================================
# 2. CONFIG PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "model2" / "saved_models"
MODEL_PATH = MODEL_DIR / "future_return_lgbm.pkl"


# ============================================================
# 3. LOAD DATA FROM CLICKHOUSE
# ============================================================

client = clickhouse_connect.get_client(
    host=CLICKHOUSE_HOST,
    port=CLICKHOUSE_PORT,
    username=CLICKHOUSE_USER,
    password=CLICKHOUSE_PASSWORD,
    database=CLICKHOUSE_DATABASE,
    secure=CLICKHOUSE_SECURE
)

query = f"""
SELECT *
FROM {CLICKHOUSE_TABLE}
"""

df = client.query_df(query)


# ============================================================
# 4. BASIC CLEANING
# ============================================================

df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")

df = df.dropna(subset=["trading_date", "symbol", "close"])

df = df.sort_values(["symbol", "trading_date"]).reset_index(drop=True)


# ============================================================
# 5. CREATE TARGET BY SYMBOL
# future_return_5d = close sau 5 ngày / close hôm nay - 1
# ============================================================

df["future_close_5d"] = (
    df.groupby("symbol")["close"]
    .shift(-5)
)

df["future_return_5d"] = (
    df["future_close_5d"] / df["close"] - 1
)

target_col = "future_return_5d"


# ============================================================
# 6. ENCODE SYMBOL + SECTOR
# ============================================================

df["symbol_encoded"] = df["symbol"].astype("category").cat.codes

symbol_mapping = dict(
    zip(
        df["symbol"].astype("category").cat.categories,
        range(len(df["symbol"].astype("category").cat.categories))
    )
)

df.rename(
    columns={
        "encode_sector": "sector_encoded"
    },
    inplace=True
)


# ============================================================
# 7. FEATURE COLUMNS
# ============================================================

feature_cols = [
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

    "symbol_encoded",
    "sector_encoded"
]


# ============================================================
# 8. FINAL CLEANING
# ============================================================

df = df.replace([np.inf, -np.inf], np.nan)

df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)


# ============================================================
# 9. EVALUATION FUNCTION
# ============================================================

def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    direction_accuracy = np.mean(
        np.sign(y_true) == np.sign(y_pred)
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "direction_accuracy": direction_accuracy
    }


# ============================================================
# 10. WALK-FORWARD VALIDATION
# Train quá khứ → test tương lai theo từng năm
# ============================================================

folds = [
    ("2020-01-01", "2021-01-01"),
    ("2021-01-01", "2022-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2024-01-01", "2025-01-01"),
]

walk_forward_results = []

print("=" * 70)
print("WALK-FORWARD VALIDATION")
print("=" * 70)

for fold_idx, (start_date, end_date) in enumerate(folds, start=1):
    print("=" * 70)
    print(f"FOLD {fold_idx}: train < {start_date}, test {start_date} -> {end_date}")

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    train_df = df[df["trading_date"] < start_dt].copy()

    test_df = df[
        (df["trading_date"] >= start_dt) &
        (df["trading_date"] < end_dt)
    ].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        print("Skip fold because train/test is empty")
        continue

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_col].copy()

    medians = X_train.median(numeric_only=True)

    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    model = LGBMRegressor(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=8,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[
            early_stopping(stopping_rounds=50),
            log_evaluation(period=0)
        ]
    )

    y_pred = model.predict(X_test)

    metrics = evaluate_regression(y_test, y_pred)

    result = {
        "fold": fold_idx,
        "test_start": start_date,
        "test_end": end_date,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "best_iteration": model.best_iteration_,
        **metrics
    }

    walk_forward_results.append(result)

    print(f"Train size         : {len(train_df)}")
    print(f"Test size          : {len(test_df)}")
    print(f"Best iteration     : {model.best_iteration_}")
    print(f"MAE                : {metrics['mae']:.6f}")
    print(f"RMSE               : {metrics['rmse']:.6f}")
    print(f"R2                 : {metrics['r2']:.6f}")
    print(f"Direction Accuracy : {metrics['direction_accuracy']:.4f}")


# ============================================================
# 11. WALK-FORWARD SUMMARY
# ============================================================

result_df = pd.DataFrame(walk_forward_results)

print("=" * 70)
print("WALK-FORWARD SUMMARY")
print("=" * 70)

print(result_df)

print("=" * 70)
print("MEAN RESULT")
print(
    result_df[
        ["mae", "rmse", "r2", "direction_accuracy"]
    ].mean()
)


# ============================================================
# 12. TRAIN FINAL MODEL
# Sau khi đánh giá bằng walk-forward, train model cuối cùng
# ============================================================

print("=" * 70)
print("TRAIN FINAL MODEL")
print("=" * 70)

FINAL_TEST_START_DATE = "2024-01-01"

train_df = df[df["trading_date"] < FINAL_TEST_START_DATE].copy()
test_df = df[df["trading_date"] >= FINAL_TEST_START_DATE].copy()

X_train = train_df[feature_cols].copy()
y_train = train_df[target_col].copy()

X_test = test_df[feature_cols].copy()
y_test = test_df[target_col].copy()

medians = X_train.median(numeric_only=True)

X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

final_model = LGBMRegressor(
    objective="regression",
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=8,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)

final_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="rmse",
    callbacks=[
        early_stopping(stopping_rounds=50),
        log_evaluation(period=100)
    ]
)

y_pred = final_model.predict(X_test)

final_metrics = evaluate_regression(y_test, y_pred)

print("=" * 70)
print("FINAL MODEL EVALUATION")
print("=" * 70)
print(f"MAE                : {final_metrics['mae']:.6f}")
print(f"RMSE               : {final_metrics['rmse']:.6f}")
print(f"R2                 : {final_metrics['r2']:.6f}")
print(f"Direction Accuracy : {final_metrics['direction_accuracy']:.4f}")


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": final_model.feature_importances_
}).sort_values("importance", ascending=False)

print("=" * 70)
print("TOP 20 FEATURE IMPORTANCE")
print("=" * 70)
print(importance_df.head(20))


# ============================================================
# 14. SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": final_model,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "medians": medians,
        "symbol_mapping": symbol_mapping,
        "walk_forward_results": walk_forward_results,
        "final_metrics": final_metrics
    },
    MODEL_PATH
)

print("=" * 70)
print(f"MODEL SAVED: {MODEL_PATH}")
print("=" * 70)