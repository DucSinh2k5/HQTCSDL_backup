import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from config1 import {
#     LIGHTGBM_PARAMS,
# }

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RISK_FEATURES_CSV = (
    PROJECT_ROOT / "models" / "model5" / "output_model5" / "risk_features.csv"
)

train = pd.read_csv(RISK_FEATURES_CSV)
target_col = "future_return_5d"
train = train.replace([np.inf, -np.inf], np.nan)
train = train.dropna(subset=[target_col]).reset_index(drop=True)
# for i in train.columns:
#     print(i)

train["trading_date"] = pd.to_datetime(train["trading_date"], errors="coerce")
train = train.dropna(subset=["trading_date"]).sort_values("trading_date").reset_index(drop=True)

drop_cols = ["trading_date", "symbol", target_col, "created_at"]
feature_cols = [col for col in train.columns if col not in drop_cols]

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

    train_mask = train["trading_date"] < start_dt
    val_mask = (train["trading_date"] >= start_dt) & (train["trading_date"] < end_dt)

    if not train_mask.any() or not val_mask.any():
        print(f"Fold {fold} - skipped (no data in range)")
        continue

    X_train_fold = train.loc[train_mask, feature_cols].copy()
    y_train_fold = train.loc[train_mask, target_col]
    X_val_fold = train.loc[val_mask, feature_cols].copy()
    y_val_fold = train.loc[val_mask, target_col]

    numeric_cols = X_train_fold.select_dtypes(include=[np.number]).columns
    medians = X_train_fold[numeric_cols].median()
    X_train_fold[numeric_cols] = X_train_fold[numeric_cols].fillna(medians)
    X_val_fold[numeric_cols] = X_val_fold[numeric_cols].fillna(medians)
    # chuyen tham so duoi day thanh tham so cho xgboost: 
    # "objective": "regression",
    # "metric": "rmse",
    # "boosting_type": "gbdt",
    # "n_estimators": 700,
    # "learning_rate": 0.03,
    # "max_depth": 6,
    # "num_leaves": 31,
    # "subsample": 0.8,
    # "colsample_bytree": 0.8,
    # "random_state": 42,
    # "verbose": -1

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
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            early_stopping_rounds=50,
            verbose=False,
        )
    except TypeError:
        # Fallback for older xgboost versions without early_stopping_rounds.
        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False,
        )

    val_pred = model.predict(X_val_fold)
    mse = mean_squared_error(y_val_fold, val_pred)
    r2 = r2_score(y_val_fold, val_pred)
    mae = mean_absolute_error(y_val_fold, val_pred)
    print(f"Fold {fold + 1} - MSE: {mse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")
