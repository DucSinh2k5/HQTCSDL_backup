import os

# =========================
# CLICKHOUSE CONFIG
# =========================
CLICKHOUSE_HOST = "zmbwqe05t3.ap-southeast-1.aws.clickhouse.cloud"
CLICKHOUSE_PORT = 8443
CLICKHOUSE_USER = "default"
CLICKHOUSE_PASSWORD = "BiHI92y_rbkgT"
CLICKHOUSE_DATABASE = "stock"
CLICKHOUSE_TABLE = "features_all"
CLICKHOUSE_SECURE = True

# =========================
# COLUMN CONFIG
# =========================
DATE_COL = "trading_date"
SYMBOL_COL = "symbol"
TARGET_COL = "future_return_5d"

# =========================
# TRAIN / TEST
# =========================
TEST_START_DATE = "2024-01-01"

# =========================
# MODEL PATH
# =========================
BASE_DIR = os.path.dirname("HQTCSDL_stocks/models/model2")

MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

LIGHTGBM_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "future_return_lgbm.pkl"
)

SYMBOL_ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "symbol_encoder.pkl"
)

# =========================
# OUTLIER
# =========================
TARGET_MIN = -0.25
TARGET_MAX = 0.25

# =========================
# BASE FEATURES
# =========================
BASE_FEATURE_COLUMNS = [
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
    "close_position"
]

# =========================
# LAG CONFIG
# =========================
LAG_DAYS = [1, 2, 3, 5]

LAG_SOURCE_COLUMNS = [
    "return_1d",
    "volume_change_1d",
    "volatility_5d"
]

LAG_FEATURE_COLUMNS = [
    f"{col}_lag{lag}"
    for col in LAG_SOURCE_COLUMNS
    for lag in LAG_DAYS
]

# =========================
# INTERACTION FEATURES
# =========================
INTERACTION_FEATURES = [
    "volume_volatility_interaction"
]

# =========================
# FINAL FEATURE COLUMNS
# =========================
FEATURE_COLUMNS = (
    BASE_FEATURE_COLUMNS
    + INTERACTION_FEATURES
    + LAG_FEATURE_COLUMNS
    + ["symbol_encoded"]
)

# =========================
# LIGHTGBM PARAMS
# =========================
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 700,
    "learning_rate": 0.03,
    "max_depth": 6,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1
}