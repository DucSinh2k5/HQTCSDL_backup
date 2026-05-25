from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = PROJECT_ROOT / "data" / "clean" / "features_all.csv"

CLICKHOUSE_HOST = "zmbwqe05t3.ap-southeast-1.aws.clickhouse.cloud"
CLICKHOUSE_PORT = 8443
CLICKHOUSE_USER = "default"
CLICKHOUSE_PASSWORD = "BiHI92y_rbkgT"
CLICKHOUSE_DATABASE = "stock"
CLICKHOUSE_TABLE = "features_all"
CLICKHOUSE_SECURE = True

MODEL_PATH = "models/price_forecasting_xgb.pkl"

METRICS_PATH = "reports/metrics.json"
PREDICTION_PATH = "reports/predictions.csv"
PREDICTION_ACCURACY_PATH = "reports/prediction_accuracy.csv"
FEATURE_IMPORTANCE_PATH = "reports/feature_importance.csv"
BACKTEST_PATH = "reports/backtest.csv"
BACKTEST_METRICS_PATH = "reports/backtest_metrics.json"
BACKTEST_SWEEP_PATH = "reports/backtest_sweep.csv"
WALK_FORWARD_PREDICTION_PATH = "reports/walk_forward_predictions.csv"
WALK_FORWARD_FOLD_METRICS_PATH = "reports/walk_forward_fold_metrics.csv"
WALK_FORWARD_BACKTEST_PATH = "reports/walk_forward_backtest.csv"
WALK_FORWARD_BACKTEST_METRICS_PATH = "reports/walk_forward_backtest_metrics.json"

HORIZON = 5

TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
EARLY_STOPPING_ROUNDS = 50
BACKTEST_TOP_K = 10
BACKTEST_MIN_VOLUME = 100000
BACKTEST_MIN_CLOSE = 5.0
BACKTEST_MIN_PREDICTED_RETURN = 0.004
BACKTEST_TOP_K_VALUES = [5, 10, 20]
BACKTEST_MIN_VOLUME_VALUES = [50000, 100000, 200000]
BACKTEST_MIN_CLOSE_VALUES = [5.0, 10.0]
BACKTEST_MIN_PREDICTED_RETURN_VALUES = [0.0, 0.004, 0.01, 0.02]
TRANSACTION_COST_RATE = 0.001
SLIPPAGE_RATE = 0.001
MAX_ABS_TARGET_RETURN = 0.2
WALK_FORWARD_INITIAL_TRAIN_RATIO = 0.5
WALK_FORWARD_VALIDATION_RATIO = 0.1
WALK_FORWARD_TEST_RATIO = 0.1
WALK_FORWARD_STEP_RATIO = 0.1
WALK_FORWARD_MAX_FOLDS = 4

FEATURES = [
    "encode_sector",

    "open", "high", "low", "close", "volume",

    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",

    "ma_5", "ma_20", "ma_50", "price_vs_ma20", "ma5_vs_ma20",

    "volume_ma_5", "volume_ma_20", "volume_ratio_5_20", "volume_change_1d",

    "volatility_5d", "volatility_20d", "volatility_change",

    "rolling_max_20d", "drawdown_20d",

    "daily_range", "body_ratio", "close_position"
]

XGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "random_state": 42,
    "n_jobs": -1
}
