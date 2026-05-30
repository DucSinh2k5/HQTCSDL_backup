# ==============================================
# MODULE 4 — BENCHMARK OUTPERFORMANCE MODEL
# File: train_benchmark_model.py
# Mục đích: Train LightGBM Classifier
#           Đánh giá mô hình
# ==============================================

import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ==================
# CONFIG
# ==================
FEATURES_CSV  = Path("model4/output/benchmark_features.csv")
MODEL_OUTPUT  = Path("model4/output")
TRAIN_RATIO   = 0.8  # 80% train, 20% test

FEATURE_COLUMNS = [
    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d",
    "ma_5", "ma_20", "ma_50",
    "price_vs_ma20", "ma5_vs_ma20",
    "volatility_5d", "volatility_20d", "volatility_change",
    "rolling_max_20d", "drawdown_20d",
    "volume_ma_5", "volume_ma_20", "volume_ratio_5_20", "volume_change_1d",
    "daily_range", "body_ratio", "close_position",
]

# ==================
# 1. ĐỌC DỮ LIỆU
# ==================
def load_features() -> pd.DataFrame:
    print("[model4] Đọc features từ CSV...")
    df = pd.read_csv(FEATURES_CSV)
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"])
    df = df.sort_values(["trading_date", "symbol"]).reset_index(drop=True)
    print(f"[model4] Đọc xong: {len(df):,} dòng")
    return df

# ==================
# 2. TRAIN/TEST SPLIT
# ==================
def time_split(df: pd.DataFrame):
    """
    Chia train/test theo thời gian
    KHÔNG xáo trộn ngẫu nhiên!
    """
    # Lấy danh sách ngày unique, sắp xếp từ cũ → mới
    unique_dates = sorted(df["trading_date"].unique())
    cutoff_idx   = int(len(unique_dates) * TRAIN_RATIO)
    cutoff_date  = unique_dates[cutoff_idx]

    train_df = df[df["trading_date"] < cutoff_date].copy()
    test_df  = df[df["trading_date"] >= cutoff_date].copy()

    print(f"[model4] Train: {len(train_df):,} dòng "
          f"({train_df['trading_date'].min().date()} "
          f"→ {train_df['trading_date'].max().date()})")
    print(f"[model4] Test:  {len(test_df):,} dòng "
          f"({test_df['trading_date'].min().date()} "
          f"→ {test_df['trading_date'].max().date()})")
    print(f"[model4] Cutoff date: {pd.Timestamp(cutoff_date).date()}")

    return train_df, test_df

# ==================
# 3. TRAIN LIGHTGBM
# ==================
def train_model(train_df: pd.DataFrame):
    """Train LightGBM Classifier"""
    print("\n[model4] Bắt đầu train LightGBM...")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]
    #SỬA
    # Khởi tạo mô hình LightGBM
    model = lgb.LGBMClassifier(
        n_estimators=500,      # số cây
        max_depth=6,           # độ sâu tối đa mỗi cây
        learning_rate=0.03,    # tốc độ học
        subsample=0.8,         # tỷ lệ dữ liệu dùng mỗi cây
        colsample_bytree=0.8,  # tỷ lệ features dùng mỗi cây
        min_child_samples=50,  # thêm mới: tránh overfit
        reg_alpha=0.1,         # thêm mới: L1 regularization
        reg_lambda=0.1,         ## thêm mới: L2 regularization
        random_state=42,       # seed cố định → kết quả tái tạo được
        n_jobs=-1,             # dùng toàn bộ CPU
        verbose=-1,            # tắt log dài dòng
    )
    #SỬA
    model.fit(X_train, y_train)
    print("[model4] Train xong!")
    return model

# ==================
# 4. ĐÁNH GIÁ
# ==================
def evaluate_model(model, test_df: pd.DataFrame):
    """Đánh giá mô hình trên tập test"""
    print("\n[model4] Đánh giá mô hình...")

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"]

    # Dự đoán
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    # Tính các metrics
    metrics = {
        "accuracy":  float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_ratio": TRAIN_RATIO,
        "test_rows": len(test_df),
        "feature_columns": FEATURE_COLUMNS,
    }

    # In kết quả
    print(f"\n{'='*40}")
    print(f"  KẾT QUẢ ĐÁNH GIÁ MODEL4")
    print(f"{'='*40}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"  {metrics['confusion_matrix']}")
    print(f"{'='*40}\n")

    return metrics, y_pred, y_prob

# ==================
# 5. LƯU KẾT QUẢ
# ==================
def save_results(model, metrics, test_df, y_pred, y_prob):
    """Lưu metrics và kết quả dự đoán"""
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Lưu metrics ra JSON
    metrics_path = MODEL_OUTPUT / "benchmark_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[model4] Đã lưu metrics: {metrics_path}")

    # Lưu kết quả dự đoán ra CSV
    predictions = test_df[["symbol", "trading_date",
                            "close", "label"]].copy()
    predictions["predicted_label"]    = y_pred
    predictions["outperform_probability"] = y_prob
    predictions["prediction_correct"] = (
        predictions["label"] == predictions["predicted_label"]
    )

    pred_path = MODEL_OUTPUT / "benchmark_predictions.csv"
    predictions.to_csv(pred_path, index=False)
    print(f"[model4] Đã lưu predictions: {pred_path}")

    # Lưu feature importance
    importance = pd.DataFrame({
        "feature":   FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_path = MODEL_OUTPUT / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    print(f"[model4] Đã lưu feature importance: {importance_path}")

    return pred_path

# ==================
# MAIN
# ==================
if __name__ == "__main__":
    # Bước 1: Đọc features
    df = load_features()

    # Bước 2: Chia train/test theo thời gian
    train_df, test_df = time_split(df)

    # Bước 3: Train LightGBM
    model = train_model(train_df)

    # Bước 4: Đánh giá
    metrics, y_pred, y_prob = evaluate_model(model, test_df)

    # Bước 5: Lưu kết quả
    save_results(model, metrics, test_df, y_pred, y_prob)

    print("\n[model4] HOÀN THÀNH! ✅")