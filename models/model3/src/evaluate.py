import json
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


SIGNAL_LABELS = {
    0: "SELL",
    1: "HOLD",
    2: "BUY",
}


def evaluate_model(model, X_test, test_df, signal_labels=None):
    signal_labels = signal_labels or SIGNAL_LABELS
    result_df = test_df.copy()

    pred_label = model.predict(X_test).astype(int)
    pred_proba = model.predict_proba(X_test)

    if "target_return" not in result_df.columns:
        result_df["target_return"] = result_df["target_close"] / result_df["close"] - 1

    result_df["predicted_signal_label"] = pred_label
    result_df["predicted_signal"] = result_df["predicted_signal_label"].map(signal_labels)

    for label_id, label_name in signal_labels.items():
        result_df[f"{label_name.lower()}_probability"] = pred_proba[:, label_id]

    result_df["predicted_signal_score"] = (
        result_df["buy_probability"] - result_df["sell_probability"]
    )

    y_true = result_df["target_signal_label"].astype(int)
    y_pred = result_df["predicted_signal_label"].astype(int)

    accuracy = accuracy_score(y_true, y_pred) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(signal_labels.keys()),
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(signal_labels.keys()))

    baseline_label = int(y_true.mode().iloc[0])
    baseline_pred = np.full(len(y_true), baseline_label)
    baseline_accuracy = accuracy_score(y_true, baseline_pred) * 100

    metrics = {
        "Accuracy": accuracy,
        "Macro_Precision": precision,
        "Macro_Recall": recall,
        "Macro_F1": f1,
        "Baseline_Accuracy": baseline_accuracy,
        "Baseline_Label": signal_labels[baseline_label],
        "Confusion_Matrix": cm.tolist(),
    }

    return metrics, result_df


def build_prediction_accuracy_table(result_df):
    required_columns = [
        "future_trading_date",
        "symbol",
        "target_signal",
        "predicted_signal",
    ]
    missing_columns = [col for col in required_columns if col not in result_df.columns]
    if missing_columns:
        raise ValueError(
            "Missing prediction accuracy columns: " + ", ".join(missing_columns)
        )

    accuracy_df = pd.DataFrame(
        {
            "date": pd.to_datetime(result_df["future_trading_date"]),
            "symbol": result_df["symbol"],
            "real_signal": result_df["target_signal"],
            "predict_signal": result_df["predicted_signal"],
        }
    )

    accuracy_df["is_correct"] = (
        accuracy_df["real_signal"] == accuracy_df["predict_signal"]
    )

    return accuracy_df[
        ["date", "symbol", "real_signal", "predict_signal", "is_correct"]
    ]


def save_metrics(metrics, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


def save_feature_importance(model, features, path):
    importance_df = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(path, index=False)
