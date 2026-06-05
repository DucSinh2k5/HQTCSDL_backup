import json
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(
    model,
    X_test,
    test_df,
    target_type="future_close_price",
    return_calibrator=None,
):
    result_df = test_df.copy()

    raw_predictions = np.asarray(model.predict(X_test))

    if target_type == "future_return":
        result_df["raw_predicted_return"] = raw_predictions
        if return_calibrator is not None:
            predicted_return = np.asarray(return_calibrator.predict(raw_predictions))
        else:
            predicted_return = raw_predictions
        if len(predicted_return) != len(result_df):
            raise ValueError("predicted return length must match test rows")
        result_df["predicted_return"] = predicted_return
        result_df["predicted_close"] = result_df["close"] * (
            1 + result_df["predicted_return"]
        )
    elif target_type == "future_close_price":
        result_df["predicted_close"] = raw_predictions
        result_df["predicted_return"] = result_df["predicted_close"] / result_df["close"] - 1
    else:
        raise ValueError("target_type must be future_close_price or future_return")

    result_df["predicted_future_close"] = result_df["predicted_close"]

    if "target_close" not in result_df.columns:
        result_df["target_close"] = result_df["future_close"]

    if "target_return" not in result_df.columns:
        result_df["target_return"] = result_df["target_close"] / result_df["close"] - 1

    y_true = result_df["target_close"]
    y_pred = result_df["predicted_close"]
    y_true_return = result_df["target_return"]
    y_pred_return = result_df["predicted_return"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return_mae = mean_absolute_error(y_true_return, y_pred_return)
    return_rmse = np.sqrt(mean_squared_error(y_true_return, y_pred_return))

    result_df["actual_direction"] = np.where(result_df["target_return"] > 0, 1, 0)
    result_df["predicted_direction"] = np.where(result_df["predicted_return"] > 0, 1, 0)

    directional_accuracy = (
        result_df["actual_direction"] == result_df["predicted_direction"]
    ).mean() * 100

    baseline_pred = result_df["close"]

    baseline_mae = mean_absolute_error(y_true, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))
    baseline_mape = np.mean(np.abs((y_true - baseline_pred) / y_true)) * 100
    baseline_return_pred = np.zeros(len(result_df))
    baseline_return_mae = mean_absolute_error(y_true_return, baseline_return_pred)
    baseline_return_rmse = np.sqrt(
        mean_squared_error(y_true_return, baseline_return_pred)
    )
    baseline_direction = np.zeros(len(result_df), dtype=int)
    baseline_directional_accuracy = (
        result_df["actual_direction"].to_numpy() == baseline_direction
    ).mean() * 100

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Return_MAE": return_mae,
        "Return_RMSE": return_rmse,
        "Directional_Accuracy": directional_accuracy,
        "Baseline_MAE": baseline_mae,
        "Baseline_RMSE": baseline_rmse,
        "Baseline_MAPE": baseline_mape,
        "Baseline_Return_MAE": baseline_return_mae,
        "Baseline_Return_RMSE": baseline_return_rmse,
        "Baseline_Directional_Accuracy": baseline_directional_accuracy,
        "Return_MAE_Improvement": baseline_return_mae - return_mae,
        "Return_RMSE_Improvement": baseline_return_rmse - return_rmse,
        "Directional_Accuracy_Over_Baseline": (
            directional_accuracy - baseline_directional_accuracy
        ),
        "Beats_Baseline_Return_MAE": bool(return_mae < baseline_return_mae),
    }

    return metrics, result_df


def build_prediction_accuracy_table(result_df):
    required_columns = [
        "future_trading_date",
        "symbol",
        "target_close",
        "predicted_close",
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
            "real_close": result_df["target_close"],
            "predict_close": result_df["predicted_close"],
        }
    )

    accuracy_df["error_pct"] = (
        (accuracy_df["predict_close"] - accuracy_df["real_close"]).abs()
        / accuracy_df["real_close"].abs()
        * 100
    )
    accuracy_df["accuracy_pct"] = (100 - accuracy_df["error_pct"]).clip(lower=0)

    return accuracy_df[
        ["date", "symbol", "real_close", "predict_close", "accuracy_pct", "error_pct"]
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
