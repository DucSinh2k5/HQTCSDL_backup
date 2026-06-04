import joblib
import pandas as pd
import numpy as np


def load_saved_model(model_path, include_metadata=False):
    saved = joblib.load(model_path)

    model = saved["model"]
    features = saved["features"]

    if include_metadata:
        metadata = {
            "horizon": saved.get("horizon"),
            "target_type": saved.get("target_type", "future_close_price"),
            "return_calibrator": saved.get("return_calibrator"),
        }
        return model, features, metadata

    return model, features


def predict_latest_price(
    df,
    model,
    features,
    target_type="future_close_price",
    return_calibrator=None,
):
    df = df.copy()

    df = df.replace(["NULL", "null", "None", ""], np.nan)

    df["trading_date"] = pd.to_datetime(df["trading_date"])
    df = df.sort_values(["symbol", "trading_date"])

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    latest_df = df.groupby("symbol").tail(1).copy()

    latest_df = latest_df.dropna(subset=features)

    X_latest = latest_df[features]
    raw_predictions = np.asarray(model.predict(X_latest), dtype=float)

    if len(raw_predictions) != len(latest_df):
        raise ValueError("model predictions must match latest rows")

    if target_type == "future_return":
        latest_df["raw_predicted_return"] = raw_predictions
        if return_calibrator is not None:
            predicted_return = return_calibrator.predict(raw_predictions)
            predicted_return = np.asarray(predicted_return, dtype=float)
        else:
            predicted_return = raw_predictions

        if len(predicted_return) != len(latest_df):
            raise ValueError("calibrated predictions must match latest rows")

        latest_df["predicted_return"] = predicted_return
        latest_df["predicted_close"] = latest_df["close"].to_numpy(dtype=float) * (
            1 + predicted_return
        )
    elif target_type == "future_close_price":
        latest_df["predicted_close"] = raw_predictions
        latest_df["predicted_return"] = (
            latest_df["predicted_close"] / latest_df["close"] - 1
        )
    else:
        raise ValueError("target_type must be future_close_price or future_return")

    latest_df["predicted_future_close_from_signal_close"] = latest_df[
        "predicted_close"
    ]
    latest_df["predicted_future_close"] = latest_df["predicted_close"]

    return latest_df
