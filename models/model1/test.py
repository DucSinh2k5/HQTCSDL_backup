from pathlib import Path
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


MODEL1_DIR = Path(__file__).resolve().parent
if str(MODEL1_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL1_DIR))

from src.config import HORIZON, MODEL_PATH, TARGET_TYPE  # noqa: E402
from src.data_loader import load_data  # noqa: E402
from src.predict import load_saved_model  # noqa: E402


REPORT_DIR = MODEL1_DIR / "reports"
PREDICTION_LOG_PATH = REPORT_DIR / "streamlit_predictions.csv"
LATEST_PREDICTION_PATH = REPORT_DIR / "latest_streamlit_prediction.json"


st.set_page_config(page_title="Model 1 Demo", layout="wide")
st.title("Model 1 - Stock Prediction Demo")


@st.cache_resource
def get_model():
    return load_saved_model(MODEL_PATH, include_metadata=True)


@st.cache_data(ttl=600)
def get_feature_data():
    df = load_data()
    df["trading_date"] = pd.to_datetime(df["trading_date"])
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    return df.sort_values(["symbol", "trading_date"])


def save_prediction(result):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame([result])
    if PREDICTION_LOG_PATH.exists():
        result_df.to_csv(PREDICTION_LOG_PATH, mode="a", header=False, index=False)
    else:
        result_df.to_csv(PREDICTION_LOG_PATH, index=False)

    with open(LATEST_PREDICTION_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4, default=str)


def predict_one_row(row_df, model, features, metadata):
    missing_features = [col for col in features if col not in row_df.columns]
    if missing_features:
        raise ValueError("Missing feature: " + ", ".join(missing_features))

    for col in features + ["close"]:
        row_df[col] = pd.to_numeric(row_df[col], errors="coerce")

    if row_df[features + ["close"]].isna().any(axis=None):
        raise ValueError("Selected row has missing or non-numeric features.")

    raw_prediction = np.asarray(model.predict(row_df[features]), dtype=float)

    target_type = metadata.get("target_type", TARGET_TYPE)
    return_calibrator = metadata.get("return_calibrator")
    close = float(row_df["close"].iloc[0])

    if target_type == "future_return":
        if return_calibrator is not None:
            predicted_return = float(return_calibrator.predict(raw_prediction)[0])
        else:
            predicted_return = float(raw_prediction[0])
        predicted_close = close * (1 + predicted_return)
    else:
        predicted_close = float(raw_prediction[0])
        predicted_return = predicted_close / close - 1

    return predicted_return, predicted_close


try:
    model, features, metadata = get_model()
    df = get_feature_data()

    symbols = sorted(df["symbol"].dropna().unique().tolist())
    default_symbol_index = symbols.index("FPT") if "FPT" in symbols else 0
    symbol = st.selectbox("Symbol", symbols, index=default_symbol_index)

    symbol_df = df[df["symbol"] == symbol].copy()
    valid_dates = sorted(symbol_df["trading_date"].dt.date.unique().tolist())
    selected_date = st.selectbox(
        "Trading date",
        valid_dates,
        index=len(valid_dates) - 1,
    )

    if st.button("Predict", type="primary"):
        row_df = symbol_df[symbol_df["trading_date"].dt.date == selected_date].copy()

        if row_df.empty:
            st.warning("No feature data found for the selected symbol and date.")
            st.stop()

        if len(row_df) > 1:
            row_df = row_df.head(1).copy()

        predicted_return, predicted_close = predict_one_row(
            row_df=row_df,
            model=model,
            features=features,
            metadata=metadata,
        )

        direction = "UP" if predicted_return >= 0 else "DOWN"

        result = {
            "run_at": datetime.now().isoformat(timespec="seconds"),
            "symbol": symbol,
            "trading_date": str(selected_date),
            "close": float(row_df["close"].iloc[0]),
            "predicted_return": predicted_return,
            "predicted_return_pct": predicted_return * 100,
            "predicted_close": predicted_close,
            "direction": direction,
            "horizon": metadata.get("horizon", HORIZON),
            "target_type": metadata.get("target_type", TARGET_TYPE),
        }

        save_prediction(result)

        st.subheader("Prediction Result")
        st.json(result)

        st.subheader("Features Used")
        st.dataframe(
            row_df[["symbol", "trading_date"] + features],
            use_container_width=True,
        )

        st.success(f"Saved prediction to {PREDICTION_LOG_PATH}")

except Exception as exc:
    st.error(f"Error running Model 1: {exc}")
