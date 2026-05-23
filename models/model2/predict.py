import joblib
import pandas as pd

from config import (
  FEATURE_COLUMNS,
  LIGHTGBM_MODEL_PATH
)

from load_data import StockDataLoader


class StockPredictor:
  def __init__(self):
      self.loader = StockDataLoader()
      self.model = joblib.load(
        LIGHTGBM_MODEL_PATH
      )

  def predict_latest_by_symbol(
    self,
    symbol: str
  ):

    # =========================
    # LOAD DATA
    # =========================
    df = self.loader.load_data()

    # =========================
    # FILTER SYMBOL
    # =========================
    df = df[df["symbol"] == symbol].copy()

    if df.empty:
      print(f"NO DATA FOR SYMBOL: {symbol}")
      return

    # =========================
    # SORT
    # =========================
    df["trading_date"] = pd.to_datetime(df["trading_date"])

    df = df.sort_values(by="trading_date")

    # =========================
    # GET LATEST ROW
    # =========================
    latest_row = df.iloc[-1:]

    # =========================
    # FEATURE DATA
    # =========================
    X = latest_row[FEATURE_COLUMNS]

    # =========================
    # PREDICT
    # =========================
    pred = self.model.predict(X)[0]

    # =========================
    # RESULT
    # =========================
    print("=" * 50)
    print(f"SYMBOL: {symbol}")

    print(
      f"PREDICTED FUTURE RETURN 5D: {pred:.4%}"
    )

    if pred > 0:
      print("PREDICT: PRICE MAY INCREASE")

    else:
      print("PREDICT: PRICE MAY DECREASE")
    print("=" * 50)


if __name__ == "__main__":
  predictor = StockPredictor()
  predictor.predict_latest_by_symbol(
    "VCB"
  )