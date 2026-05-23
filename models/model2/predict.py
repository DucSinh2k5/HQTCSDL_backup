import joblib
import pandas as pd
import numpy as np

from config import (
  DATE_COL,
  SYMBOL_COL,
  FEATURE_COLUMNS,
  LIGHTGBM_MODEL_PATH
)

from prepare_data import DataPreparator


class StockPredictor:
  def __init__(self):
    self.preparator = DataPreparator()
    self.model = joblib.load(LIGHTGBM_MODEL_PATH)

  def predict_latest_by_symbol(self, symbol: str):
    df = self.preparator.load_and_prepare()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    symbol = symbol.upper().strip()

    symbol_df = df[
      df[SYMBOL_COL].str.upper().str.strip() == symbol
    ].copy()

    if symbol_df.empty:
      print("=" * 60)
      print(f"NO DATA FOR SYMBOL: {symbol}")
      print("=" * 60)
      return None

    symbol_df = symbol_df.sort_values(DATE_COL)

    latest_row = symbol_df.iloc[-1:]

    X = latest_row[FEATURE_COLUMNS]

    pred = self.model.predict(X)[0]

    latest_date = latest_row[DATE_COL].iloc[0]
    latest_close = latest_row["close"].iloc[0]

    print("=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"SYMBOL              : {symbol}")
    print(f"LATEST DATE         : {latest_date}")
    print(f"LATEST CLOSE        : {latest_close}")
    print(f"PREDICT RETURN 5D   : {pred:.4%}")

    if pred > 0:
      print("DIRECTION           : INCREASE")
    else:
      print("DIRECTION           : DECREASE")

    return pred


if __name__ == "__main__":
  predictor = StockPredictor()

  predictor.predict_latest_by_symbol("ACB")