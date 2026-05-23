import os
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from config import (
  SYMBOL_COL,
  SYMBOL_ENCODER_PATH,
  LAG_DAYS,
  LAG_SOURCE_COLUMNS,
  FEATURE_COLUMNS
)


class FeatureEngineer:
  def __init__(self):
    self.encoder = LabelEncoder()

  def encode_symbol(self, df: pd.DataFrame) -> pd.DataFrame:
    encoded = np.asarray(
      self.encoder.fit_transform(
          df[SYMBOL_COL].astype(str)
      ),
      dtype=np.int32
    )

    df["symbol_encoded"] = encoded

    os.makedirs(os.path.dirname(SYMBOL_ENCODER_PATH), exist_ok=True)

    joblib.dump(self.encoder, SYMBOL_ENCODER_PATH)

    return df
  
  def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
    for column in LAG_SOURCE_COLUMNS:
      for lag in LAG_DAYS:
        lag_col = f"{column}_lag{lag}"

        df[lag_col] = (
          df.groupby(SYMBOL_COL)[column]
          .shift(lag)
          )

    return df

  def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
    df["volume_volatility_interaction"] = (
      df["volume_ratio_5_20"]
      * df["volatility_5d"]
    )

    return df

  def run(self, df: pd.DataFrame) -> pd.DataFrame:
    df = self.encode_symbol(df)

    df = self.create_lag_features(df)

    df = self.create_interaction_features(df)

    return df


if __name__ == "__main__":
  from load_data import StockDataLoader

  loader = StockDataLoader()

  df = loader.load_data()

  engineer = FeatureEngineer()

  df = engineer.run(df)

  print("=" * 50)
  print("FEATURE ENGINEERING DONE")

  print("=" * 50)
  print(df.head())

  print("=" * 50)
  print("TOTAL FEATURES:")
  print(len(FEATURE_COLUMNS))

  print("=" * 50)
  print("FEATURE COLUMNS:")
  print(FEATURE_COLUMNS)