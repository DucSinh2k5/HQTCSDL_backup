import numpy as np
import pandas as pd

from config import (
    DATE_COL,
    SYMBOL_COL,
    TARGET_COL,
    FEATURE_COLUMNS,
    TARGET_MIN,
    TARGET_MAX
)

from load_data import StockDataLoader
from feature_engineering import FeatureEngineer


class DataPreparator:
  def __init__(self):
    self.loader = StockDataLoader()
    self.engineer = FeatureEngineer()

  def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
    df[TARGET_COL] = (
      df.groupby(SYMBOL_COL)["close"]
      .shift(-5) / df["close"] - 1
    )
    return df

  def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
    df = df[
      df[TARGET_COL].between(TARGET_MIN, TARGET_MAX)
    ].copy()
    return df

  def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COL]
    )

    return df

  def load_and_prepare(self) -> pd.DataFrame:
    df = self.loader.load_data()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    df = df.sort_values(
        by=[SYMBOL_COL, DATE_COL]
    ).reset_index(drop=True)

    df = self.create_target(df)

    df = self.engineer.run(df)

    df = self.remove_outliers(df)

    df = self.clean_data(df)

    return df


if __name__ == "__main__":
  preparator = DataPreparator()

  df = preparator.load_and_prepare()

  print("=" * 50)
  print("PREPARED DATA SHAPE")
  print(df.shape)

  print("=" * 50)
  print("FEATURE COUNT")
  print(len(FEATURE_COLUMNS))

  print("=" * 50)
  print("SAMPLE")
  print(df[[SYMBOL_COL, DATE_COL, "close", TARGET_COL]].head())