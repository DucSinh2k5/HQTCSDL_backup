import pandas as pd
import numpy as np

from config import (
  DATE_COL,
  SYMBOL_COL,
  TARGET_COL,
  TEST_START_DATE,
  FEATURE_COLUMNS
)

from load_data import StockDataLoader


class DataPreparator:

  def __init__(self):
    self.loader = StockDataLoader()

  def load_and_prepare(self):
    df = self.loader.load_data() #Load Data
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])  # DATE TYPE

      # SORT
    df = df.sort_values(by = [SYMBOL_COL, DATE_COL] ).reset_index(drop=True)

    # CREATE TARGET: future_return_5d
    future_close = (
      df.groupby(SYMBOL_COL)["close"]
      .shift(-5)
    )

    df[TARGET_COL] = (
      future_close / df["close"]
    ) - 1
  
    # REMOVE MISSING TARGET
    df = df.dropna(
      subset=[TARGET_COL]
    )

    # REMOVE MISSING FEATURES
    df = df.dropna(
      subset=FEATURE_COLUMNS
    )

    # REMOVE INF
    df = df.replace(
      [np.inf, -np.inf],
      np.nan
    )

    df = df.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COL]
    )
  
    # SPLIT TRAIN / TEST
    train_df = df[df[DATE_COL] < TEST_START_DATE].copy()

    test_df = df[df[DATE_COL] >= TEST_START_DATE ].copy()

    return train_df, test_df


if __name__ == "__main__":
  preparator = DataPreparator()

  train_df, test_df = preparator.load_and_prepare()

  print("=" * 50)
  print("TRAIN SHAPE")
  print(train_df.shape)

  print("=" * 50)
  print("TEST SHAPE")
  print(test_df.shape)

  print("=" * 50)
  print("TARGET SAMPLE")
  print(
      train_df[
          [
              SYMBOL_COL,
              DATE_COL,
              "close",
              TARGET_COL
          ]
      ].head(10)
  )