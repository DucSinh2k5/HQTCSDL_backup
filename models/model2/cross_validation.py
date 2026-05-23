import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor

from sklearn.metrics import (
  mean_absolute_error,
  mean_squared_error,
  r2_score
)

from config import (
  DATE_COL,
  FEATURE_COLUMNS,
  TARGET_COL,
  LIGHTGBM_PARAMS
)

from prepare_data import DataPreparator


class WalkForwardValidator:
  def __init__(self):
    self.preparator = DataPreparator()

  def evaluate_fold(
    self,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
  ) -> dict:

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COL]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COL]

    model = LGBMRegressor(
        **LIGHTGBM_PARAMS
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
      mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    direction_acc = (
      (y_test > 0) == (y_pred > 0)
    ).mean()

    return {
      "mae": mae,
      "rmse": rmse,
      "r2": r2,
      "direction_accuracy": direction_acc,
      "train_size": len(train_df),
      "test_size": len(test_df)
    }

  def run(self):
      df = self.preparator.load_and_prepare()

      df[DATE_COL] = pd.to_datetime(df[DATE_COL])

      folds = [
          ("2020-01-01", "2021-01-01"),
          ("2021-01-01", "2022-01-01"),
          ("2022-01-01", "2023-01-01"),
          ("2023-01-01", "2024-01-01"),
          ("2024-01-01", "2025-01-01"),
      ]

      results = []

      for test_start, test_end in folds:

          train_df = df[
              df[DATE_COL] < test_start
          ].copy()

          test_df = df[
              (df[DATE_COL] >= test_start)
              & (df[DATE_COL] < test_end)
          ].copy()

          if train_df.empty or test_df.empty:
              continue

          print("=" * 60)
          print(f"FOLD: train < {test_start}, test {test_start} -> {test_end}")

          metrics = self.evaluate_fold(
              train_df,
              test_df
          )

          metrics["test_start"] = test_start
          metrics["test_end"] = test_end

          results.append(metrics)

          print(metrics)

      results_df = pd.DataFrame(results)

      print("=" * 60)
      print("WALK-FORWARD VALIDATION RESULT")
      print(results_df)

      print("=" * 60)
      print("MEAN RESULT")
      print(
          results_df[
              ["mae", "rmse", "r2", "direction_accuracy"]
          ].mean()
      )

      return results_df


if __name__ == "__main__":
  validator = WalkForwardValidator()
  validator.run()