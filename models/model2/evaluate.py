import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
  mean_absolute_error,
  mean_squared_error,
  r2_score
)

from config import (
  DATE_COL,
  FEATURE_COLUMNS,
  TARGET_COL,
  TEST_START_DATE,
  LIGHTGBM_MODEL_PATH
)

from prepare_data import DataPreparator


class ModelEvaluator:
  def __init__(self):
    self.preparator = DataPreparator()
    self.model = joblib.load(LIGHTGBM_MODEL_PATH)

  def evaluate(self):
    df = self.preparator.load_and_prepare()

    test_df = df[
      df[DATE_COL] >= TEST_START_DATE
    ].copy()

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COL]

    y_pred = self.model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
      mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    direction_acc = (
      (y_test > 0) == (y_pred > 0)
    ).mean()

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"MAE                : {mae:.6f}")
    print(f"RMSE               : {rmse:.6f}")
    print(f"R2 SCORE           : {r2:.6f}")
    print(f"DIRECTION ACCURACY : {direction_acc:.6f}")

    importance_df = pd.DataFrame({
      "feature": FEATURE_COLUMNS,
      "importance": self.model.feature_importances_
    }).sort_values(
      by="importance",
      ascending=False
    )

    print("=" * 60)
    print("TOP 20 FEATURE IMPORTANCE")
    print(importance_df.head(20))

    return {
      "mae": mae,
      "rmse": rmse,
      "r2": r2,
      "direction_accuracy": direction_acc
    }


if __name__ == "__main__":
  evaluator = ModelEvaluator()
  evaluator.evaluate()