import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
  mean_absolute_error,
  mean_squared_error,
  r2_score
)

from config import (
  FEATURE_COLUMNS,
  TARGET_COL,
  LIGHTGBM_MODEL_PATH
)

from prepare_data import DataPreparator

class ModelEvaluator:
  def __init__(self):
    self.preparator = DataPreparator()
    self.model = joblib.load(
      LIGHTGBM_MODEL_PATH
    )

  def evaluate(self):
    # =========================
    # LOAD DATA
    # =========================
    train_df, test_df = (
      self.preparator.load_and_prepare()
    )

    # =========================
    # TEST DATA
    # =========================
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COL]

    # =========================
    # PREDICT
    # =========================
    y_pred = self.model.predict(X_test)

    # =========================
    # METRICS
    # =========================
    mae = mean_absolute_error(
      y_test,
      y_pred
    )

    rmse = np.sqrt(
      mean_squared_error(
        y_test,
        y_pred
      )
    )

    r2 = r2_score(
      y_test,
      y_pred
    )

    # =========================
    # DIRECTION ACCURACY
    # =========================
    direction_acc = (
        (y_test > 0) ==
        (y_pred > 0)
    ).mean()

    # =========================
    # RESULT
    # =========================
    print("=" * 50)
    print("MODEL EVALUATION")

    print(f"MAE                 : {mae:.6f}")
    print(f"RMSE                : {rmse:.6f}")
    print(f"R2 SCORE            : {r2:.6f}")
    print(f"DIRECTION ACCURACY  : {direction_acc:.6f}")

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    importance_df = pd.DataFrame({
      "feature": FEATURE_COLUMNS,
      "importance": self.model.feature_importances_
    })

    importance_df = importance_df.sort_values(
      by="importance",
      ascending=False
    )

    print("=" * 50)
    print("TOP FEATURE IMPORTANCE")

    print(
      importance_df.head(15)
    )


if __name__ == "__main__":
  evaluator = ModelEvaluator()
  evaluator.evaluate()