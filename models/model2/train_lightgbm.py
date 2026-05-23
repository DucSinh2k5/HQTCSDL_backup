import os
import joblib

from lightgbm import LGBMRegressor

from config import (
  DATE_COL,
  FEATURE_COLUMNS,
  TARGET_COL,
  TEST_START_DATE,
  MODEL_DIR,
  LIGHTGBM_MODEL_PATH,
  LIGHTGBM_PARAMS
)

from prepare_data import DataPreparator


class LightGBMTrainer:
  def __init__(self):
    self.preparator = DataPreparator()

    self.model = LGBMRegressor(
      **LIGHTGBM_PARAMS
    )

  def train(self):
    df = self.preparator.load_and_prepare()

    train_df = df[
        df[DATE_COL] < TEST_START_DATE
    ].copy()

    test_df = df[
        df[DATE_COL] >= TEST_START_DATE
    ].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COL]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COL]

    print("=" * 50)
    print("TRAIN SHAPE:", X_train.shape)
    print("TEST SHAPE :", X_test.shape)

    print("=" * 50)
    print("TRAINING LIGHTGBM...")

    self.model.fit(
        X_train,
        y_train
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    joblib.dump(
        self.model,
        LIGHTGBM_MODEL_PATH
    )

    print("=" * 50)
    print("MODEL SAVED:")
    print(LIGHTGBM_MODEL_PATH)

    return self.model, train_df, test_df


if __name__ == "__main__":
  trainer = LightGBMTrainer()
  trainer.train()