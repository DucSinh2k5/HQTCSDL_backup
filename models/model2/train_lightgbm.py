import os
import joblib

from lightgbm import LGBMRegressor

from config import (
  FEATURE_COLUMNS,
  TARGET_COL,
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
    # LOAD DATA
    train_df, test_df = (self.preparator.load_and_prepare())

    # TRAIN DATA
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COL]


    # TEST DATA
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COL]


    # TRAIN MODEL
    print("=" * 50)
    print("TRAINING LIGHTGBM MODEL...")

    self.model.fit(
        X_train,
        y_train
    )

    print("TRAIN DONE")

    # SAVE MODEL
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

    return self.model


if __name__ == "__main__":
  trainer = LightGBMTrainer()
  trainer.train()