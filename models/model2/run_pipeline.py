from cross_validation import WalkForwardValidator
from train_lightgbm import LightGBMTrainer
from evaluate import ModelEvaluator
from predict import StockPredictor


def main():
  print("=" * 70)
  print("FUTURE RETURN 5D REGRESSION PIPELINE")
  print("=" * 70)

  # =========================
  # WALK-FORWARD VALIDATION
  # =========================

  validator = WalkForwardValidator()
  validator.run()

  # =========================
  # TRAIN FINAL MODEL
  # =========================

  trainer = LightGBMTrainer()
  trainer.train()

  # =========================
  # EVALUATE FINAL MODEL
  # =========================

  evaluator = ModelEvaluator()
  evaluator.evaluate()

  # =========================
  # PREDICT SAMPLE SYMBOL
  # =========================

  predictor = StockPredictor()
  predictor.predict_latest_by_symbol("ACB")

  print("=" * 70)
  print("PIPELINE FINISHED")
  print("=" * 70)


if __name__ == "__main__":
  main()