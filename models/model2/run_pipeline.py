from train_lightgbm import LightGBMTrainer
from evaluate import ModelEvaluator
from predict import StockPredictor


def main():

    print("=" * 60)
    print("FUTURE RETURN PREDICTION PIPELINE")
    print("=" * 60)

    # =========================
    # TRAIN MODEL
    # =========================

    trainer = LightGBMTrainer()

    trainer.train()

    # =========================
    # EVALUATE MODEL
    # =========================

    evaluator = ModelEvaluator()

    evaluator.evaluate()

    # =========================
    # PREDICT SAMPLE
    # =========================

    predictor = StockPredictor()

    predictor.predict_latest_by_symbol(
        "ACB"
    )

    print("=" * 60)
    print("PIPELINE FINISHED")
    print("=" * 60)


if __name__ == "__main__":

    main()