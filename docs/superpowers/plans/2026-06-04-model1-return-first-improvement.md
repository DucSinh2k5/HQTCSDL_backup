# Model 1 Return-First Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Model 1 optimize 5-session return directly, reduce `Return_MAE`, improve `Directional_Accuracy`, and report clear baseline wins.

**Architecture:** Keep the current XGBoost regression pipeline, but train Model 1 on `target_return` instead of `target_close`. Add a small deterministic return calibrator fitted on validation predictions, then convert calibrated returns back to `predicted_close` so existing reports and backtests keep their columns.

**Tech Stack:** Python, pandas, numpy, scikit-learn metrics, xgboost, joblib, unittest.

---

## File Structure

- Create `models/model1/src/return_calibration.py`
  - Owns return calibration only.
  - Exposes `ReturnCalibrator` and `fit_return_calibrator`.
- Create `models/model1/tests/test_return_calibration.py`
  - Covers calibration shrinkage, identity fallback, thresholding, and output shape.
- Modify `models/model1/src/evaluate.py`
  - Supports both price-target and return-target model outputs.
  - Adds baseline delta metrics.
- Modify `models/model1/tests/test_evaluate.py`
  - Keeps existing price-target behavior.
  - Adds return-target evaluation tests.
- Modify `models/model1/src/train_model.py`
  - Saves `target_type` and optional `return_calibrator` metadata.
- Modify `models/model1/tests/test_train_model.py`
  - Verifies metadata is saved for return-target artifacts.
- Modify `models/model1/src/predict.py`
  - Loads metadata without breaking old callers.
  - Predicts latest prices from either price-target or return-target artifacts.
- Modify `models/model1/tests/test_model1_predict.py`
  - Replaces the current misplaced model2-predict test file with Model 1 prediction tests.
- Modify `models/model1/main.py`
  - Trains on `target_return`.
  - Fits validation calibration.
  - Evaluates and saves return-target metadata.
- Modify `models/model1/src/walk_forward.py`
  - Trains folds on `target_return`.
  - Fits fold calibrators on validation predictions.
  - Evaluates fold tests with calibrated returns.
- Modify `models/model1/tests/test_walk_forward.py`
  - Verifies walk-forward trains on `target_return` and emits baseline delta metrics.
- Modify `models/model1/src/config.py`
  - Adds `TARGET_TYPE = "future_return"` and `RETURN_CALIBRATION_MIN_ABS_SIGNAL`.
  - Changes XGBoost early-stopping metric from `rmse` to `mae`.

## Task 1: Add Return Calibration Utility

**Files:**
- Create: `models/model1/src/return_calibration.py`
- Create: `models/model1/tests/test_return_calibration.py`

- [ ] **Step 1: Write the failing calibration tests**

Create `models/model1/tests/test_return_calibration.py` with:

```python
import unittest

import numpy as np

from src.return_calibration import ReturnCalibrator, fit_return_calibrator


class ReturnCalibrationTests(unittest.TestCase):
    def test_fit_return_calibrator_shrinks_biased_predictions(self):
        raw_predicted_returns = np.array([0.20, -0.20, 0.10, -0.10])
        actual_returns = np.array([0.10, -0.10, 0.05, -0.05])

        calibrator = fit_return_calibrator(
            raw_predicted_returns,
            actual_returns,
            min_abs_signal=0.0,
        )

        calibrated = calibrator.predict(raw_predicted_returns)

        self.assertAlmostEqual(0.5, calibrator.slope)
        self.assertAlmostEqual(0.0, calibrator.intercept)
        np.testing.assert_allclose(actual_returns, calibrated)

    def test_fit_return_calibrator_uses_identity_when_raw_predictions_are_constant(self):
        raw_predicted_returns = np.array([0.02, 0.02, 0.02])
        actual_returns = np.array([0.01, -0.01, 0.00])

        calibrator = fit_return_calibrator(
            raw_predicted_returns,
            actual_returns,
            min_abs_signal=0.0,
        )

        self.assertAlmostEqual(1.0, calibrator.slope)
        self.assertAlmostEqual(0.0, calibrator.intercept)
        np.testing.assert_allclose(raw_predicted_returns, calibrator.predict(raw_predicted_returns))

    def test_return_calibrator_zeroes_small_signals_after_calibration(self):
        calibrator = ReturnCalibrator(
            slope=0.5,
            intercept=0.0,
            min_abs_signal=0.02,
        )

        calibrated = calibrator.predict(np.array([0.01, 0.10, -0.03]))

        np.testing.assert_allclose(np.array([0.0, 0.05, 0.0]), calibrated)

    def test_return_calibrator_preserves_series_length(self):
        calibrator = ReturnCalibrator(
            slope=1.0,
            intercept=0.0,
            min_abs_signal=0.0,
        )

        calibrated = calibrator.predict([0.01, -0.02, 0.00])

        self.assertEqual(3, len(calibrated))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run from `models/model1`:

```bash
python -m unittest discover -s tests -p test_return_calibration.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.return_calibration'`.

- [ ] **Step 3: Implement the calibration utility**

Create `models/model1/src/return_calibration.py` with:

```python
from dataclasses import dataclass

import numpy as np


@dataclass
class ReturnCalibrator:
    slope: float = 1.0
    intercept: float = 0.0
    min_abs_signal: float = 0.0

    def predict(self, raw_returns):
        raw_array = np.asarray(raw_returns, dtype=float)
        calibrated = raw_array * self.slope + self.intercept

        if self.min_abs_signal > 0:
            calibrated = np.where(
                np.abs(calibrated) >= self.min_abs_signal,
                calibrated,
                0.0,
            )

        return calibrated


def fit_return_calibrator(raw_predicted_returns, actual_returns, min_abs_signal=0.0):
    if min_abs_signal < 0:
        raise ValueError("min_abs_signal must be non-negative")

    raw_array = np.asarray(raw_predicted_returns, dtype=float)
    actual_array = np.asarray(actual_returns, dtype=float)

    if raw_array.shape != actual_array.shape:
        raise ValueError("raw_predicted_returns and actual_returns must have the same shape")

    finite_mask = np.isfinite(raw_array) & np.isfinite(actual_array)
    raw_array = raw_array[finite_mask]
    actual_array = actual_array[finite_mask]

    if len(raw_array) < 2 or np.isclose(np.var(raw_array), 0.0):
        return ReturnCalibrator(min_abs_signal=min_abs_signal)

    raw_mean = raw_array.mean()
    actual_mean = actual_array.mean()
    slope = float(np.sum((raw_array - raw_mean) * (actual_array - actual_mean)))
    slope /= float(np.sum((raw_array - raw_mean) ** 2))
    slope = float(np.clip(slope, 0.0, 1.0))
    intercept = float(actual_mean - slope * raw_mean)

    return ReturnCalibrator(
        slope=slope,
        intercept=intercept,
        min_abs_signal=min_abs_signal,
    )
```

- [ ] **Step 4: Run calibration tests to verify they pass**

Run from `models/model1`:

```bash
python -m unittest discover -s tests -p test_return_calibration.py
```

Expected: PASS, 4 tests.

- [ ] **Step 5: Commit the calibration utility**

Run from repo root if Git index is available:

```bash
git add models/model1/src/return_calibration.py models/model1/tests/test_return_calibration.py
git commit -m "feat(model1): add return calibration utility"
```

If `git add` fails because `.git/index.lock` already exists, do not remove the lock; record the blocker and continue with implementation.

## Task 2: Extend Evaluation for Return-Target Predictions

**Files:**
- Modify: `models/model1/src/evaluate.py`
- Modify: `models/model1/tests/test_evaluate.py`

- [ ] **Step 1: Write the failing return-target evaluation test**

Append this test method to `EvaluateModelTests` in `models/model1/tests/test_evaluate.py`:

```python
    def test_evaluate_model_supports_return_target_predictions_with_calibration(self):
        test_df = pd.DataFrame(
            {
                "close": [100.0, 100.0, 100.0],
                "future_close": [110.0, 95.0, 100.0],
                "target_close": [110.0, 95.0, 100.0],
                "target_return": [0.10, -0.05, 0.00],
            }
        )
        X_test = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        model = DummyModel([0.20, -0.10, 0.01])

        class HalfCalibrator:
            def predict(self, raw_returns):
                return np.asarray(raw_returns) * 0.5

        metrics, result_df = evaluate_model(
            model=model,
            X_test=X_test,
            test_df=test_df,
            target_type="future_return",
            return_calibrator=HalfCalibrator(),
        )

        self.assertIn("raw_predicted_return", result_df.columns)
        np.testing.assert_allclose([0.20, -0.10, 0.01], result_df["raw_predicted_return"])
        np.testing.assert_allclose([0.10, -0.05, 0.005], result_df["predicted_return"])
        np.testing.assert_allclose([110.0, 95.0, 100.5], result_df["predicted_close"])
        self.assertAlmostEqual(0.005 / 3, metrics["Return_MAE"])
        self.assertGreater(metrics["Return_MAE_Improvement"], 0.0)
        self.assertTrue(metrics["Beats_Baseline_Return_MAE"])
        self.assertIn("Directional_Accuracy_Over_Baseline", metrics)
```

- [ ] **Step 2: Run the evaluation test to verify it fails**

Run from `models/model1`:

```bash
python -m unittest discover -s tests -p test_evaluate.py
```

Expected: FAIL with `TypeError: evaluate_model() got an unexpected keyword argument 'target_type'`.

- [ ] **Step 3: Replace `evaluate_model` with return-target support**

In `models/model1/src/evaluate.py`, replace the existing `evaluate_model` function with:

```python
def evaluate_model(
    model,
    X_test,
    test_df,
    target_type="future_close_price",
    return_calibrator=None,
):
    result_df = test_df.copy()

    raw_predictions = model.predict(X_test)

    if target_type == "future_return":
        result_df["raw_predicted_return"] = raw_predictions
        if return_calibrator is not None:
            predicted_return = return_calibrator.predict(raw_predictions)
        else:
            predicted_return = raw_predictions
        result_df["predicted_return"] = predicted_return
        result_df["predicted_close"] = result_df["close"] * (
            1 + result_df["predicted_return"]
        )
    elif target_type == "future_close_price":
        result_df["predicted_close"] = raw_predictions
        result_df["predicted_return"] = result_df["predicted_close"] / result_df["close"] - 1
    else:
        raise ValueError("target_type must be future_close_price or future_return")

    result_df["predicted_future_close"] = result_df["predicted_close"]

    if "target_close" not in result_df.columns:
        result_df["target_close"] = result_df["future_close"]

    if "target_return" not in result_df.columns:
        result_df["target_return"] = result_df["target_close"] / result_df["close"] - 1

    y_true = result_df["target_close"]
    y_pred = result_df["predicted_close"]
    y_true_return = result_df["target_return"]
    y_pred_return = result_df["predicted_return"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return_mae = mean_absolute_error(y_true_return, y_pred_return)
    return_rmse = np.sqrt(mean_squared_error(y_true_return, y_pred_return))

    result_df["actual_direction"] = np.where(result_df["target_return"] > 0, 1, 0)
    result_df["predicted_direction"] = np.where(result_df["predicted_return"] > 0, 1, 0)

    directional_accuracy = (
        result_df["actual_direction"] == result_df["predicted_direction"]
    ).mean() * 100

    baseline_pred = result_df["close"]
    baseline_mae = mean_absolute_error(y_true, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))
    baseline_mape = np.mean(np.abs((y_true - baseline_pred) / y_true)) * 100
    baseline_return_pred = np.zeros(len(result_df))
    baseline_return_mae = mean_absolute_error(y_true_return, baseline_return_pred)
    baseline_return_rmse = np.sqrt(
        mean_squared_error(y_true_return, baseline_return_pred)
    )
    baseline_direction = np.zeros(len(result_df), dtype=int)
    baseline_directional_accuracy = (
        result_df["actual_direction"].to_numpy() == baseline_direction
    ).mean() * 100

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "Return_MAE": return_mae,
        "Return_RMSE": return_rmse,
        "Directional_Accuracy": directional_accuracy,
        "Baseline_MAE": baseline_mae,
        "Baseline_RMSE": baseline_rmse,
        "Baseline_MAPE": baseline_mape,
        "Baseline_Return_MAE": baseline_return_mae,
        "Baseline_Return_RMSE": baseline_return_rmse,
        "Baseline_Directional_Accuracy": baseline_directional_accuracy,
        "Return_MAE_Improvement": baseline_return_mae - return_mae,
        "Return_RMSE_Improvement": baseline_return_rmse - return_rmse,
        "Directional_Accuracy_Over_Baseline": (
            directional_accuracy - baseline_directional_accuracy
        ),
        "Beats_Baseline_Return_MAE": bool(return_mae < baseline_return_mae),
    }

    return metrics, result_df
```

- [ ] **Step 4: Run evaluation tests to verify they pass**

Run from `models/model1`:

```bash
python -m unittest discover -s tests -p test_evaluate.py
```

Expected: PASS, including the existing price-target test and the new return-target test.

- [ ] **Step 5: Commit evaluation support**

Run from repo root if Git index is available:

```bash
git add models/model1/src/evaluate.py models/model1/tests/test_evaluate.py
git commit -m "feat(model1): evaluate return-target predictions"
```

If `git add` fails because `.git/index.lock` already exists, do not remove the lock; record the blocker and continue with implementation.

## Task 3: Save and Load Return-Target Model Metadata

**Files:**
- Modify: `models/model1/src/train_model.py`
- Modify: `models/model1/src/predict.py`
- Modify: `models/model1/tests/test_train_model.py`
- Create: `models/model1/tests/test_model1_predict.py`

- [ ] **Step 1: Write the failing save metadata test**

Append this test method to `TrainModelTests` in `models/model1/tests/test_train_model.py`:

```python
    def test_save_model_records_return_target_metadata(self):
        from pathlib import Path
        import tempfile

        from src.return_calibration import ReturnCalibrator
        from src.train_model import save_model
        import joblib

        calibrator = ReturnCalibrator(slope=0.5, intercept=0.01, min_abs_signal=0.001)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.pkl"

            save_model(
                model=object(),
                features=["feature"],
                horizon=5,
                model_path=model_path,
                target_type="future_return",
                return_calibrator=calibrator,
            )

            saved = joblib.load(model_path)

        self.assertEqual("future_return", saved["target_type"])
        self.assertIs(saved["return_calibrator"], calibrator)
        self.assertEqual(["feature"], saved["features"])
        self.assertEqual(5, saved["horizon"])
```

- [ ] **Step 2: Run the save metadata test to verify it fails**

Run from `models/model1`:

```bash
python -m unittest discover -s tests -p test_train_model.py
```

Expected: FAIL with `TypeError: save_model() got an unexpected keyword argument 'target_type'`.

- [ ] **Step 3: Update `save_model`**

In `models/model1/src/train_model.py`, replace `save_model` with:

```python
def save_model(
    model,
    features,
    horizon,
    model_path,
    target_type="future_close_price",
    return_calibrator=None,
):
    joblib.dump(
        {
            "model": model,
            "features": features,
            "horizon": horizon,
            "target_type": target_type,
            "return_calibrator": return_calibrator,
        },
        model_path,
    )
```

- [ ] **Step 4: Run train model tests to verify they pass**

Run from `models/model1`:

```bash
python -m unittest discover -s tests -p test_train_model.py
```

Expected: PASS.

- [ ] **Step 5: Write failing Model 1 prediction metadata tests**

Create `models/model1/tests/test_model1_predict.py` with:

```python
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.predict import load_saved_model, predict_latest_price
from src.return_calibration import ReturnCalibrator


class DummyModel:
    def __init__(self, predictions):
        self.predictions = np.array(predictions)

    def predict(self, X):
        return self.predictions[: len(X)]


class Model1PredictTests(unittest.TestCase):
    def test_load_saved_model_can_return_metadata_for_new_artifacts(self):
        artifact = {
            "model": DummyModel([0.01]),
            "features": ["feature"],
            "horizon": 5,
            "target_type": "future_return",
            "return_calibrator": ReturnCalibrator(slope=0.5),
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.pkl"
            joblib.dump(artifact, model_path)

            model, features, metadata = load_saved_model(
                model_path,
                include_metadata=True,
            )

        self.assertIs(artifact["model"], model)
        self.assertEqual(["feature"], features)
        self.assertEqual("future_return", metadata["target_type"])
        self.assertIs(artifact["return_calibrator"], metadata["return_calibrator"])

    def test_predict_latest_price_uses_return_target_artifact_metadata(self):
        df = pd.DataFrame(
            {
                "trading_date": ["2024-01-01", "2024-01-02"],
                "symbol": ["AAA", "AAA"],
                "close": [100.0, 120.0],
                "feature": [1.0, 2.0],
            }
        )
        model = DummyModel([0.20])
        calibrator = ReturnCalibrator(slope=0.5, intercept=0.0, min_abs_signal=0.0)

        latest_df = predict_latest_price(
            df=df,
            model=model,
            features=["feature"],
            target_type="future_return",
            return_calibrator=calibrator,
        )

        self.assertAlmostEqual(0.10, latest_df.iloc[0]["raw_predicted_return"])
        self.assertAlmostEqual(0.10, latest_df.iloc[0]["predicted_return"])
        self.assertAlmostEqual(132.0, latest_df.iloc[0]["predicted_close"])

    def test_predict_latest_price_keeps_price_target_compatibility(self):
        df = pd.DataFrame(
            {
                "trading_date": ["2024-01-01"],
                "symbol": ["AAA"],
                "close": [100.0],
                "feature": [1.0],
            }
        )
        model = DummyModel([110.0])

        latest_df = predict_latest_price(
            df=df,
            model=model,
            features=["feature"],
        )

        self.assertAlmostEqual(110.0, latest_df.iloc[0]["predicted_close"])
        self.assertAlmostEqual(0.10, latest_df.iloc[0]["predicted_return"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 6: Run prediction tests to verify they fail**

Run from `models/model1`:

```bash
python -m unittest discover -s tests -p test_model1_predict.py
```

Expected: FAIL with `TypeError: load_saved_model() got an unexpected keyword argument 'include_metadata'`.

- [ ] **Step 7: Update saved-model loading and latest prediction**

In `models/model1/src/predict.py`, replace `load_saved_model` and `predict_latest_price` with:

```python
def load_saved_model(model_path, include_metadata=False):
    saved = joblib.load(model_path)

    model = saved["model"]
    features = saved["features"]

    if include_metadata:
        metadata = {
            "horizon": saved.get("horizon"),
            "target_type": saved.get("target_type", "future_close_price"),
            "return_calibrator": saved.get("return_calibrator"),
        }
        return model, features, metadata

    return model, features


def predict_latest_price(
    df,
    model,
    features,
    target_type="future_close_price",
    return_calibrator=None,
):
    df = df.copy()

    df = df.replace(["NULL", "null", "None", ""], np.nan)

    df["trading_date"] = pd.to_datetime(df["trading_date"])
    df = df.sort_values(["symbol", "trading_date"])

    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    latest_df = df.groupby("symbol").tail(1).copy()
    latest_df = latest_df.dropna(subset=features)

    X_latest = latest_df[features]
    raw_predictions = model.predict(X_latest)

    if target_type == "future_return":
        latest_df["raw_predicted_return"] = raw_predictions
        if return_calibrator is not None:
            latest_df["predicted_return"] = return_calibrator.predict(raw_predictions)
        else:
            latest_df["predicted_return"] = raw_predictions
        latest_df["predicted_close"] = latest_df["close"] * (
            1 + latest_df["predicted_return"]
        )
    elif target_type == "future_close_price":
        latest_df["predicted_close"] = raw_predictions
        latest_df["predicted_return"] = latest_df["predicted_close"] / latest_df["close"] - 1
    else:
        raise ValueError("target_type must be future_close_price or future_return")

    latest_df["predicted_future_close"] = latest_df["predicted_close"]
    latest_df["predicted_future_close_from_signal_close"] = latest_df[
        "predicted_close"
    ]

    return latest_df
```

- [ ] **Step 8: Run train and prediction tests to verify they pass**

Run from `models/model1`:

```bash
python -m unittest discover -s tests -p "test_train_model.py"
python -m unittest discover -s tests -p "test_model1_predict.py"
```

Expected: both commands PASS.

- [ ] **Step 9: Commit artifact metadata and prediction support**

Run from repo root if Git index is available:

```bash
git add models/model1/src/train_model.py models/model1/src/predict.py models/model1/tests/test_train_model.py models/model1/tests/test_model1_predict.py
git commit -m "feat(model1): save and load return-target artifacts"
```

If `git add` fails because `.git/index.lock` already exists, do not remove the lock; record the blocker and continue with implementation.

## Task 4: Configure Model 1 for Return-First Training

**Files:**
- Modify: `models/model1/src/config.py`
- Modify: `models/model1/tests/test_config.py`

- [ ] **Step 1: Write failing config assertions**

Append these assertions inside `test_backtest_config_is_available` in `models/model1/tests/test_config.py`:

```python
        self.assertEqual("future_return", config.TARGET_TYPE)
        self.assertGreaterEqual(config.RETURN_CALIBRATION_MIN_ABS_SIGNAL, 0)
        self.assertEqual("mae", config.XGB_PARAMS["eval_metric"])
```

- [ ] **Step 2: Run config tests to verify they fail**

Run from `models/model1`:

```bash
python -m unittest discover -s tests -p test_config.py
```

Expected: FAIL with `AttributeError: module 'src.config' has no attribute 'TARGET_TYPE'`.

- [ ] **Step 3: Add return-first config values**

In `models/model1/src/config.py`, add these constants near `HORIZON`:

```python
TARGET_TYPE = "future_return"
RETURN_CALIBRATION_MIN_ABS_SIGNAL = 0.001
```

In `XGB_PARAMS`, change:

```python
    "eval_metric": "rmse",
```

to:

```python
    "eval_metric": "mae",
```

- [ ] **Step 4: Run config tests**

Run from `models/model1`:

```bash
python -m unittest discover -s tests -p test_config.py
```

Expected: PASS if `DATA_PATH` exists in the current environment. If this test fails because `data/clean/features_all.csv` is missing, run the narrower config availability test manually by importing config:

```bash
python -c "from src import config; print(config.TARGET_TYPE, config.RETURN_CALIBRATION_MIN_ABS_SIGNAL, config.XGB_PARAMS['eval_metric'])"
```

Expected: prints `future_return 0.001 mae`.

- [ ] **Step 5: Commit config**

Run from repo root if Git index is available:

```bash
git add models/model1/src/config.py models/model1/tests/test_config.py
git commit -m "feat(model1): configure return-first training"
```

If `git add` fails because `.git/index.lock` already exists, do not remove the lock; record the blocker and continue with implementation.

## Task 5: Switch Main Pipeline to Train on Returns

**Files:**
- Modify: `models/model1/main.py`

- [ ] **Step 1: Write a failing pipeline wiring test**

Create `models/model1/tests/test_main_pipeline.py` with:

```python
import unittest
from unittest.mock import Mock, patch

import pandas as pd

import main as model1_main


class MainPipelineTests(unittest.TestCase):
    def test_main_trains_on_target_return_and_saves_return_artifact(self):
        raw_df = pd.DataFrame({"raw": [1]})
        processed_df = pd.DataFrame(
            {
                "trading_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "symbol": ["AAA", "AAA", "AAA"],
                "feature": [1.0, 2.0, 3.0],
                "close": [100.0, 100.0, 100.0],
                "target_close": [101.0, 102.0, 103.0],
                "target_return": [0.01, 0.02, 0.03],
            }
        )
        train_df = processed_df.iloc[[0]]
        validation_df = processed_df.iloc[[1]]
        test_df = processed_df.iloc[[2]]
        model = Mock()
        model.predict.return_value = [0.02]

        with patch.object(model1_main, "FEATURES", ["feature"]), \
            patch.object(model1_main, "create_folders"), \
            patch.object(model1_main, "load_data", return_value=raw_df), \
            patch.object(model1_main, "preprocess_data", return_value=(processed_df, ["feature"])), \
            patch.object(
                model1_main,
                "split_train_validation_test_by_time",
                return_value=(train_df, validation_df, test_df, pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")),
            ), \
            patch.object(model1_main, "train_xgboost_model", return_value=model) as train_fn, \
            patch.object(model1_main, "fit_return_calibrator") as fit_calibrator, \
            patch.object(model1_main, "evaluate_model", return_value=({"Return_MAE": 0.01}, test_df.copy())) as evaluate_fn, \
            patch.object(model1_main, "compute_top_k_backtest", return_value=(pd.DataFrame(), {})), \
            patch.object(model1_main, "run_backtest_sweep", return_value=pd.DataFrame()), \
            patch.object(model1_main, "save_model") as save_model_fn, \
            patch.object(model1_main, "save_metrics"), \
            patch.object(model1_main, "build_prediction_accuracy_table", return_value=pd.DataFrame()), \
            patch.object(model1_main, "save_backtest_metrics"), \
            patch.object(model1_main, "save_feature_importance"), \
            patch.object(pd.DataFrame, "to_csv"):

            calibrator = Mock()
            fit_calibrator.return_value = calibrator

            model1_main.main()

        self.assertEqual([0.01], train_fn.call_args.kwargs["y_train"].tolist())
        self.assertEqual([0.02], train_fn.call_args.kwargs["y_val"].tolist())
        fit_calibrator.assert_called_once()
        self.assertEqual("future_return", evaluate_fn.call_args.kwargs["target_type"])
        self.assertIs(calibrator, evaluate_fn.call_args.kwargs["return_calibrator"])
        self.assertEqual("future_return", save_model_fn.call_args.kwargs["target_type"])
        self.assertIs(calibrator, save_model_fn.call_args.kwargs["return_calibrator"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the main pipeline wiring test to verify it fails**

Run from `models/model1`:

```bash
python -m unittest discover -s tests -p test_main_pipeline.py
```

Expected: FAIL because `fit_return_calibrator` is not imported by `main.py`, or because `y_train` is still `target_close`.

- [ ] **Step 3: Update main pipeline imports**

In `models/model1/main.py`, add `TARGET_TYPE` and `RETURN_CALIBRATION_MIN_ABS_SIGNAL` to the config import list:

```python
    TARGET_TYPE,
    RETURN_CALIBRATION_MIN_ABS_SIGNAL
```

Add this import near the other `src` imports:

```python
from src.return_calibration import fit_return_calibrator
```

- [ ] **Step 4: Update main pipeline target selection and calibration**

In `models/model1/main.py`, replace the target setup and evaluation section with:

```python
    X_train = train_df[final_features]
    y_train = train_df["target_return"]

    X_val = validation_df[final_features]
    y_val = validation_df["target_return"]

    X_test = test_df[final_features]

    print("Training XGBoost Regressor")
    model = train_xgboost_model(
        X_train=X_train,
        y_train=y_train,
        params=XGB_PARAMS,
        X_val=X_val,
        y_val=y_val,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=False
    )

    print("Fitting return calibration")
    validation_raw_predicted_return = model.predict(X_val)
    return_calibrator = fit_return_calibrator(
        validation_raw_predicted_return,
        y_val,
        min_abs_signal=RETURN_CALIBRATION_MIN_ABS_SIGNAL,
    )

    print("Evaluating model")
    metrics, result_df = evaluate_model(
        model=model,
        X_test=X_test,
        test_df=test_df,
        target_type=TARGET_TYPE,
        return_calibrator=return_calibrator,
    )
```

In the `save_model` call, add:

```python
        target_type=TARGET_TYPE,
        return_calibrator=return_calibrator
```

- [ ] **Step 5: Run the main pipeline wiring test**

Run from `models/model1`:

```bash
python -m unittest discover -s tests -p test_main_pipeline.py
```

Expected: PASS.

- [ ] **Step 6: Commit main pipeline wiring**

Run from repo root if Git index is available:

```bash
git add models/model1/main.py models/model1/tests/test_main_pipeline.py
git commit -m "feat(model1): train main pipeline on returns"
```

If `git add` fails because `.git/index.lock` already exists, do not remove the lock; record the blocker and continue with implementation.

## Task 6: Switch Walk-Forward to Return Targets

**Files:**
- Modify: `models/model1/src/walk_forward.py`
- Modify: `models/model1/walk_forward.py`
- Modify: `models/model1/tests/test_walk_forward.py`

- [ ] **Step 1: Update the walk-forward test to expect return training**

In `models/model1/tests/test_walk_forward.py`, replace this assertion:

```python
        self.assertTrue((train_calls[0][1] == 101.0).all())
```

with:

```python
        self.assertTrue((train_calls[0][1] == 0.01).all())
```

Add these assertions after the existing fold metric assertions:

```python
        self.assertIn("Return_MAE_Improvement", fold_metrics_df.columns)
        self.assertIn("Beats_Baseline_Return_MAE", fold_metrics_df.columns)
```

- [ ] **Step 2: Run walk-forward tests to verify they fail**

Run from `models/model1`:

```bash
python -m unittest discover -s tests -p test_walk_forward.py
```

Expected: FAIL because the fold still trains on `target_close`.

- [ ] **Step 3: Update walk-forward imports and function signature**

In `models/model1/src/walk_forward.py`, add:

```python
from src.return_calibration import fit_return_calibrator
```

Update `run_walk_forward_backtest` signature to include:

```python
    target_type="future_close_price",
    calibration_min_abs_signal=0.0,
```

- [ ] **Step 4: Update walk-forward target selection and calibration**

Inside each fold loop in `run_walk_forward_backtest`, before calling `train_model_fn`, add:

```python
        if target_type == "future_return":
            train_target = train_df["target_return"]
            validation_target = validation_df["target_return"]
        elif target_type == "future_close_price":
            train_target = train_df["target_close"]
            validation_target = validation_df["target_close"]
        else:
            raise ValueError("target_type must be future_close_price or future_return")
```

Then replace the current `y_train` and `y_val` arguments with:

```python
            y_train=train_target,
            y_val=validation_target,
```

After the model is trained and before `evaluate_model`, add:

```python
        return_calibrator = None
        if target_type == "future_return":
            validation_raw_predicted_return = model.predict(validation_df[features])
            return_calibrator = fit_return_calibrator(
                validation_raw_predicted_return,
                validation_df["target_return"],
                min_abs_signal=calibration_min_abs_signal,
            )
```

Replace the `evaluate_model` call with:

```python
        metrics, fold_prediction_df = evaluate_model(
            model=model,
            X_test=test_df[features],
            test_df=test_df,
            target_type=target_type,
            return_calibrator=return_calibrator,
        )
```

- [ ] **Step 5: Update walk-forward CLI script**

In `models/model1/walk_forward.py`, import `TARGET_TYPE` and `RETURN_CALIBRATION_MIN_ABS_SIGNAL` from config, then pass:

```python
            target_type=TARGET_TYPE,
            calibration_min_abs_signal=RETURN_CALIBRATION_MIN_ABS_SIGNAL,
```

to `run_walk_forward_backtest`.

- [ ] **Step 6: Run walk-forward tests**

Run from `models/model1`:

```bash
python -m unittest discover -s tests -p test_walk_forward.py
```

Expected: PASS.

- [ ] **Step 7: Commit walk-forward return training**

Run from repo root if Git index is available:

```bash
git add models/model1/src/walk_forward.py models/model1/walk_forward.py models/model1/tests/test_walk_forward.py
git commit -m "feat(model1): train walk-forward folds on returns"
```

If `git add` fails because `.git/index.lock` already exists, do not remove the lock; record the blocker and continue with implementation.

## Task 7: Full Test Verification

**Files:**
- Read: `models/model1/tests/`

- [ ] **Step 1: Run all Model 1 unit tests**

Run from `models/model1`:

```bash
python -m unittest discover -s tests
```

Expected: PASS for all Model 1 unit tests. If `test_config.py` fails only because `data/clean/features_all.csv` is absent in this workspace, record that as an environment limitation and run all non-config tests:

```bash
python -m unittest discover -s tests -p "test_return_calibration.py"
python -m unittest discover -s tests -p "test_evaluate.py"
python -m unittest discover -s tests -p "test_train_model.py"
python -m unittest discover -s tests -p "test_model1_predict.py"
python -m unittest discover -s tests -p "test_main_pipeline.py"
python -m unittest discover -s tests -p "test_walk_forward.py"
python -m unittest discover -s tests -p "test_backtest.py"
python -m unittest discover -s tests -p "test_preprocessing.py"
```

Expected: all listed commands PASS.

- [ ] **Step 2: Run syntax check for Model 1 source**

Run from `models/model1`:

```bash
python -m compileall src main.py walk_forward.py
```

Expected: completes with no syntax errors.

- [ ] **Step 3: Commit test verification updates if Git is available**

Run from repo root if Git index is available:

```bash
git status --short
```

Expected: shows only intended Model 1 source, tests, and docs changes plus any pre-existing unrelated files. Do not revert unrelated changes.

## Task 8: Optional Training Verification and Report Review

**Files:**
- Read/write by pipeline: `models/model1/models/price_forecasting_xgb.pkl`
- Read/write by pipeline: configured Model 1 reports

- [ ] **Step 1: Check whether ClickHouse environment is configured**

Run from repo root:

```bash
python -c "from models.model1.src import config; print(bool(config.CLICKHOUSE_HOST), bool(config.CLICKHOUSE_USER), bool(config.CLICKHOUSE_PASSWORD))"
```

Expected: prints three booleans. Continue to the next step only if all three are `True`.

- [ ] **Step 2: Train Model 1**

Run from `models/model1`:

```bash
python main.py
```

Expected: training completes and prints metrics containing `Return_MAE_Improvement`, `Directional_Accuracy_Over_Baseline`, and `Beats_Baseline_Return_MAE`.

- [ ] **Step 3: Run walk-forward backtest**

Run from `models/model1`:

```bash
python walk_forward.py
```

Expected: completes and writes fold metrics containing `Return_MAE_Improvement` and `Beats_Baseline_Return_MAE`.

- [ ] **Step 4: Review metrics against success criteria**

Run from repo root:

```bash
python -c "import json; from pathlib import Path; p=Path('reports/metrics.json'); print(json.dumps(json.loads(p.read_text()), indent=2))"
```

Expected: metrics JSON includes `Beats_Baseline_Return_MAE`. Treat the implementation as successful only if this value is `true` on holdout. For walk-forward, count folds with `Beats_Baseline_Return_MAE = true` and report the count.

## Self-Review Notes

- Spec coverage: return-first training is covered by Tasks 4, 5, and 6; calibration is covered by Task 1 and wired in Tasks 5 and 6; compatibility is covered by Tasks 2 and 3; baseline deltas are covered by Task 2; verification is covered by Tasks 7 and 8.
- Placeholder scan: no deferred implementation markers are used. Every code-changing step includes exact code or exact assertions.
- Type consistency: the plan consistently uses `target_type`, `return_calibrator`, `ReturnCalibrator.predict`, `Return_MAE_Improvement`, `Directional_Accuracy_Over_Baseline`, and `Beats_Baseline_Return_MAE`.
