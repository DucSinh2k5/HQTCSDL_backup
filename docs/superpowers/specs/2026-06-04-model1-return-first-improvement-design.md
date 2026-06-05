# Model 1 Return-First Improvement Design

## Context

Model 1 currently trains an XGBoost regressor on `target_close`. It then derives
`predicted_return` from `predicted_close / close - 1`. This makes price-level
metrics and R2 look strong, but the model is not optimized for the metrics that
matter for trading decisions:

- Lower `Return_MAE`
- Higher `Directional_Accuracy`
- More consistent wins over the naive baseline

The repository also has two report locations in practice: root-level `reports/`
and `models/model1/reports/`. The current config writes root-level reports, while
the IDE tab is looking at the model-local report. The implementation should avoid
introducing more ambiguity here.

## Goals

1. Make Model 1 optimize 5-session return directly.
2. Preserve existing report and backtest compatibility by still emitting
   `predicted_close`, `predicted_future_close`, and `predicted_return`.
3. Use validation data to reduce noisy return predictions before test/backtest.
4. Treat baseline comparison as a first-class success signal.
5. Verify results on both the holdout test split and walk-forward folds.

## Non-Goals

- Do not optimize for R2.
- Do not add a separate direction classifier in this iteration.
- Do not change the ClickHouse loading contract.
- Do not broaden the scope to Model 2, Model 3, or portfolio strategy redesign.

## Recommended Approach

Use a return-first regression pipeline:

1. Train XGBoost on `target_return` instead of `target_close`.
2. Predict raw returns.
3. Fit a lightweight validation calibrator that shrinks biased or noisy return
   predictions toward zero.
4. Convert calibrated returns back to close prices:

   `predicted_close = close * (1 + predicted_return)`

5. Evaluate return metrics, direction metrics, and baseline deltas.

This keeps the existing downstream shape while making the model care about the
actual trading target.

## Components

### Training

`train_xgboost_model` should remain generic. Callers will pass either close
targets or return targets. Model 1 main and walk-forward will pass
`target_return`.

The saved artifact should record:

- `target_type = "future_return"`
- `return_calibrator`
- `features`
- `horizon`

### Calibration

Add a small calibration utility. It should:

- Fit on validation raw predicted returns and actual validation returns.
- Prefer a simple linear shrinkage model with intercept.
- Fall back to identity calibration when validation data is unusable.
- Optionally zero out very small calibrated predictions using a configurable
  minimum absolute signal threshold.

The calibration should be deterministic and cheap. It should not add another
heavy model dependency.

### Evaluation

`evaluate_model` should support return-target models. It should accept an
optional prediction target mode or infer it from a saved model wrapper where
practical.

For return-target evaluation:

- `raw_predicted_return = model.predict(X_test)`
- `predicted_return = calibrator(raw_predicted_return)`
- `predicted_close = close * (1 + predicted_return)`

Metrics should retain existing fields and add baseline deltas:

- `Return_MAE_Improvement`
- `Return_RMSE_Improvement`
- `Directional_Accuracy`
- `Directional_Accuracy_Over_Baseline`, where baseline direction is no-change
  and therefore only correct when the future return is non-positive
- `Beats_Baseline_Return_MAE`

### Prediction

`predict_latest_price` should load and apply the saved calibrator when the model
artifact declares `target_type = "future_return"`. Existing price-target
artifacts should still work.

### Walk-Forward

Walk-forward training should use the same target mode as the main pipeline:

- train on fold train `target_return`
- fit calibration on fold validation predictions
- evaluate fold test with calibrated returns

Fold metrics should make it easy to count how often Model 1 beats baseline on
`Return_MAE`.

## Success Criteria

The implementation is considered successful when:

1. Unit tests pass.
2. Holdout metrics report `Beats_Baseline_Return_MAE = true`.
3. Holdout `Directional_Accuracy` improves versus the current model-local value
   of about 55.14 percent, or the change is explicitly reported if data drift
   prevents a direct comparison.
4. Walk-forward fold metrics include enough baseline delta fields to count fold
   wins.
5. The final report explains whether the model now beats baseline consistently
   or only partially.

## Testing Plan

Add tests before implementation for:

1. Return-target evaluation converts predicted returns into predicted closes.
2. Calibration shrinks biased validation predictions and preserves output shape.
3. Saved model metadata records `future_return` and the calibrator.
4. Walk-forward trains on `target_return` and evaluates calibrated returns.
5. Latest prediction supports both old price-target artifacts and new
   return-target artifacts.

## Risks

- Directional accuracy can improve while return magnitude worsens. The composite
  gate prevents treating that as success.
- Shrinkage can over-flatten signals and reduce backtest trade count. The
  calibration threshold should be configurable and conservative.
- Report path ambiguity can hide results. The implementation should either keep
  the current config behavior clear or write the same outputs to the path the
  pipeline actually uses.
