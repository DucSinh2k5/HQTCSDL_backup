from dataclasses import dataclass

import numpy as np


MIN_ABS_SIGNAL_ERROR = "min_abs_signal must be finite and non-negative"


def _validate_min_abs_signal(min_abs_signal):
    if not np.isfinite(min_abs_signal) or min_abs_signal < 0:
        raise ValueError(MIN_ABS_SIGNAL_ERROR)


@dataclass
class ReturnCalibrator:
    slope: float = 1.0
    intercept: float = 0.0
    min_abs_signal: float = 0.0

    def __post_init__(self):
        _validate_min_abs_signal(self.min_abs_signal)

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
    _validate_min_abs_signal(min_abs_signal)

    raw_array = np.asarray(raw_predicted_returns, dtype=float)
    actual_array = np.asarray(actual_returns, dtype=float)

    if raw_array.shape != actual_array.shape:
        raise ValueError("raw_predicted_returns and actual_returns must have the same shape")

    finite_mask = np.isfinite(raw_array) & np.isfinite(actual_array)
    raw_array = raw_array[finite_mask]
    actual_array = actual_array[finite_mask]

    if len(raw_array) < 2:
        return ReturnCalibrator(min_abs_signal=min_abs_signal)

    raw_mean = raw_array.mean()
    actual_mean = actual_array.mean()
    denominator = float(np.sum((raw_array - raw_mean) ** 2))

    if denominator == 0.0:
        return ReturnCalibrator(min_abs_signal=min_abs_signal)

    slope = float(np.sum((raw_array - raw_mean) * (actual_array - actual_mean)))
    slope /= denominator
    slope = float(np.clip(slope, 0.0, 1.0))
    intercept = float(actual_mean - slope * raw_mean)

    return ReturnCalibrator(
        slope=slope,
        intercept=intercept,
        min_abs_signal=min_abs_signal,
    )
