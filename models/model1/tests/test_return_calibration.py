import unittest
import warnings

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

    def test_fit_return_calibrator_handles_basis_point_level_variance(self):
        raw_predicted_returns = np.array([0.00010, 0.00015, 0.00020, 0.00025])
        actual_returns = raw_predicted_returns * 0.5

        calibrator = fit_return_calibrator(
            raw_predicted_returns,
            actual_returns,
            min_abs_signal=0.0,
        )

        calibrated = calibrator.predict(raw_predicted_returns)

        self.assertAlmostEqual(0.5, calibrator.slope)
        np.testing.assert_allclose(actual_returns, calibrated)

    def test_fit_return_calibrator_handles_tiny_non_constant_variance(self):
        raw_predicted_returns = np.array([1e-12, 2e-12, 3e-12, 4e-12])
        actual_returns = raw_predicted_returns * 0.5

        calibrator = fit_return_calibrator(
            raw_predicted_returns,
            actual_returns,
            min_abs_signal=0.0,
        )

        calibrated = calibrator.predict(raw_predicted_returns)

        self.assertAlmostEqual(0.5, calibrator.slope)
        np.testing.assert_allclose(actual_returns, calibrated)

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

    def test_fit_return_calibrator_rejects_non_finite_min_abs_signal(self):
        raw_predicted_returns = np.array([0.01, 0.02])
        actual_returns = np.array([0.01, 0.02])

        for min_abs_signal in (np.nan, np.inf, -np.inf):
            with self.subTest(min_abs_signal=min_abs_signal):
                with self.assertRaisesRegex(
                    ValueError,
                    "min_abs_signal must be finite and non-negative",
                ):
                    fit_return_calibrator(
                        raw_predicted_returns,
                        actual_returns,
                        min_abs_signal=min_abs_signal,
                    )

    def test_fit_return_calibrator_rejects_negative_min_abs_signal(self):
        with self.assertRaisesRegex(
            ValueError,
            "min_abs_signal must be finite and non-negative",
        ):
            fit_return_calibrator(
                np.array([0.01, 0.02]),
                np.array([0.01, 0.02]),
                min_abs_signal=-0.01,
            )

    def test_return_calibrator_rejects_non_finite_min_abs_signal(self):
        for min_abs_signal in (np.nan, np.inf, -np.inf):
            with self.subTest(min_abs_signal=min_abs_signal):
                with self.assertRaisesRegex(
                    ValueError,
                    "min_abs_signal must be finite and non-negative",
                ):
                    ReturnCalibrator(min_abs_signal=min_abs_signal)

    def test_return_calibrator_rejects_negative_min_abs_signal(self):
        with self.assertRaisesRegex(
            ValueError,
            "min_abs_signal must be finite and non-negative",
        ):
            ReturnCalibrator(min_abs_signal=-0.01)

    def test_fit_return_calibrator_rejects_mismatched_shapes(self):
        with self.assertRaisesRegex(
            ValueError,
            "raw_predicted_returns and actual_returns must have the same shape",
        ):
            fit_return_calibrator(
                np.array([0.01, 0.02, 0.03]),
                np.array([0.01, 0.02]),
            )

    def test_fit_return_calibrator_filters_non_finite_pairs(self):
        raw_predicted_returns = np.array([0.20, np.nan, -0.20, 0.10, np.inf, -0.10])
        actual_returns = np.array([0.10, 0.00, -0.10, 0.05, 0.00, -0.05])

        calibrator = fit_return_calibrator(
            raw_predicted_returns,
            actual_returns,
            min_abs_signal=0.0,
        )

        self.assertAlmostEqual(0.5, calibrator.slope)
        self.assertAlmostEqual(0.0, calibrator.intercept)

    def test_fit_return_calibrator_clips_slope_to_upper_bound(self):
        raw_predicted_returns = np.array([0.10, 0.20, 0.30])
        actual_returns = raw_predicted_returns * 2.0

        calibrator = fit_return_calibrator(
            raw_predicted_returns,
            actual_returns,
            min_abs_signal=0.0,
        )

        self.assertAlmostEqual(1.0, calibrator.slope)

    def test_fit_return_calibrator_clips_slope_to_lower_bound(self):
        raw_predicted_returns = np.array([0.10, 0.20, 0.30])
        actual_returns = raw_predicted_returns * -0.5

        calibrator = fit_return_calibrator(
            raw_predicted_returns,
            actual_returns,
            min_abs_signal=0.0,
        )

        self.assertAlmostEqual(0.0, calibrator.slope)

    def test_identity_fallback_preserves_min_abs_signal(self):
        calibrator = fit_return_calibrator(
            np.array([0.02, 0.02, 0.02]),
            np.array([0.01, -0.01, 0.00]),
            min_abs_signal=0.03,
        )

        self.assertAlmostEqual(1.0, calibrator.slope)
        self.assertAlmostEqual(0.03, calibrator.min_abs_signal)

    def test_all_non_finite_pairs_return_identity_without_runtime_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            calibrator = fit_return_calibrator(
                np.array([np.nan, np.inf]),
                np.array([0.01, 0.02]),
                min_abs_signal=0.0,
            )

        self.assertAlmostEqual(1.0, calibrator.slope)
        self.assertAlmostEqual(0.0, calibrator.intercept)


if __name__ == "__main__":
    unittest.main()
