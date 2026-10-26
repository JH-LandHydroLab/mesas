"""
Stage 6: Tests for the recursive_split model estimation module.

Tests the core functionality of recursive_split after fixing the
sas_blends -> sas_specs reference bug. The module requires:
- sklearn (KFold)
- scipy.optimize (least_squares)
- scipy.stats (ttest_rel)

Tests focus on:
1. Import succeeds (no broken references at import time)
2. fit_model converges on a simple problem
3. cross_validation_rmse runs without error
"""

import numpy as np
import pandas as pd
import pytest

from mesas.sas.model import Model


def _has_sklearn():
    try:
        import sklearn  # noqa: F401

        return True
    except ImportError:
        return False


requires_sklearn = pytest.mark.skipif(not _has_sklearn(), reason="sklearn not installed")


def _make_estimation_data(n=200, dt=0.1, seed=42):
    """Create synthetic data for parameter estimation.

    Generates steady-state data with a 'true' SAS function, runs the model
    to produce 'observed' concentrations, then returns data suitable for
    fitting.
    """
    rng = np.random.default_rng(seed)
    Q_0 = 1.0
    S_0 = 5.0

    data_df = pd.DataFrame(index=range(n))
    data_df["J"] = Q_0
    data_df["Q"] = Q_0
    data_df["S_0"] = S_0
    data_df["C"] = rng.uniform(0, 10, n)

    # Generate "true" observations using a known SAS function
    true_model = Model(
        data_df.copy(),
        sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
        solute_parameters={"C": {"C_old": 1.0}},
        dt=dt,
        verbose=False,
    )
    true_model.run()

    # Add observations column (what recursive_split will try to match)
    data_df["C_obs"] = true_model.data_df["C --> Q"].values
    # Add a bit of noise
    data_df["C_obs"] += rng.normal(0, 0.01, n)

    return data_df


@requires_sklearn
class TestRecursiveSplitImport:
    """Verify the module imports cleanly after sas_blends fix."""

    def test_import(self):
        import mesas.me.recursive_split as rs

        assert hasattr(rs, "run")
        assert hasattr(rs, "fit_model")
        assert hasattr(rs, "cross_validation_rmse")


@requires_sklearn
class TestFitModel:
    """Test the fit_model function on a simple problem."""

    def test_fit_model_basic(self):
        """fit_model should converge and reduce RMSE."""
        from mesas.me.recursive_split import fit_model

        data_df = _make_estimation_data()

        # Create a model with initial guess (slightly off from true)
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 7.0]}}},
            solute_parameters={
                "C": {
                    "C_old": 0.8,
                    "observations": {"Q": "C_obs"},
                }
            },
            dt=0.1,
            verbose=False,
            components_to_learn={"Q": ["Q_SAS"]},
        )
        model.run()

        # Get initial RMSE
        residuals_before = model.get_residuals()
        rmse_before = np.sqrt(np.mean(residuals_before**2))

        # Fit the model
        fitted_model, rmse_after = fit_model(model, verbose=False, jacobian_mode="numerical")

        # RMSE should decrease
        assert rmse_after < rmse_before, f"RMSE did not decrease: {rmse_before:.6f} -> {rmse_after:.6f}"

    def test_fit_model_analytical_jacobian_raises(self):
        """Analytical jacobian mode is unsupported (solver returns zero
        sensitivities) and must raise rather than silently do nothing."""
        from mesas.me.recursive_split import fit_model

        data_df = _make_estimation_data()

        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 7.0]}}},
            solute_parameters={
                "C": {
                    "C_old": 0.8,
                    "observations": {"Q": "C_obs"},
                }
            },
            dt=0.1,
            verbose=False,
            components_to_learn={"Q": ["Q_SAS"]},
        )

        with pytest.raises(NotImplementedError, match="analytical"):
            fit_model(model, verbose=False, jacobian_mode="analytical")


@requires_sklearn
class TestCrossValidation:
    """Test the cross_validation_rmse function."""

    def test_cross_validation_returns_array(self):
        """cross_validation_rmse should return an array of RMSE values."""
        from mesas.me.recursive_split import cross_validation_rmse

        data_df = _make_estimation_data(n=100)

        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
            solute_parameters={
                "C": {
                    "C_old": 1.0,
                    "observations": {"Q": "C_obs"},
                }
            },
            dt=0.1,
            verbose=False,
            components_to_learn={"Q": ["Q_SAS"]},
        )
        model.run()

        rmse_cv = cross_validation_rmse(model, n_splits=3, verbose=False)
        assert isinstance(rmse_cv, np.ndarray)
        assert len(rmse_cv) == 3
        assert np.all(np.isfinite(rmse_cv))
        assert np.all(rmse_cv > 0)
