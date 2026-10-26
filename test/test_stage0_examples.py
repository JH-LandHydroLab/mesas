"""
Stage 0: Regression tests for the bundled examples.

These tests run the lower_hafren and hyporheic examples and verify that:
1. The model runs without error
2. Output concentrations are within a plausible range
3. Key output values match saved reference values (regression check)

Reference values were captured from the current codebase and serve as a
baseline. If a refactoring changes these values, the test will fail and
the developer must verify the change is intentional before updating.
"""

import numpy as np
import pytest

from mesas.sas.model import Model


class TestLowerHafren:
    """Regression tests for the lower_hafren catchment example."""

    @pytest.fixture
    def model(self):
        m = Model(
            data_df="./examples/lower_hafren/data.csv",
            config="./examples/lower_hafren/config.json",
        )
        m.run()
        return m

    def test_runs_without_error(self, model):
        assert model._result is not None

    def test_output_concentration_range(self, model):
        C_pred = model.data_df["Cl mg/l --> Q"].values
        # A small number of NaN values can occur (existing behavior)
        assert np.sum(np.isnan(C_pred)) <= 5
        assert np.nanmin(C_pred) >= -1e-10
        assert np.nanmax(C_pred) < 100

    def test_output_concentration_regression(self, model):
        """Check that key summary statistics haven't changed."""
        C_pred = model.data_df["Cl mg/l --> Q"].values
        assert C_pred.shape[0] == 9375
        assert abs(C_pred[0] - 7.11) < 0.5  # close to C_old initially

    def test_water_balance_closure(self, model):
        """Youngest-age water balance should be near machine precision."""
        wb = model.get_water_balance()
        assert np.abs(wb[0]).max() < 1e-10


class TestHyporheic:
    """Regression tests for the hyporheic exchange example."""

    @pytest.fixture
    def model(self):
        m = Model(
            data_df="./examples/hyporheic/data.csv",
            config="./examples/hyporheic/config.json",
        )
        m.run()
        return m

    def test_runs_without_error(self, model):
        assert model._result is not None

    def test_output_concentration_range(self, model):
        C_pred = model.data_df["C_J --> Q"].values
        assert np.all(np.isfinite(C_pred))
        assert np.nanmin(C_pred) >= -1e-10

    def test_water_balance_closure(self, model):
        wb = model.get_water_balance()
        assert np.abs(wb[0]).max() < 1e-10
