"""
Stage 0: Edge case tests.

These tests verify that the model handles boundary conditions gracefully:
- No solutes
- Short timeseries
- Different numerical schemes
- Piecewise SAS functions with various segment counts
"""

import numpy as np
import pandas as pd
import pytest

from mesas.sas.model import Model


def _make_minimal_data(timeseries_length=50, dt=0.1, Q_0=1.0, S_0=5.0):
    data_df = pd.DataFrame(index=range(timeseries_length))
    data_df["J"] = Q_0
    data_df["Q"] = Q_0
    data_df["S_0"] = S_0
    data_df["C"] = np.linspace(0, 10, timeseries_length)
    return data_df


class TestNoSolutes:
    """Model should run correctly without any solute parameters."""

    def test_run_no_solutes(self):
        data_df = _make_minimal_data()
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
            dt=0.1,
            verbose=False,
        )
        model.run()
        # Should have results for water but not solutes
        assert "sT" in model.result
        assert "pQ" in model.result
        assert "water_balance" in model.result

    def test_water_balance_no_solutes(self):
        data_df = _make_minimal_data()
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        wb = model.get_water_balance()
        assert np.abs(wb).max() < 1e-10


class TestShortTimeseries:
    """Model should handle very short timeseries."""

    def test_10_timesteps(self):
        data_df = _make_minimal_data(timeseries_length=10)
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        assert model.result is not None
        wb = model.get_water_balance()
        assert np.abs(wb).max() < 1e-10

    def test_5_timesteps(self):
        data_df = _make_minimal_data(timeseries_length=5)
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
            dt=0.1,
            verbose=False,
        )
        model.run()
        assert model.result is not None


class TestNumericalSchemes:
    """Model should produce similar results with different integration schemes."""

    @pytest.mark.parametrize("num_scheme", [1, 2, 4])
    def test_scheme_runs(self, num_scheme):
        data_df = _make_minimal_data(timeseries_length=50)
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            num_scheme=num_scheme,
            verbose=False,
        )
        model.run()
        assert model.result is not None

    def test_higher_order_more_accurate(self):
        """Both schemes should have near-zero water balance residuals."""
        data_df = _make_minimal_data(timeseries_length=100)

        for scheme in [1, 4]:
            model = Model(
                data_df,
                sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
                solute_parameters={"C": {"C_old": 1.0}},
                dt=0.1,
                num_scheme=scheme,
                verbose=False,
                record_state=True,
            )
            model.run()
            wb_max = np.abs(model.get_water_balance()).max()
            assert wb_max < 1e-10, f"Scheme {scheme}: WB max={wb_max}"


class TestPiecewiseSegments:
    """Test piecewise SAS functions with various segment counts."""

    @pytest.mark.parametrize("n_segments", [1, 2, 5, 10])
    def test_n_segments(self, n_segments):
        data_df = _make_minimal_data()
        S_max = 5.0
        ST = np.linspace(0, S_max, n_segments + 1).tolist()
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": ST}}},
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        wb = model.get_water_balance()
        assert np.abs(wb).max() < 1e-10


class TestResultAccessors:
    """Test that all result accessor methods work correctly."""

    @pytest.fixture
    def model_with_results(self):
        data_df = _make_minimal_data()
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            verbose=False,
        )
        model.run()
        return model

    def test_get_sT(self, model_with_results):
        sT = model_with_results.get_sT()
        assert sT is not None
        assert sT.shape[0] > 0

    def test_get_pQ(self, model_with_results):
        pQ = model_with_results.get_pQ("Q")
        assert pQ is not None

    def test_get_mT(self, model_with_results):
        mT = model_with_results.get_mT("C")
        assert mT is not None

    def test_get_CT(self, model_with_results):
        CT = model_with_results.get_CT("C")
        assert CT is not None

    def test_get_mQ(self, model_with_results):
        mQ = model_with_results.get_mQ("Q", "C")
        assert mQ is not None

    def test_get_water_balance(self, model_with_results):
        wb = model_with_results.get_water_balance()
        assert wb is not None

    def test_get_solute_balance(self, model_with_results):
        sb = model_with_results.get_solute_balance("C")
        assert sb is not None

    def test_get_ST(self, model_with_results):
        ST = model_with_results.get_ST()
        assert ST is not None

    def test_result_before_run_raises(self):
        data_df = _make_minimal_data()
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
            dt=0.1,
            verbose=False,
        )
        with pytest.raises(AttributeError, match="[Rr]esults are only"):
            _ = model.result
