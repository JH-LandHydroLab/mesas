"""
Stage 0: Mass balance and solute balance tests.

The water balance array is computed by the Fortran solver as the residual of the
age-ranked conservation equation at each output timestep.  For the balance to be
meaningful across all age cohorts, consecutive timesteps must be recorded
(record_state=True).  With record_state=True the full water-balance matrix
should be near machine precision.

These tests verify that:
1. The water balance matrix is near zero (all ages, all timesteps)
2. The solute balance at age > 0 is near zero (age 0 has structural residual)
3. Output concentrations are finite and reasonable
"""

import numpy as np
import pandas as pd

from mesas.sas.model import Model


def _make_basic_data(timeseries_length=100, dt=0.1, Q_0=1.0, S_0=5.0):
    """Create a basic steady-state DataFrame."""
    rng = np.random.default_rng(123)
    data_df = pd.DataFrame(index=range(timeseries_length))
    data_df["J"] = Q_0
    data_df["Q"] = Q_0
    data_df["S_0"] = S_0
    data_df["C"] = rng.uniform(0, 10, timeseries_length)
    return data_df


class TestWaterBalance:
    """Verify water balance closure for various configurations."""

    def _check_water_balance(self, model, tolerance=1e-10):
        """Check that the full water balance matrix is near zero."""
        wb = model.get_water_balance()
        max_wb = np.abs(wb).max()
        assert max_wb < tolerance, f"Water balance max residual: {max_wb} (tolerance: {tolerance})"

    def test_uniform_sas(self):
        data_df = _make_basic_data()
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        self._check_water_balance(model)

    def test_gamma_sas(self):
        data_df = _make_basic_data()
        model = Model(
            data_df,
            sas_specs={
                "Q": {
                    "Q_SAS": {
                        "func": "gamma",
                        "args": {"a": 1.0, "scale": "S_0", "loc": 0.0},
                    }
                }
            },
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        self._check_water_balance(model)

    def test_multiple_fluxes(self):
        data_df = _make_basic_data()
        data_df["Q1"] = 0.3
        data_df["Q2"] = 0.7
        model = Model(
            data_df,
            sas_specs={
                "Q1": {"Q1_SAS": {"ST": [0, 5.0]}},
                "Q2": {"Q2_SAS": {"ST": [0, 5.0]}},
            },
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        self._check_water_balance(model)

    def test_with_substeps(self):
        data_df = _make_basic_data()
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
            dt=0.1,
            n_substeps=5,
            verbose=False,
            record_state=True,
        )
        model.run()
        # Substeps introduce slightly larger numerical residuals
        self._check_water_balance(model, tolerance=1e-8)


class TestSoluteBalance:
    """Verify solute balance closure for various configurations."""

    TOLERANCE = 1e-10

    def _check_solute_balance(self, model, sol):
        sb = model.get_solute_balance(sol)
        max_sb = np.abs(sb).max()
        assert max_sb < self.TOLERANCE, f"Solute balance max residual: {max_sb} (tolerance: {self.TOLERANCE})"

    def test_solute_balance_uniform(self):
        data_df = _make_basic_data()
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        self._check_solute_balance(model, "C")

    def test_solute_balance_with_reaction(self):
        data_df = _make_basic_data()
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
            solute_parameters={"C": {"C_old": 1.0, "k1": 0.01, "C_eq": 5.0}},
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        self._check_solute_balance(model, "C")

    def test_solute_balance_multiple_fluxes_with_alpha(self):
        data_df = _make_basic_data()
        data_df["Q1"] = 0.4
        data_df["Q2"] = 0.6
        model = Model(
            data_df,
            sas_specs={
                "Q1": {"Q1_SAS": {"ST": [0, 5.0]}},
                "Q2": {"Q2_SAS": {"ST": [0, 5.0]}},
            },
            solute_parameters={"C": {"C_old": 1.0, "alpha": {"Q1": 0.5, "Q2": 1.5}}},
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        self._check_solute_balance(model, "C")


class TestOutputConsistency:
    """Verify that outputs are finite and consistent across configurations."""

    def test_concentrations_finite(self):
        data_df = _make_basic_data()
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            verbose=False,
        )
        model.run()
        C_pred = model.data_df["C --> Q"].values
        assert np.all(np.isfinite(C_pred))

    def test_pQ_sums_to_less_than_one(self):
        """Cumulative probability of flow should not exceed 1."""
        data_df = _make_basic_data()
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
            dt=0.1,
            verbose=False,
        )
        model.run()
        pQ = model.get_pQ("Q")
        # pQ is age-ranked; sum over ages * dt should be <= 1
        # (can be < 1 due to old water fraction)
        pQ_sum = np.sum(pQ, axis=0) * 0.1
        assert np.all(pQ_sum <= 1.0 + 1e-10)

    def test_sT_non_negative(self):
        """Age-ranked storage should be non-negative."""
        data_df = _make_basic_data()
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
            dt=0.1,
            verbose=False,
        )
        model.run()
        sT = model.get_sT()
        assert np.all(sT >= -1e-15)  # allow tiny numerical noise
