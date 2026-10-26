"""
Stage 6: Time-varying parameter tests.

Verifies that SAS function parameters referencing DataFrame columns
correctly update at each timestep. Tests:
1. Time-varying ST bounds (piecewise)
2. Time-varying distribution parameters (gamma scale, loc)
3. Time-varying reaction rates (k1, C_eq)
4. Multiple solutes with time-varying parameters
5. Consistency between constant and column-referenced parameters
"""

import numpy as np
import pandas as pd

from mesas.sas.model import Model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_unsteady_data(n=100, dt=0.1):
    """Create a DataFrame with time-varying fluxes and storage."""
    rng = np.random.default_rng(77)
    data_df = pd.DataFrame(index=range(n))
    # Sinusoidal inflow/outflow to create genuine unsteadiness
    t = np.arange(n) * dt
    Q_mean = 1.0
    Q_amp = 0.3
    data_df["J"] = Q_mean + Q_amp * np.sin(2 * np.pi * t / (n * dt))
    data_df["Q"] = Q_mean - Q_amp * np.sin(2 * np.pi * t / (n * dt))
    # Storage varies with cumulative imbalance
    S_base = 5.0
    S_cumulative = S_base + np.cumsum(data_df["J"].values - data_df["Q"].values) * dt
    data_df["S_0"] = np.maximum(S_cumulative, 0.5)
    data_df["S_m"] = 0.0
    data_df["C"] = rng.uniform(0, 10, n)
    return data_df


# ---------------------------------------------------------------------------
# Tests: Time-varying piecewise ST bounds
# ---------------------------------------------------------------------------


class TestTimeVaryingST:
    """Piecewise SAS with ST bounds that reference DataFrame columns."""

    def test_varying_upper_bound(self):
        """ST upper bound read from a column."""
        data_df = _make_unsteady_data()
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, "S_0"]}}},
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        wb = model.get_water_balance()
        assert np.abs(wb).max() < 1e-8

    def test_varying_both_bounds(self):
        """Both ST bounds read from columns."""
        data_df = _make_unsteady_data()
        data_df["S_m"] = 0.5  # constant lower bound
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": ["S_m", "S_0"]}}},
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        wb = model.get_water_balance()
        assert np.abs(wb).max() < 1e-8
        sb = model.get_solute_balance("C")
        assert np.abs(sb).max() < 1e-8

    def test_varying_multi_segment_ST(self):
        """Multi-segment piecewise with one varying breakpoint."""
        data_df = _make_unsteady_data()
        data_df["S_mid"] = data_df["S_0"] * 0.4
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, "S_mid", "S_0"]}}},
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        wb = model.get_water_balance()
        assert np.abs(wb).max() < 1e-8


# ---------------------------------------------------------------------------
# Tests: Time-varying distribution parameters
# ---------------------------------------------------------------------------


class TestTimeVaryingDistribution:
    """Continuous SAS functions with time-varying parameters."""

    def test_gamma_varying_scale(self):
        """Gamma scale parameter read from a column."""
        data_df = _make_unsteady_data()
        model = Model(
            data_df,
            sas_specs={
                "Q": {
                    "Q_SAS": {
                        "func": "gamma",
                        "args": {"a": 1.5, "scale": "S_0", "loc": 0.0},
                    }
                }
            },
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        wb = model.get_water_balance()
        assert np.abs(wb).max() < 1e-8

    def test_gamma_varying_loc(self):
        """Gamma loc parameter read from a column."""
        data_df = _make_unsteady_data()
        data_df["S_m"] = np.linspace(0, 1, len(data_df))
        model = Model(
            data_df,
            sas_specs={
                "Q": {
                    "Q_SAS": {
                        "func": "gamma",
                        "args": {"a": 1.0, "scale": "S_0", "loc": "S_m"},
                    }
                }
            },
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        wb = model.get_water_balance()
        assert np.abs(wb).max() < 1e-8
        sb = model.get_solute_balance("C")
        assert np.abs(sb).max() < 1e-8

    def test_kumaraswamy_varying_scale(self):
        """Kumaraswamy scale parameter varies over time."""
        data_df = _make_unsteady_data()
        model = Model(
            data_df,
            sas_specs={
                "Q": {
                    "Q_SAS": {
                        "func": "kumaraswamy",
                        "args": {"a": 1.0, "b": 2.0, "scale": "S_0", "loc": 0.0},
                    }
                }
            },
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        wb = model.get_water_balance()
        assert np.abs(wb).max() < 1e-8

    def test_beta_varying_scale(self):
        """Beta scale parameter varies over time."""
        data_df = _make_unsteady_data()
        model = Model(
            data_df,
            sas_specs={
                "Q": {
                    "Q_SAS": {
                        "func": "beta",
                        "args": {"a": 2.0, "b": 3.0, "scale": "S_0", "loc": 0.0},
                    }
                }
            },
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        wb = model.get_water_balance()
        assert np.abs(wb).max() < 1e-8


# ---------------------------------------------------------------------------
# Tests: Consistency between constant and column-referenced parameters
# ---------------------------------------------------------------------------


class TestConstantVsColumnConsistency:
    """When a column has constant values, results should match scalar spec."""

    def test_piecewise_constant_column_matches_scalar(self):
        """A constant column should give same result as a scalar."""
        rng = np.random.default_rng(55)
        n = 50
        data_df = pd.DataFrame(index=range(n))
        data_df["J"] = 1.0
        data_df["Q"] = 1.0
        data_df["S_upper"] = 5.0  # constant column
        data_df["C"] = rng.uniform(0, 10, n)

        # Model with scalar ST upper bound
        model_scalar = Model(
            data_df.copy(),
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            verbose=False,
        )
        model_scalar.run()

        # Model with column reference for ST upper bound
        model_column = Model(
            data_df.copy(),
            sas_specs={"Q": {"Q_SAS": {"ST": [0, "S_upper"]}}},
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            verbose=False,
        )
        model_column.run()

        C_scalar = model_scalar.data_df["C --> Q"].values
        C_column = model_column.data_df["C --> Q"].values
        np.testing.assert_allclose(C_scalar, C_column, atol=1e-10)

    def test_gamma_constant_column_matches_scalar(self):
        """Gamma with constant-valued column should match scalar parameters."""
        rng = np.random.default_rng(55)
        n = 50
        data_df = pd.DataFrame(index=range(n))
        data_df["J"] = 1.0
        data_df["Q"] = 1.0
        data_df["S_0"] = 5.0  # constant column
        data_df["C"] = rng.uniform(0, 10, n)

        model_scalar = Model(
            data_df.copy(),
            sas_specs={
                "Q": {
                    "Q_SAS": {
                        "func": "gamma",
                        "args": {"a": 1.5, "scale": 5.0, "loc": 0.0},
                    }
                }
            },
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            verbose=False,
        )
        model_scalar.run()

        model_column = Model(
            data_df.copy(),
            sas_specs={
                "Q": {
                    "Q_SAS": {
                        "func": "gamma",
                        "args": {"a": 1.5, "scale": "S_0", "loc": 0.0},
                    }
                }
            },
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            verbose=False,
        )
        model_column.run()

        C_scalar = model_scalar.data_df["C --> Q"].values
        C_column = model_column.data_df["C --> Q"].values
        np.testing.assert_allclose(C_scalar, C_column, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: Time-varying reaction parameters
# ---------------------------------------------------------------------------


class TestTimeVaryingReactions:
    """Reaction parameters that vary over time via column references."""

    def test_reaction_with_varying_k1(self):
        """First-order rate constant read from a column."""
        data_df = _make_unsteady_data()
        # Linearly increasing reaction rate
        data_df["k1_col"] = np.linspace(0.001, 0.05, len(data_df))
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, "S_0"]}}},
            solute_parameters={"C": {"C_old": 1.0, "k1": "k1_col", "C_eq": 0.0}},
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        wb = model.get_water_balance()
        assert np.abs(wb).max() < 1e-8
        sb = model.get_solute_balance("C")
        assert np.abs(sb).max() < 1e-8
        # Check outputs are finite
        C_pred = model.data_df["C --> Q"].values
        assert np.all(np.isfinite(C_pred))

    def test_reaction_with_varying_Ceq(self):
        """Equilibrium concentration read from a column."""
        data_df = _make_unsteady_data()
        data_df["Ceq_col"] = np.linspace(0, 5, len(data_df))
        model = Model(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, "S_0"]}}},
            solute_parameters={"C": {"C_old": 1.0, "k1": 0.02, "C_eq": "Ceq_col"}},
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        wb = model.get_water_balance()
        assert np.abs(wb).max() < 1e-8
        sb = model.get_solute_balance("C")
        assert np.abs(sb).max() < 1e-8


# ---------------------------------------------------------------------------
# Tests: Multiple fluxes with different time-varying SAS
# ---------------------------------------------------------------------------


class TestTimeVaryingMultipleFluxes:
    """Multiple fluxes where each has time-varying SAS parameters."""

    def test_two_fluxes_different_varying_params(self):
        data_df = _make_unsteady_data()
        data_df["Q1"] = data_df["Q"] * 0.4
        data_df["Q2"] = data_df["Q"] * 0.6
        data_df["S_0_half"] = data_df["S_0"] * 0.5

        model = Model(
            data_df,
            sas_specs={
                "Q1": {
                    "Q1_SAS": {
                        "func": "gamma",
                        "args": {"a": 1.0, "scale": "S_0", "loc": 0.0},
                    }
                },
                "Q2": {
                    "Q2_SAS": {"ST": [0, "S_0_half", "S_0"]},
                },
            },
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        wb = model.get_water_balance()
        assert np.abs(wb).max() < 1e-8
        sb = model.get_solute_balance("C")
        assert np.abs(sb).max() < 1e-8


# ---------------------------------------------------------------------------
# Tests: Multiple solutes
# ---------------------------------------------------------------------------


class TestMultipleSolutes:
    """Multiple solutes tracked simultaneously with time-varying parameters."""

    def test_two_solutes_different_reactions(self):
        data_df = _make_unsteady_data()
        data_df["C1"] = np.sin(np.arange(len(data_df)) * 0.1) + 2
        data_df["C2"] = np.cos(np.arange(len(data_df)) * 0.05) + 3
        model = Model(
            data_df,
            sas_specs={
                "Q": {
                    "Q_SAS": {
                        "func": "gamma",
                        "args": {"a": 1.5, "scale": "S_0", "loc": 0.0},
                    }
                }
            },
            solute_parameters={
                "C1": {"C_old": 2.0},
                "C2": {"C_old": 3.0, "k1": 0.01, "C_eq": 1.0},
            },
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        wb = model.get_water_balance()
        assert np.abs(wb).max() < 1e-8
        for sol in ["C1", "C2"]:
            sb = model.get_solute_balance(sol)
            assert np.abs(sb).max() < 1e-8
            C_pred = model.data_df[f"{sol} --> Q"].values
            assert np.all(np.isfinite(C_pred))
