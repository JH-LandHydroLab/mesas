"""
Stage 6: Systematic tests of all SAS function types.

Verifies that every supported SAS function type (piecewise, gamma, beta,
kumaraswamy) produces correct results across a range of parameters:
- CDF/inverse-CDF round-trips
- Boundary values (0 and 1)
- Model integration (water balance, mass balance)
- Cross-validation against scipy.stats reference implementations
"""

import numpy as np
import pandas as pd
import pytest

from mesas.sas.functions import Continuous, Piecewise
from mesas.sas.model import Model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(n=100, Q_0=1.0, S_0=5.0, S_m=0.0):
    """Create a steady-state DataFrame for model tests."""
    rng = np.random.default_rng(99)
    data_df = pd.DataFrame(index=range(n))
    data_df["J"] = Q_0
    data_df["Q"] = Q_0
    data_df["S_0"] = S_0
    data_df["S_m"] = S_m
    data_df["C"] = rng.uniform(0, 10, n)
    return data_df


def _run_model_check(data_df, sas_specs, solute_params=None, tolerance=1e-8):
    """Run a model and verify water balance + basic output sanity."""
    model = Model(
        data_df,
        sas_specs=sas_specs,
        solute_parameters=solute_params,
        dt=0.1,
        verbose=False,
        record_state=True,
    )
    model.run()

    # Water balance
    wb = model.get_water_balance()
    assert np.abs(wb).max() < tolerance, f"Water balance residual: {np.abs(wb).max()}"

    # sT non-negative
    sT = model.get_sT()
    assert np.all(sT >= -1e-12), f"Negative sT: min = {sT.min()}"

    # pQ non-negative
    pQ = model.get_pQ("Q")
    assert np.all(pQ >= -1e-12), f"Negative pQ: min = {pQ.min()}"

    # Finite outputs
    assert np.all(np.isfinite(sT))
    assert np.all(np.isfinite(pQ))

    if solute_params is not None:
        for sol in solute_params:
            C_pred = model.data_df[f"{sol} --> Q"].values
            assert np.all(np.isfinite(C_pred)), f"Non-finite {sol} output"
            sb = model.get_solute_balance(sol)
            assert np.abs(sb).max() < tolerance, f"Solute balance residual for {sol}: {np.abs(sb).max()}"

    return model


# ---------------------------------------------------------------------------
# Piecewise SAS function unit tests
# ---------------------------------------------------------------------------


class TestPiecewiseFunction:
    """Test the Piecewise SAS function class directly."""

    def test_uniform_cdf(self):
        f = Piecewise(ST=[0, 10])
        assert f(0.0) == pytest.approx(0.0)
        assert f(5.0) == pytest.approx(0.5)
        assert f(10.0) == pytest.approx(1.0)

    def test_uniform_inv(self):
        f = Piecewise(ST=[0, 10])
        assert f.inv(0.0) == pytest.approx(0.0)
        assert f.inv(0.5) == pytest.approx(5.0)
        assert f.inv(1.0) == pytest.approx(10.0)

    def test_roundtrip_cdf_inv(self):
        """CDF(inv(P)) should equal P."""
        f = Piecewise(ST=[0, 3, 8, 10], P=[0, 0.2, 0.7, 1.0])
        P_vals = np.linspace(0, 1, 50)
        ST_vals = f.inv(P_vals)
        P_recovered = f(ST_vals)
        np.testing.assert_allclose(P_recovered, P_vals, atol=1e-12)

    def test_multi_segment(self):
        """Non-uniform piecewise with custom P values."""
        f = Piecewise(ST=[0, 2, 5, 10], P=[0, 0.5, 0.8, 1.0])
        # Within first segment [0,2], P goes 0->0.5
        assert f(1.0) == pytest.approx(0.25)
        # Within second segment [2,5], P goes 0.5->0.8
        assert f(3.5) == pytest.approx(0.65)

    def test_cdf_clipping(self):
        """Values outside range should clip to 0 or 1."""
        f = Piecewise(ST=[1, 5])
        assert f(-1.0) == pytest.approx(0.0)
        assert f(10.0) == pytest.approx(1.0)

    def test_subdivided_copy(self):
        f = Piecewise(ST=[0, 10])
        f2 = f.subdivided_copy(0, 0.5)
        assert f2.nsegment == 2
        assert f2.ST[1] == pytest.approx(5.0)
        assert f2.P[1] == pytest.approx(0.5)

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="at least 2"):
            Piecewise(ST=[5])
        with pytest.raises(ValueError, match="P must start at 0"):
            Piecewise(ST=[0, 10], P=[0.1, 1.0])
        with pytest.raises(ValueError, match="P must end at 1"):
            Piecewise(ST=[0, 10], P=[0, 0.9])
        with pytest.raises(ValueError, match="non-decreasing"):
            Piecewise(ST=[0, 5, 10, 15], P=[0, 0.8, 0.5, 1.0])

    def test_parameter_list_roundtrip(self):
        """Setting parameter_list should update ST consistently."""
        f = Piecewise(ST=[2, 5, 10])
        original_ST = f.ST.copy()
        pl = f.parameter_list.copy()
        f.parameter_list = pl
        np.testing.assert_allclose(f.ST, original_ST)


# ---------------------------------------------------------------------------
# Continuous SAS function unit tests
# ---------------------------------------------------------------------------


class TestContinuousFunction:
    """Test the Continuous SAS function class directly."""

    def test_gamma_cdf_matches_scipy(self):
        from scipy.stats import gamma as scipy_gamma

        f = Continuous("builtin", "gamma", {"a": 2.0, "loc": 0.0, "scale": 5.0})
        ref = scipy_gamma(a=2.0, loc=0.0, scale=5.0)
        ST_vals = np.linspace(0, 30, 50)
        np.testing.assert_allclose(f(ST_vals), ref.cdf(ST_vals), atol=1e-12)

    def test_beta_cdf_matches_scipy(self):
        from scipy.stats import beta as scipy_beta

        f = Continuous("builtin", "beta", {"a": 2.0, "b": 3.0, "loc": 0.0, "scale": 10.0})
        ref = scipy_beta(a=2.0, b=3.0, loc=0.0, scale=10.0)
        ST_vals = np.linspace(0, 10, 50)
        np.testing.assert_allclose(f(ST_vals), ref.cdf(ST_vals), atol=1e-12)

    def test_kumaraswamy_cdf(self):
        f = Continuous("builtin", "kumaraswamy", {"a": 1.0, "b": 2.0, "loc": 0.0, "scale": 10.0})
        # Kumaraswamy CDF: 1 - (1 - (x/scale)^a)^b for x in [0, scale]
        x = 5.0
        expected = 1 - (1 - (x / 10.0) ** 1.0) ** 2.0
        assert f(x) == pytest.approx(expected, abs=1e-6)

    def test_gamma_inv_matches_scipy(self):
        from scipy.stats import gamma as scipy_gamma

        f = Continuous("builtin", "gamma", {"a": 1.5, "loc": 0.0, "scale": 3.0})
        ref = scipy_gamma(a=1.5, loc=0.0, scale=3.0)
        P_vals = np.linspace(0.01, 0.99, 50)
        np.testing.assert_allclose(f.inv(P_vals), ref.ppf(P_vals), atol=1e-10)

    def test_scipy_stats_mode(self):
        """Continuous with use='scipy.stats' and a string function name."""
        f = Continuous(
            "scipy.stats",
            "gamma",
            {"a": 2.0, "loc": 0.0, "scale": 5.0},
            nsegment=50,
        )
        # Should have a piecewise lookup table
        assert len(f.ST) > 2
        assert f(0.0) == pytest.approx(0.0, abs=1e-6)

    def test_builtin_argsS_gamma(self):
        f = Continuous("builtin", "gamma", {"a": 2.0, "loc": 1.0, "scale": 5.0})
        assert f.argsS == [1.0, 5.0, 2.0]  # [loc, scale, a]

    def test_builtin_argsS_beta(self):
        f = Continuous("builtin", "beta", {"a": 2.0, "b": 3.0, "loc": 0.0, "scale": 10.0})
        assert f.argsS == [0.0, 10.0, 2.0, 3.0]  # [loc, scale, a, b]

    def test_builtin_argsS_kumaraswamy(self):
        f = Continuous("builtin", "kumaraswamy", {"a": 1.5, "b": 2.5, "loc": 0.0, "scale": 10.0})
        assert f.argsS == [0.0, 10.0, 1.5, 2.5]  # [loc, scale, a, b]


# ---------------------------------------------------------------------------
# Integrated model tests for each SAS type
# ---------------------------------------------------------------------------


class TestPiecewiseModel:
    """Run model with piecewise SAS and check conservation."""

    @pytest.mark.parametrize("n_segments", [1, 2, 3, 5])
    def test_uniform_segments(self, n_segments):
        data_df = _make_data()
        ST = np.linspace(0, 5.0, n_segments + 1).tolist()
        _run_model_check(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": ST}}},
            solute_params={"C": {"C_old": 1.0}},
        )

    def test_young_biased_piecewise(self):
        """Piecewise with steep rise near young water."""
        data_df = _make_data()
        _run_model_check(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 1, 5], "P": [0, 0.8, 1.0]}}},
            solute_params={"C": {"C_old": 1.0}},
        )

    def test_old_biased_piecewise(self):
        """Piecewise with steep rise near old water."""
        data_df = _make_data()
        _run_model_check(
            data_df,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 4, 5], "P": [0, 0.2, 1.0]}}},
            solute_params={"C": {"C_old": 1.0}},
        )


class TestGammaModel:
    """Run model with gamma SAS and check conservation."""

    @pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 5.0])
    def test_gamma_shape_parameter(self, a):
        data_df = _make_data()
        _run_model_check(
            data_df,
            sas_specs={
                "Q": {
                    "Q_SAS": {
                        "func": "gamma",
                        "args": {"a": a, "scale": "S_0", "loc": 0.0},
                    }
                }
            },
            solute_params={"C": {"C_old": 1.0}},
        )

    def test_gamma_with_loc(self):
        """Gamma with nonzero loc (minimum age)."""
        data_df = _make_data(S_m=1.0)
        _run_model_check(
            data_df,
            sas_specs={
                "Q": {
                    "Q_SAS": {
                        "func": "gamma",
                        "args": {"a": 1.5, "scale": "S_0", "loc": "S_m"},
                    }
                }
            },
            solute_params={"C": {"C_old": 1.0}},
        )


class TestBetaModel:
    """Run model with beta SAS and check conservation."""

    @pytest.mark.parametrize("a,b", [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (2.0, 5.0)])
    def test_beta_shape_parameters(self, a, b):
        data_df = _make_data()
        _run_model_check(
            data_df,
            sas_specs={
                "Q": {
                    "Q_SAS": {
                        "func": "beta",
                        "args": {"a": a, "b": b, "scale": "S_0", "loc": 0.0},
                    }
                }
            },
            solute_params={"C": {"C_old": 1.0}},
        )


class TestKumaraswamyModel:
    """Run model with Kumaraswamy SAS and check conservation."""

    @pytest.mark.parametrize("a,b", [(1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (0.5, 3.0)])
    def test_kumaraswamy_shape_parameters(self, a, b):
        data_df = _make_data()
        _run_model_check(
            data_df,
            sas_specs={
                "Q": {
                    "Q_SAS": {
                        "func": "kumaraswamy",
                        "args": {"a": a, "b": b, "scale": "S_0", "loc": 0.0},
                    }
                }
            },
            solute_params={"C": {"C_old": 1.0}},
        )


class TestScipyStatsModel:
    """Run model with scipy.stats-backed SAS (piecewise approximation)."""

    def test_scipy_gamma(self):
        data_df = _make_data()
        _run_model_check(
            data_df,
            sas_specs={
                "Q": {
                    "Q_SAS": {
                        "scipy.stats": "gamma",
                        "args": {"a": 2.0, "scale": "S_0", "loc": 0.0},
                    }
                }
            },
            solute_params={"C": {"C_old": 1.0}},
        )

    def test_scipy_beta(self):
        data_df = _make_data()
        _run_model_check(
            data_df,
            sas_specs={
                "Q": {
                    "Q_SAS": {
                        "scipy.stats": "beta",
                        "args": {"a": 2.0, "b": 3.0, "scale": "S_0", "loc": 0.0},
                    }
                }
            },
            solute_params={"C": {"C_old": 1.0}},
        )


class TestMultipleFluxes:
    """Test models with more than one outflux."""

    @pytest.mark.parametrize("func_type", ["piecewise", "gamma", "kumaraswamy"])
    def test_two_fluxes_water_balance(self, func_type):
        data_df = _make_data()
        data_df["Q1"] = 0.4
        data_df["Q2"] = 0.6

        if func_type == "piecewise":
            specs = {
                "Q1": {"Q1_SAS": {"ST": [0, 5.0]}},
                "Q2": {"Q2_SAS": {"ST": [0, 3.0, 5.0], "P": [0, 0.7, 1.0]}},
            }
        elif func_type == "gamma":
            specs = {
                "Q1": {"Q1_SAS": {"func": "gamma", "args": {"a": 1.0, "scale": "S_0", "loc": 0.0}}},
                "Q2": {"Q2_SAS": {"func": "gamma", "args": {"a": 2.0, "scale": "S_0", "loc": 0.0}}},
            }
        else:  # kumaraswamy
            specs = {
                "Q1": {"Q1_SAS": {"func": "kumaraswamy", "args": {"a": 1.0, "b": 2.0, "scale": "S_0", "loc": 0.0}}},
                "Q2": {"Q2_SAS": {"func": "kumaraswamy", "args": {"a": 2.0, "b": 1.0, "scale": "S_0", "loc": 0.0}}},
            }

        model = Model(
            data_df,
            sas_specs=specs,
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


class TestReactions:
    """Verify solute reactions work with all SAS types."""

    @pytest.mark.parametrize(
        "spec",
        [
            {"ST": [0, 5.0]},
            {"func": "gamma", "args": {"a": 1.5, "scale": "S_0", "loc": 0.0}},
            {"func": "kumaraswamy", "args": {"a": 1.0, "b": 2.0, "scale": "S_0", "loc": 0.0}},
        ],
        ids=["piecewise", "gamma", "kumaraswamy"],
    )
    def test_first_order_reaction(self, spec):
        data_df = _make_data()
        _run_model_check(
            data_df,
            sas_specs={"Q": {"Q_SAS": spec}},
            solute_params={"C": {"C_old": 5.0, "k1": 0.05, "C_eq": 0.0}},
        )

    @pytest.mark.parametrize(
        "spec",
        [
            {"ST": [0, 5.0]},
            {"func": "gamma", "args": {"a": 1.5, "scale": "S_0", "loc": 0.0}},
        ],
        ids=["piecewise", "gamma"],
    )
    def test_reaction_with_nonzero_Ceq(self, spec):
        data_df = _make_data()
        _run_model_check(
            data_df,
            sas_specs={"Q": {"Q_SAS": spec}},
            solute_params={"C": {"C_old": 1.0, "k1": 0.02, "C_eq": 3.0}},
        )
