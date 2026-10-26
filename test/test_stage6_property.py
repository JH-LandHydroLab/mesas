"""
Stage 6: Property-based tests using Hypothesis.

These tests verify fundamental invariants that must hold regardless of
model configuration:
1. Water balance closure (conservation of mass)
2. Non-negativity of age-ranked storage
3. Probability constraints on pQ
4. Finiteness of all outputs
5. Solute balance closure
"""

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mesas.sas.model import Model

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def steady_data_st(draw, min_length=20, max_length=100):
    """Generate a valid steady-state DataFrame."""
    n = draw(st.integers(min_value=min_length, max_value=max_length))
    Q_0 = draw(st.floats(min_value=0.5, max_value=5.0))
    S_0 = draw(st.floats(min_value=1.0, max_value=20.0))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)

    data_df = pd.DataFrame(index=range(n))
    data_df["J"] = Q_0
    data_df["Q"] = Q_0
    data_df["S_0"] = S_0
    data_df["C"] = rng.uniform(0, 10, n)
    return data_df


@st.composite
def piecewise_spec_st(draw, S_0_range=(1.0, 20.0)):
    """Generate a valid piecewise-linear SAS spec."""
    n_segments = draw(st.integers(min_value=1, max_value=5))
    S_max = draw(st.floats(min_value=S_0_range[0], max_value=S_0_range[1]))
    ST = np.linspace(0, S_max, n_segments + 1).tolist()
    return {"Q": {"Q_SAS": {"ST": ST}}}


@st.composite
def gamma_spec_st(draw):
    """Generate a valid gamma SAS spec."""
    a = draw(st.floats(min_value=0.3, max_value=3.0))
    return {
        "Q": {
            "Q_SAS": {
                "func": "gamma",
                "args": {"a": a, "scale": "S_0", "loc": 0.0},
            }
        }
    }


@st.composite
def sas_spec_st(draw):
    """Generate either a piecewise or gamma SAS spec."""
    use_gamma = draw(st.booleans())
    if use_gamma:
        return draw(gamma_spec_st())
    else:
        return draw(piecewise_spec_st())


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestWaterBalanceInvariant:
    """Water balance must close regardless of configuration."""

    @given(data=steady_data_st(), spec=piecewise_spec_st())
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_water_balance_piecewise(self, data, spec):
        model = Model(
            data,
            sas_specs=spec,
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        wb = model.get_water_balance()
        assert np.abs(wb).max() < 1e-8, f"Water balance residual: {np.abs(wb).max()}"

    @given(data=steady_data_st(), spec=gamma_spec_st())
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_water_balance_gamma(self, data, spec):
        model = Model(
            data,
            sas_specs=spec,
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        wb = model.get_water_balance()
        assert np.abs(wb).max() < 1e-8, f"Water balance residual: {np.abs(wb).max()}"


class TestStorageNonNegativity:
    """Age-ranked storage sT must be non-negative."""

    @given(data=steady_data_st(), spec=sas_spec_st())
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_sT_non_negative(self, data, spec):
        model = Model(
            data,
            sas_specs=spec,
            dt=0.1,
            verbose=False,
        )
        model.run()
        sT = model.get_sT()
        assert np.all(sT >= -1e-12), f"Negative sT: min = {sT.min()}"


class TestProbabilityConstraints:
    """Discharge probabilities must satisfy basic constraints."""

    @given(data=steady_data_st(), spec=sas_spec_st())
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_pQ_non_negative(self, data, spec):
        model = Model(
            data,
            sas_specs=spec,
            dt=0.1,
            verbose=False,
        )
        model.run()
        pQ = model.get_pQ("Q")
        assert np.all(pQ >= -1e-12), f"Negative pQ: min = {pQ.min()}"

    @given(data=steady_data_st(), spec=sas_spec_st())
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_pQ_cumulative_bounded(self, data, spec):
        """Sum of pQ * dt over ages should not exceed 1."""
        model = Model(
            data,
            sas_specs=spec,
            dt=0.1,
            verbose=False,
        )
        model.run()
        pQ = model.get_pQ("Q")
        pQ_sum = np.sum(pQ, axis=0) * 0.1
        assert np.all(pQ_sum <= 1.0 + 1e-8), f"pQ sum exceeds 1: max = {pQ_sum.max()}"


class TestOutputFiniteness:
    """All model outputs must be finite."""

    @given(data=steady_data_st(), spec=sas_spec_st())
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_concentrations_finite(self, data, spec):
        model = Model(
            data,
            sas_specs=spec,
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            verbose=False,
        )
        model.run()
        C_pred = model.data_df["C --> Q"].values
        assert np.all(np.isfinite(C_pred)), f"Non-finite C: {C_pred[~np.isfinite(C_pred)]}"

    @given(data=steady_data_st(), spec=sas_spec_st())
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_sT_finite(self, data, spec):
        model = Model(
            data,
            sas_specs=spec,
            dt=0.1,
            verbose=False,
        )
        model.run()
        sT = model.get_sT()
        assert np.all(np.isfinite(sT)), "Non-finite sT found"


class TestSoluteBalanceInvariant:
    """Solute balance must close when record_state=True."""

    @given(data=steady_data_st(min_length=20, max_length=60))
    @settings(
        max_examples=15,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_solute_balance_uniform(self, data):
        model = Model(
            data,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        sb = model.get_solute_balance("C")
        assert np.abs(sb).max() < 1e-8, f"Solute balance residual: {np.abs(sb).max()}"

    @given(
        data=steady_data_st(min_length=20, max_length=60),
        k1=st.floats(min_value=0.001, max_value=0.1),
        C_eq=st.floats(min_value=0.0, max_value=10.0),
    )
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_solute_balance_with_reaction(self, data, k1, C_eq):
        model = Model(
            data,
            sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
            solute_parameters={"C": {"C_old": 1.0, "k1": k1, "C_eq": C_eq}},
            dt=0.1,
            verbose=False,
            record_state=True,
        )
        model.run()
        sb = model.get_solute_balance("C")
        assert np.abs(sb).max() < 1e-8, f"Solute balance residual: {np.abs(sb).max()}"


class TestNumericalSchemeConsistency:
    """Different RK schemes should give similar results for smooth problems."""

    @given(data=steady_data_st(min_length=30, max_length=60))
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_schemes_agree_on_water_balance(self, data):
        """All three schemes should close water balance."""
        for scheme in [1, 2, 4]:
            model = Model(
                data,
                sas_specs={"Q": {"Q_SAS": {"ST": [0, 5.0]}}},
                dt=0.1,
                num_scheme=scheme,
                verbose=False,
                record_state=True,
            )
            model.run()
            wb = model.get_water_balance()
            assert np.abs(wb).max() < 1e-8, f"Scheme {scheme}: water balance residual {np.abs(wb).max()}"
