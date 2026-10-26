"""Regression tests for Phase 4: input validation and error messages.

Covers IMPROVEMENT_PLAN items A11 (silent acceptance of invalid inputs) and
A12 (cryptic error messages).
"""

import numpy as np
import pandas as pd
import pytest

from mesas.sas.functions import Piecewise
from mesas.sas.model import Model


def _df(n=30, **overrides):
    data = {
        "J": np.ones(n),
        "Q": np.ones(n),
        "C": np.linspace(1.0, 2.0, n),
    }
    data.update(overrides)
    return pd.DataFrame(data)


SPEC = {"Q": {"u": {"ST": [0.0, 10.0]}}}
SOL = {"C": {"C_old": 1.0}}


class TestValidateInputs:
    def test_nan_in_influx_raises(self):
        J = np.ones(30)
        J[7] = np.nan
        m = Model(data_df=_df(J=J), sas_specs=SPEC, solute_parameters=SOL, verbose=False)
        with pytest.raises(ValueError, match=r"'J' contains 1 NaN.*index 7"):
            m.run()

    def test_nan_in_solute_raises(self):
        C = np.ones(30)
        C[3:5] = np.nan
        m = Model(data_df=_df(C=C), sas_specs=SPEC, solute_parameters=SOL, verbose=False)
        with pytest.raises(ValueError, match=r"'C' contains 2 NaN"):
            m.run()

    def test_negative_flux_raises(self):
        Q = np.ones(30)
        Q[10] = -2.0
        m = Model(data_df=_df(Q=Q), sas_specs=SPEC, solute_parameters=SOL, verbose=False)
        with pytest.raises(ValueError, match=r"'Q' contains 1 negative"):
            m.run()

    def test_negative_concentration_allowed(self):
        # isotope delta values are legitimately negative
        C = np.full(30, -8.5)
        m = Model(data_df=_df(C=C), sas_specs=SPEC, solute_parameters=SOL, verbose=False)
        m.run()

    def test_opt_out(self):
        J = np.ones(30)
        J[7] = np.nan
        m = Model(data_df=_df(J=J), sas_specs=SPEC, solute_parameters=SOL, verbose=False, validate_inputs=False)
        m.run()  # runs (produces NaN, but explicitly allowed)


class TestConstructionErrors:
    def test_typo_option_kwarg_raises(self):
        with pytest.raises(TypeError, match="veborse"):
            Model(data_df=_df(), sas_specs=SPEC, veborse=True)

    def test_typo_config_option_raises(self):
        with pytest.raises(KeyError, match="n_substep"):
            Model(data_df=_df(), config={"sas_specs": SPEC, "options": {"n_substep": 5}})

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Model(data_df=pd.DataFrame({"J": [], "Q": []}), sas_specs=SPEC)

    def test_unknown_solute_column_raises(self):
        with pytest.raises(ValueError, match="Solute 'X'"):
            Model(data_df=_df(), sas_specs=SPEC, solute_parameters={"X": {"C_old": 1.0}})

    def test_missing_param_column_raises_clearly(self):
        m = Model(
            data_df=_df(),
            sas_specs=SPEC,
            solute_parameters={"C": {"k1": "k1_missing"}},
            verbose=False,
        )
        with pytest.raises(ValueError, match=r"'k1_missing' given for parameter 'k1' of solute 'C'"):
            m.run()

    def test_bad_record_state_column_raises(self):
        with pytest.raises(ValueError, match="record_state column 'nope'"):
            Model(data_df=_df(), sas_specs=SPEC, record_state="nope")

    def test_bad_record_state_type_raises(self):
        with pytest.raises(TypeError, match="record_state"):
            Model(data_df=_df(), sas_specs=SPEC, record_state=3.5)


class TestSasSpecsSetter:
    def test_raw_dict_assignment_is_parsed(self):
        m = Model(data_df=_df(), sas_specs=SPEC, solute_parameters=SOL, verbose=False)
        m.sas_specs = {"Q": {"u": {"ST": [0.0, 20.0]}}}
        m.run()
        assert "C --> Q" in m.data_df

    def test_caller_dict_not_mutated(self):
        spec = {"Q": {"u": {"ST": [0.0, 10.0]}}}
        Model(data_df=_df(), sas_specs=spec, verbose=False)
        # the caller's nested dict must still be a plain dict spec
        assert spec["Q"] == {"u": {"ST": [0.0, 10.0]}}


class TestPiecewiseConstruction:
    def test_decreasing_ST_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            Piecewise(ST=[0.0, 5.0, 3.0])

    def test_repeated_ST_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            Piecewise(ST=[0.0, 50.0, 50.0, 100.0])

    def test_model_with_bad_ST_raises(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            Model(data_df=_df(), sas_specs={"Q": {"u": {"ST": [0.0, 5.0, 3.0]}}})
