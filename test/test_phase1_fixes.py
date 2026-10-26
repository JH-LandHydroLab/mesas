"""Regression tests for the Phase 1 bug fixes in IMPROVEMENT_PLAN.md.

Covers (letters refer to plan items):
- A3: SAS parameter updates must reach every timestep, not just t=0
- A4: Model and ModelResult must survive pickle/deepcopy round-trips
- A5: recursive_split leftfirst mode must not call a nonexistent method
- A6: sT_init/max_age conflicts must raise instead of silently corrupting
- A8: Model.set_sas_fun must work (was assigning to a getter-only property)
- A9: options jacobian=True must raise instead of returning silent zeros
"""

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

from mesas.sas.functions import Piecewise
from mesas.sas.model import Model


def _simple_model(n=50, run=False, **kwargs):
    df = pd.DataFrame(
        {
            "J": np.ones(n),
            "Q": np.ones(n),
            "C": np.linspace(1.0, 2.0, n),
        }
    )
    m = Model(
        data_df=df,
        sas_specs={"Q": {"u": {"ST": [0.0, 10.0]}}},
        solute_parameters={"C": {"C_old": 1.0}},
        verbose=False,
        **kwargs,
    )
    if run:
        m.run()
    return m


class TestParameterUpdatePropagation:
    """A3: update_from_parameter_list must update all timesteps."""

    def test_argsS_updated_at_every_timestep(self):
        m = _simple_model()
        spec = m.sas_specs["Q"]
        p0 = spec.get_parameter_list()
        spec.update_from_parameter_list(p0 * 2.0)
        argsS = spec.components["u"].argsS
        # every timestep column must equal the updated t=0 column
        assert np.allclose(argsS, argsS[:, :1])
        assert np.allclose(argsS[:, -1], [0.0, 20.0])

    def test_model_output_changes_after_update(self):
        m = _simple_model(run=True)
        before = m.data_df["C --> Q"].to_numpy().copy()
        spec = m.sas_specs["Q"]
        spec.update_from_parameter_list(spec.get_parameter_list() * 5.0)
        m.run()
        after = m.data_df["C --> Q"].to_numpy()
        # a 5x larger storage must change late-time outputs, not just t=0
        assert not np.allclose(before[10:], after[10:])


class TestPickling:
    """A4: pickle/deepcopy round-trips must not recurse infinitely."""

    def test_pickle_model_before_run(self):
        m = _simple_model()
        m2 = pickle.loads(pickle.dumps(m))
        m2.run()
        assert "C --> Q" in m2.data_df

    def test_pickle_model_after_run(self):
        m = _simple_model(run=True)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(m.result.sT, m2.result.sT)

    def test_deepcopy_model(self):
        m = _simple_model(run=True)
        m2 = copy.deepcopy(m)
        np.testing.assert_array_equal(m.result.sT, m2.result.sT)

    def test_pickle_result(self):
        m = _simple_model(run=True)
        r2 = pickle.loads(pickle.dumps(m.result))
        np.testing.assert_array_equal(m.result.sT, r2.sT)

    def test_missing_attribute_raises_attributeerror(self):
        m = _simple_model(run=True)
        with pytest.raises(AttributeError):
            _ = m.result.no_such_result


class TestRecursiveSplitLeftfirst:
    """A5: the default leftfirst search mode must run to completion."""

    def test_run_leftfirst(self):
        pytest.importorskip("sklearn")
        from mesas.me import recursive_split

        n = 40
        rng = np.random.default_rng(7)
        df = pd.DataFrame(
            {
                "J": np.ones(n),
                "Q": np.ones(n),
                "C": 1.0 + rng.random(n),
            }
        )
        df["C obs"] = 1.0
        m = Model(
            data_df=df,
            sas_specs={"Q": {"u": {"ST": [0.0, 5.0]}}},
            solute_parameters={"C": {"C_old": 1.0, "observations": {"Q": "C obs"}}},
            verbose=False,
            record_state=True,
            components_to_learn={"Q": ["u"]},
        )
        # must not raise AttributeError: trim_unused_ST
        result = recursive_split.run(m, verbose=False, search_mode="leftfirst", jacobian_mode="numerical")
        assert result is not None


class TestSTInitValidation:
    """A6: sT_init longer than the timeseries must not silently corrupt."""

    def test_sT_init_longer_than_timeseries_raises(self):
        with pytest.raises(ValueError, match="max_age"):
            _simple_model(n=20, sT_init=np.ones(40), run=True)

    def test_conflicting_max_age_and_sT_init_raises(self):
        with pytest.raises(ValueError, match="conflicts"):
            _simple_model(n=20, max_age=5, sT_init=np.ones(15))

    def test_consistent_max_age_and_sT_init_ok(self):
        m = _simple_model(n=20, max_age=15, sT_init=np.ones(15), run=True)
        assert m.result.sT.shape[0] == 15


class TestSetSasFun:
    """A8: Model.set_sas_fun was assigning to a getter-only property."""

    def test_set_single_function(self):
        m = _simple_model()
        new_fun = Piecewise(ST=[0.0, 25.0])
        m.set_sas_fun("Q", "u", new_fun)
        comp = m.sas_specs["Q"].components["u"]
        assert np.allclose(comp.argsS[:, 0], [0.0, 25.0])
        assert np.allclose(comp.argsS[:, -1], [0.0, 25.0])
        m.run()

    def test_set_wrong_length_list_raises(self):
        m = _simple_model(n=20)
        with pytest.raises(ValueError, match="one per timestep"):
            m.set_sas_fun("Q", "u", [Piecewise(ST=[0.0, 25.0])] * 3)


class TestJacobianGuard:
    """A9: jacobian=True would silently return all-zero sensitivities."""

    def test_jacobian_option_raises(self):
        m = _simple_model(jacobian=True)
        with pytest.raises(NotImplementedError, match="jacobian"):
            m.run()
