"""Regression tests for the 2.0.2 rough-edge fixes found during workshop prep.

Covers: SAS parameter validation (NaN / non-positive), the _gammad NaN
guard, vis support for built-in continuous SAS functions, friendlier
construction errors, data_df reassignment warning, alpha defaults,
weights checks, observations string form, and the MESAS_PARALLEL=0
cache-collision fix.
"""

import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from mesas.sas.model import Model


def make_df(n=30, **extra):
    df = pd.DataFrame({"J": 1.0, "Q": 1.0, "S_0": 5.0, "C": 0.5}, index=range(n))
    for k, v in extra.items():
        df[k] = v
    return df


GAMMA = {"func": "gamma", "args": {"a": 1.0, "scale": "S_0", "loc": 0.0}}


class TestSASParameterValidation:
    def test_nan_scale_raises_with_timestep(self):
        df = make_df()
        df.loc[7, "S_0"] = np.nan
        m = Model(df, sas_specs={"Q": {"g": GAMMA}}, solute_parameters={"C": {}}, dt=0.1)
        with pytest.raises(ValueError, match=r"'scale' of component 'g' \(flux 'Q'\).*\[7\]"):
            m.run()

    def test_negative_scale_raises(self):
        df = make_df()
        df.loc[4, "S_0"] = -33.7
        m = Model(df, sas_specs={"Q": {"g": GAMMA}}, dt=0.1)
        with pytest.raises(ValueError, match="must be > 0"):
            m.run()

    def test_zero_shape_raises(self):
        spec = {"func": "gamma", "args": {"a": 0.0, "scale": 5.0, "loc": 0.0}}
        m = Model(make_df(), sas_specs={"Q": {"g": spec}}, dt=0.1)
        with pytest.raises(ValueError, match="'a' of component 'g'.*must be > 0"):
            m.run()

    def test_opt_out(self):
        df = make_df()
        df.loc[4, "S_0"] = -33.7
        m = Model(df, sas_specs={"Q": {"g": GAMMA}}, dt=0.1, validate_inputs=False)
        m.run()  # no exception; output quality is the user's problem

    def test_gammad_nan_terminates(self):
        from mesas.sas import _solve_numba as sn

        assert np.isnan(sn._gammad(np.nan, 0.7))
        assert np.isnan(sn._gammad(1.0, np.nan))


class TestConstructionErrors:
    def test_missing_required_arg(self):
        spec = {"func": "gamma", "args": {"a": 1.0, "scale": 5.0}}
        with pytest.raises(ValueError, match=r"missing: \['loc'\]"):
            Model(make_df(), sas_specs={"Q": {"g": spec}}, dt=0.1)

    def test_unknown_builtin_name_mentions_case(self):
        spec = {"func": "Gamma", "args": {"a": 1.0, "scale": 5.0, "loc": 0.0}}
        with pytest.raises(ValueError, match="case-sensitive"):
            Model(make_df(), sas_specs={"Q": {"g": spec}}, dt=0.1)

    def test_nan_in_st_column_mentions_nan(self):
        df = make_df()
        df.loc[5, "S_0"] = np.nan
        with pytest.raises(ValueError, match="finite"):
            Model(df, sas_specs={"Q": {"g": {"ST": [0.0, "S_0"]}}}, dt=0.1)

    def test_misspelt_parameter_column(self):
        spec = {"func": "gamma", "args": {"a": 1.0, "scale": "S_oops", "loc": 0.0}}
        with pytest.raises(ValueError, match="S_oops.*Available columns"):
            Model(make_df(), sas_specs={"Q": {"g": spec}}, dt=0.1)

    def test_missing_weight_column(self):
        specs = {"Q": {"g1": GAMMA, "g2": {"ST": [0, 5.0]}}}
        with pytest.raises(ValueError, match="blending weights.*'g1'"):
            Model(make_df(), sas_specs=specs, dt=0.1)


class TestVisBuiltinContinuous:
    @pytest.fixture()
    def gamma_model(self):
        m = Model(
            make_df(),
            sas_specs={"Q": {"g": GAMMA}},
            solute_parameters={"C": {"C_old": 1.0}},
            dt=0.1,
            record_state=True,
        )
        m.run()
        return m

    def test_plot_sas_cumulative(self, gamma_model):
        import mesas.utils.vis as vis

        vis.plot_SAS_cumulative(gamma_model, "Q")

    def test_plot_transport_column_with_timeseries(self, gamma_model):
        import mesas.utils.vis as vis

        vis.plot_transport_column_with_timeseries(gamma_model, "Q", "C")

    def test_repr(self, gamma_model):
        fn = gamma_model.sas_specs["Q"].components["g"].sas_fun[0]
        assert "gamma" in repr(fn)

    def test_grid_is_finite(self, gamma_model):
        fn = gamma_model.sas_specs["Q"].components["g"].sas_fun[0]
        assert np.all(np.isfinite(fn.ST))
        assert np.all(np.isfinite(fn.P))


class TestUsabilityGuards:
    def test_data_df_reassignment_warns(self):
        m = Model(make_df(), sas_specs={"Q": {"g": GAMMA}}, dt=0.1)
        with pytest.warns(UserWarning, match="SAS parameters"):
            m.data_df = make_df()

    def test_alpha_missing_flux_defaults_to_one(self):
        df = make_df(Q2=0.5)
        df["Q"] = 0.5
        m = Model(
            df,
            sas_specs={"Q": {"a": {"ST": [0, 5.0]}}, "Q2": {"b": {"ST": [0, 5.0]}}},
            solute_parameters={"C": {"alpha": {"Q": 1.0}}},  # Q2 omitted
            dt=0.1,
        )
        m.run()

    def test_weights_not_summing_to_one_warns(self):
        df = make_df(g1=0.3, g2=0.3)  # sums to 0.6
        m = Model(
            df,
            sas_specs={"Q": {"g1": {"ST": [0, 5.0]}, "g2": {"ST": [0, 2.5]}}},
            dt=0.1,
        )
        with pytest.warns(UserWarning, match="do not sum to 1"):
            m.run()

    def test_observations_string_single_flux(self):
        df = make_df(C_obs=0.4)
        m = Model(
            df,
            sas_specs={"Q": {"g": {"ST": [0, 5.0]}}},
            solute_parameters={"C": {"observations": "C_obs"}},
            dt=0.1,
        )
        m.run()
        res = m.get_residuals()
        assert res is not None and len(res) == len(df)
        assert len(m.get_obs_index()) == len(df)

    def test_observations_string_multi_flux_raises(self):
        df = make_df(Q2=0.5)
        df["Q"] = 0.5
        with pytest.raises(ValueError, match="multiple fluxes"):
            Model(
                df,
                sas_specs={"Q": {"a": {"ST": [0, 5.0]}}, "Q2": {"b": {"ST": [0, 5.0]}}},
                solute_parameters={"C": {"observations": "C"}},
                dt=0.1,
            )


class TestMesasParallelCache:
    def test_serial_mode_runs_after_parallel_cache(self, tmp_path):
        """MESAS_PARALLEL=0 must run correctly (and not segfault) even when a
        parallel-compiled numba cache already exists: the serial kernel is
        compiled with cache=False so the modes can no longer collide."""
        script = tmp_path / "serial_run.py"
        script.write_text(
            "import pandas as pd\n"
            "from mesas.sas.model import Model\n"
            "df = pd.DataFrame({'J': 1.0, 'Q': 1.0, 'S_0': 5.0, 'C': 0.5}, index=range(10))\n"
            "m = Model(df, sas_specs={'Q': {'g': {'ST': [0, 5.0]}}},\n"
            "          solute_parameters={'C': {'C_old': 1.0}}, dt=0.1)\n"
            "m.run()\n"
            "print('SERIAL_OK')\n"
        )
        env = dict(os.environ, MESAS_PARALLEL="0")
        # The parent test session has already populated the parallel cache.
        out = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=600)
        assert out.returncode == 0, f"serial-mode subprocess failed: {out.stderr[-500:]}"
        assert "SERIAL_OK" in out.stdout
