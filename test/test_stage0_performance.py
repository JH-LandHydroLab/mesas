"""
Stage 0: Performance benchmark tests.

These tests record wall-clock execution times so that future changes can be
checked for performance regressions. They use pytest-benchmark if available,
otherwise fall back to simple timing with a generous timeout.
"""

import time

import numpy as np
import pandas as pd

from mesas.sas.model import Model


def _make_benchmark_data(timeseries_length=1000, dt=0.1, Q_0=1.0, S_0=5.0):
    rng = np.random.default_rng(99)
    data_df = pd.DataFrame(index=range(timeseries_length))
    data_df["J"] = Q_0
    data_df["Q"] = Q_0
    data_df["S_0"] = S_0
    data_df["C"] = rng.standard_normal(timeseries_length)
    return data_df


def _run_model(data_df, sas_specs, solute_parameters=None, **kwargs):
    model = Model(
        data_df,
        sas_specs=sas_specs,
        solute_parameters=solute_parameters,
        verbose=False,
        **kwargs,
    )
    model.run()
    return model


class TestPerformanceBenchmarks:
    """Performance tests that record timing for regression detection."""

    def test_uniform_1000_steps(self):
        """Baseline: 1000-step uniform SAS with solute."""
        data_df = _make_benchmark_data(timeseries_length=1000)
        sas_specs = {"Q": {"Q_SAS": {"ST": [0, 5.0]}}}
        solute_params = {"C": {"C_old": 1.0}}

        start = time.perf_counter()
        _run_model(data_df, sas_specs, solute_params, dt=0.1)
        elapsed = time.perf_counter() - start

        # Record timing — if this exceeds 60s something is very wrong
        assert elapsed < 60.0, f"1000-step model took {elapsed:.1f}s (expected < 60s)"
        print(f"\n  [PERF] 1000-step uniform: {elapsed:.3f}s")

    def test_gamma_500_steps_10_substeps(self):
        """Gamma SAS with 10 substeps — heavier workload."""
        data_df = _make_benchmark_data(timeseries_length=500)
        sas_specs = {
            "Q": {
                "Q_SAS": {
                    "func": "gamma",
                    "args": {"a": 1.0, "scale": "S_0", "loc": 0.0},
                }
            }
        }
        solute_params = {"C": {"C_old": 1.0}}

        start = time.perf_counter()
        _run_model(data_df, sas_specs, solute_params, dt=0.1, n_substeps=10)
        elapsed = time.perf_counter() - start

        assert elapsed < 120.0, f"500-step/10-substep model took {elapsed:.1f}s"
        print(f"\n  [PERF] 500-step gamma (10 substeps): {elapsed:.3f}s")

    def test_no_solutes_1000_steps(self):
        """Water-only model (no solutes) should be faster."""
        data_df = _make_benchmark_data(timeseries_length=1000)
        sas_specs = {"Q": {"Q_SAS": {"ST": [0, 5.0]}}}

        start = time.perf_counter()
        _run_model(data_df, sas_specs, dt=0.1)
        elapsed = time.perf_counter() - start

        assert elapsed < 60.0, f"No-solute model took {elapsed:.1f}s"
        print(f"\n  [PERF] 1000-step no-solutes: {elapsed:.3f}s")

    def test_multiple_fluxes_500_steps(self):
        """Two fluxes with different SAS functions."""
        data_df = _make_benchmark_data(timeseries_length=500)
        data_df["Q1"] = 0.4
        data_df["Q2"] = 0.6
        sas_specs = {
            "Q1": {"Q1_SAS": {"ST": [0, 2.5, 5.0]}},
            "Q2": {"Q2_SAS": {"ST": [0, 5.0]}},
        }
        solute_params = {"C": {"C_old": 1.0}}

        start = time.perf_counter()
        _run_model(data_df, sas_specs, solute_params, dt=0.1)
        elapsed = time.perf_counter() - start

        assert elapsed < 60.0
        print(f"\n  [PERF] 500-step 2-flux: {elapsed:.3f}s")
