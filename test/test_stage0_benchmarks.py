"""
Stage 0: Steady-state analytical benchmark tests.

These tests compare mesas model output against known analytical solutions for
steady-state systems with various SAS function types. They establish numerical
accuracy baselines that must be maintained through all refactoring stages.

Each test uses a steady-state configuration where analytical solutions exist,
runs the model, and asserts that the RMS error is below a defined tolerance.
"""

import numpy as np
import pandas as pd
import pytest

from mesas.sas.model import Model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rms(x):
    return np.sqrt(np.mean(x**2))


def _make_steady_data(timeseries_length=500, dt=0.1, Q_0=1.0, S_0=5.0, S_m=1.0):
    """Create a steady-state DataFrame with random tracer input."""
    rng = np.random.default_rng(42)
    C_J = rng.standard_normal(timeseries_length)

    data_df = pd.DataFrame(index=range(timeseries_length))
    data_df["t"] = data_df.index * dt
    data_df["J"] = Q_0
    data_df["S_0"] = S_0
    data_df["S_m"] = S_m
    data_df["S_m0"] = S_m + S_0
    data_df["C"] = C_J
    return data_df


def _compute_benchmark_concentration(data_df, pQdisc, C_old, S_m, dt):
    """Convolve tracer input with transit time distribution to get benchmark output."""
    timeseries_length = len(data_df)
    C_J = data_df["C"].values
    im = int(S_m / (data_df["J"].iloc[0] * dt))  # index of minimum age
    benchmark = np.full(timeseries_length, C_old, dtype=float)
    conv = np.convolve(C_J, pQdisc, mode="full")[: timeseries_length - im] * dt
    old_frac = C_old * (1 - np.cumsum(pQdisc)[: timeseries_length - im] * dt)
    benchmark[im:] = conv + old_frac
    return benchmark


# ---------------------------------------------------------------------------
# Analytical transit-time distributions for steady-state SAS functions
# ---------------------------------------------------------------------------

STEADY_BENCHMARKS = {
    "uniform": {
        "spec": {"ST": ["S_m", "S_m0"]},
        "pQdisc": lambda d, i: (-1 + np.exp(d)) ** 2 / (np.exp((1 + i) * d) * d),
        "pQdisc0": lambda d: (1 + np.exp(d) * (-1 + d)) / (np.exp(d) * d),
    },
    "exponential_gamma": {
        "spec": {
            "func": "gamma",
            "args": {"a": 1.0 - 1e-5, "scale": "S_0", "loc": "S_m"},
        },
        "pQdisc": lambda d, i: (2 * np.log(1 + i * d) - np.log((1 + (-1 + i) * d) * (1 + d + i * d))) / d,
        "pQdisc0": lambda d: (d + np.log(1 / (1 + d))) / d,
    },
    "kumaraswamy_young_biased": {
        "spec": {
            "func": "kumaraswamy",
            "args": {"a": 1.0 - 1e-9, "b": 2.0 - 1e-9, "scale": "S_0", "loc": "S_m"},
        },
        "pQdisc": lambda d, i: (2 * d) / ((1 + (-1 + i) * d) * (1 + i * d) * (1 + d + i * d)),
        "pQdisc0": lambda d: d / (1 + d),
    },
}


@pytest.fixture
def steady_data():
    return _make_steady_data()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSteadyStateBenchmarks:
    """Test model accuracy against analytical steady-state solutions."""

    TOLERANCE = 1e-2  # RMS error tolerance
    C_OLD = 1.0
    DT = 0.1
    N_SUBSTEPS = 10
    Q_0 = 1.0
    S_0 = 5.0
    S_M = 1.0

    @pytest.mark.parametrize("name", list(STEADY_BENCHMARKS.keys()))
    def test_steady_benchmark(self, steady_data, name):
        bm = STEADY_BENCHMARKS[name]
        data_df = steady_data.copy()
        timeseries_length = len(data_df)
        delta = self.DT * self.Q_0 / self.S_0

        # Compute analytical transit-time distribution
        i = np.arange(timeseries_length)
        pQdisc = np.zeros(timeseries_length, dtype=float)
        pQdisc[0] = bm["pQdisc0"](delta) / self.DT
        pQdisc[1:] = bm["pQdisc"](delta, i[1:]) / self.DT

        # Compute benchmark concentration
        benchmark_C = _compute_benchmark_concentration(data_df, pQdisc, self.C_OLD, self.S_M, self.DT)

        # Set up and run the model
        flux_name = f"Q_{name}"
        data_df[flux_name] = self.Q_0

        sas_specs = {flux_name: {f"{name}_SAS": bm["spec"]}}
        solute_parameters = {"C": {"C_old": self.C_OLD}}

        model = Model(
            data_df,
            sas_specs=sas_specs,
            solute_parameters=solute_parameters,
            dt=self.DT,
            n_substeps=self.N_SUBSTEPS,
            max_age=timeseries_length,
            verbose=False,
        )
        model.run()

        # Compare
        predicted_C = model.data_df[f"C --> {flux_name}"].values
        error = benchmark_C - predicted_C
        rms_error = _rms(error)

        assert rms_error < self.TOLERANCE, (
            f"Steady-state benchmark '{name}' failed: RMS error = {rms_error:.6f} (tolerance = {self.TOLERANCE})"
        )


class TestUnsteadyBenchmark:
    """Test model accuracy against analytical unsteady uniform SAS solution."""

    TOLERANCE = 1e-2

    def test_unsteady_uniform(self):
        data_df = pd.read_csv("test/unsteady_data.csv")
        data_df = data_df[:500]
        data_df["Q"] = data_df["Q"] + data_df["ET"]
        data_df["ET"] = 0

        Storage_init = 1000.0
        C_old = 50.0
        dt = 1

        # Compute analytical solution
        benchmark_C = _analytical_unsteady_uniform(data_df, S_init=Storage_init, C_old=C_old, dt=dt)

        data_df["S0"] = Storage_init + (data_df["J"] - data_df["Q"] - data_df["ET"]).cumsum() * dt
        data_df.loc[data_df.index[1:], "S0"] = data_df["S0"].rolling(2).mean().iloc[1:]
        data_df["Smin"] = 0.0

        sas_spec = {
            "Q": {
                "Q SAS": {
                    "func": "kumaraswamy",
                    "args": {"a": 1.0, "b": 1.0, "scale": "S0", "loc": "Smin"},
                }
            }
        }
        solute_parameter = {"C in": {"C_old": C_old}}

        model = Model(
            data_df,
            sas_specs=sas_spec,
            solute_parameters=solute_parameter,
            dt=dt,
            influx="J",
            n_substeps=1,
            verbose=False,
        )
        model.run()

        predicted_C = model.data_df["C in --> Q"].values
        error = predicted_C - benchmark_C
        rms_error = _rms(error)

        assert rms_error < self.TOLERANCE, f"Unsteady uniform benchmark failed: RMS = {rms_error:.6f}"


def _analytical_unsteady_uniform(df, S_init=1000.0, C_old=50.0, dt=1):
    """Compute analytical solution for unsteady uniform SAS."""
    total_t = len(df)
    C_J = df["C in"].values.ravel()
    J = df["J"].values.ravel()
    Q = df["Q"].values.ravel()
    ET = df["ET"].values.ravel()
    S = S_init + (J - Q - ET).cumsum() * dt
    S = np.append(S_init, S)

    delta = dt * (Q + ET) / S[:-1]
    eta = S[1:] / S[:-1] - 1.0
    phi = np.where(np.abs(eta) < 1e-12, 1.0, np.log(eta + 1) / eta)

    C_Q = np.zeros(total_t)
    for t in range(total_t):
        pq = np.zeros(t + 2)
        pq[0] = (np.exp(-delta[t] * phi[t]) + delta[t] - 1) / delta[t]
        if np.isnan(pq[0]):
            pq[0] = 0.0
        C_Q[t] += C_J[t] * pq[0] * dt

        for T in range(1, t + 1):
            expo = np.exp(-np.sum(delta[t - T - 1 : t + 1] * phi[t - T - 1 : t + 1]))
            pq[T] = (
                S[t - T]
                / S[t]
                * expo
                * (np.exp(delta[t] * phi[t]) - 1)
                * (np.exp((delta[t - T] + eta[t - T]) * phi[t - T]) - 1.0)
                / delta[t]
            )
            C_Q[t] += C_J[t - T] * pq[T] * dt

        C_Q[t] += C_old * (1 - pq.sum() * dt)
    return C_Q
