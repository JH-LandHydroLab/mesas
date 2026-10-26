# %%
import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from scipy.stats import gamma, beta
from scipy.special import lambertw

from mesas.sas.model import Model

# %%
logging.basicConfig(filename="test.log", level=logging.INFO)

# %%

"""
"""


def M(delta, i):
    return np.where(
        i == 0,
        1,
        -lambertw(-np.exp(-1 - (i * delta) / 2.0)) * (2 + lambertw(-np.exp(-1 - (i * delta) / 2.0))),
    )


def RMS(x):
    return np.sqrt(np.mean(x**2))


def partial_piston_pQdisc(delta, i):
    n = 2 / delta - 0.5
    return np.where(i <= np.floor(n), delta / 2, np.where(i > np.ceil(n), 0, delta / 2 * (i - n)))


steady_benchmarks = {
    "Uniform": {
        "spec": {
            # "func": "kumaraswamy",
            # "args": {"a": 1.0-0.000000001, "b":1.0-0.000000001, "scale": "S_0", "loc": "S_m"},
            "ST": ["S_m", "S_m0"]
        },
        "pQdisc": lambda delta, i: (-1 + np.exp(delta)) ** 2 / (np.exp((1 + i) * delta) * delta),
        "pQdisc0": lambda delta: (1 + np.exp(delta) * (-1 + delta)) / (np.exp(delta) * delta),
        "subplot": 0,
        "distname": "Uniform",
    },
    "Exponential": {
        "spec": {
            "func": "gamma",
            "args": {"a": 1.0, "scale": "S_0", "loc": "S_m"},
        },
        "pQdisc": lambda delta, i: (
            (2 * np.log(1 + i * delta) - np.log((1 + (-1 + i) * delta) * (1 + delta + i * delta))) / delta
        ),
        "pQdisc0": lambda delta: (delta + np.log(1 / (1 + delta))) / delta,
        "subplot": 1,
        "distname": "Gamma(1.0)",
    },
    "Biased old (Beta)": {
        "spec": {
            "func": "beta",
            "args": {
                "a": 2.0 - 0.000000001,
                "b": 1.0 - 0.000000001,
                "scale": "S_0",
                "loc": "S_m",
            },
        },
        "pQdisc": lambda delta, i: (
            (
                2
                * 1
                / np.cosh(delta - i * delta)
                * 1
                / np.cosh(delta + i * delta)
                * np.sinh(delta) ** 2
                * np.tanh(i * delta)
            )
            / delta
        ),
        "pQdisc0": lambda delta: 1 - np.tanh(delta) / delta,
        "subplot": 2,
        "distname": "Beta(2,1)",
    },
    "Biased young (Beta)": {
        "spec": {
            "func": "beta",
            "args": {
                "a": 1.0 - 0.000000001,
                "b": 2.0 - 0.000000001,
                "scale": "S_0",
                "loc": "S_m",
            },
        },
        "pQdisc": lambda delta, i: (2 * delta) / ((1 + (-1 + i) * delta) * (1 + i * delta) * (1 + delta + i * delta)),
        "pQdisc0": lambda delta: delta / (1 + delta),
        "subplot": 3,
        "distname": "Beta(1,2)",
    },
    "Partial bypass (Beta)": {
        "spec": {
            "func": "beta",
            "args": {"a": 1.0 / 2, "b": 1.0, "scale": "S_0", "loc": "S_m"},
        },
        "pQdisc": lambda delta, i: (M(delta, -1 + i) - 2 * M(delta, i) + M(delta, 1 + i)) / delta,
        "pQdisc0": lambda delta: (-1 + delta + M(delta, 1)) / delta,
        "subplot": 4,
        "distname": "Beta(1/2,1)",
    },
    "Partial piston (Beta)": {
        "spec": {
            "func": "beta",
            "args": {"a": 1.0, "b": 1.0 / 2, "scale": "S_0", "loc": "S_m"},
        },
        "pQdisc": partial_piston_pQdisc,
        "pQdisc0": lambda delta: delta / 4,
        "subplot": 5,
        "distname": "Beta(1,1/2)",
    },
    "Biased old (Kumaraswamy)": {
        "spec": {
            "func": "kumaraswamy",
            "args": {
                "a": 2.0 - 0.000000001,
                "b": 1.0 - 0.000000001,
                "scale": "S_0",
                "loc": "S_m",
            },
        },
        "pQdisc": lambda delta, i: (
            (
                2
                * 1
                / np.cosh(delta - i * delta)
                * 1
                / np.cosh(delta + i * delta)
                * np.sinh(delta) ** 2
                * np.tanh(i * delta)
            )
            / delta
        ),
        "pQdisc0": lambda delta: 1 - np.tanh(delta) / delta,
        "subplot": 2,
        "distname": "Kumaraswamy(2,1)",
    },
    "Biased young (Kumaraswamy)": {
        "spec": {
            "func": "kumaraswamy",
            "args": {
                "a": 1.0 - 0.000000001,
                "b": 2.0 - 0.000000001,
                "scale": "S_0",
                "loc": "S_m",
            },
        },
        "pQdisc": lambda delta, i: (2 * delta) / ((1 + (-1 + i) * delta) * (1 + i * delta) * (1 + delta + i * delta)),
        "pQdisc0": lambda delta: delta / (1 + delta),
        "subplot": 3,
        "distname": "Kumaraswamy(1,2)",
    },
    "Partial bypass (Kumaraswamy)": {
        "spec": {
            "func": "kumaraswamy",
            "args": {"a": 1.0 / 2, "b": 1.0, "scale": "S_0", "loc": "S_m"},
        },
        "pQdisc": lambda delta, i: (M(delta, -1 + i) - 2 * M(delta, i) + M(delta, 1 + i)) / delta,
        "pQdisc0": lambda delta: (-1 + delta + M(delta, 1)) / delta,
        "subplot": 4,
        "distname": "Kumaraswamy(1/2,1)",
    },
    "Partial piston (Kumaraswamy)": {
        "spec": {
            "func": "kumaraswamy",
            "args": {"a": 1.0, "b": 1.0 / 2, "scale": "S_0", "loc": "S_m"},
        },
        "pQdisc": partial_piston_pQdisc,
        "pQdisc0": lambda delta: delta / 4,
        "subplot": 5,
        "distname": "Kumaraswamy(1,1/2)",
    },
}
# %%
letter = "abcdefghijklmnopqrstuvwxyz"


def test_steady(makefigure=False):
    # %%
    np.random.seed(2)
    timeseries_length = 250
    max_age = timeseries_length
    dt = 0.1
    Q_0 = 1.0  # <-- steady-state flow rate
    C_J = 0.0 * np.ones(timeseries_length) + np.random.randn(timeseries_length) * 1.0

    C_old = 1.0
    J = Q_0
    S_0 = 5 * Q_0
    S_m = 0 * Q_0
    n_substeps = 1
    debug = False
    verbose = True
    jacobian = False
    num_scheme = 4

    data_df = pd.DataFrame(index=range(timeseries_length))
    data_df["t"] = data_df.index * dt
    data_df["J"] = J
    data_df["S_0"] = S_0
    data_df["S_m"] = S_m
    data_df["S_m0"] = S_m + S_0
    data_df["C"] = C_J

    if makefigure:
        fig = plt.figure(figsize=[21, 12])
        nrow = 4
        ncol = 6  # int(len(steady_benchmarks)/nrow)+1
        figcount = 0
        ax0_dict = {}
        ax1_dict = {}
        ax2_dict = {}
        ax3_dict = {}
    for name, bm in steady_benchmarks.items():
        print(name)
        print(bm)
        i = np.arange(timeseries_length)
        pQdisc = np.zeros_like(i, dtype=float)
        delta = dt * Q_0 / (S_0)
        Tm = S_m / Q_0
        pQdisc[0] = bm["pQdisc0"](delta) / dt
        pQdisc[1:] = bm["pQdisc"](delta, i[1:]) / dt
        im = int(Tm / dt)
        data_df[f"{name} benchmark C"] = C_old
        data_df.loc[data_df.index[im:], f"{name} benchmark C"] = np.convolve(C_J, pQdisc, mode="full")[
            : timeseries_length - im
        ] * dt + C_old * (1 - np.cumsum(pQdisc)[: timeseries_length - im] * dt)
        data_df[f"{name} benchmark t"] = data_df["t"]

        data_df[name] = Q_0
        solute_parameters = {
            "C": {
                "C_old": C_old,
                "observations": {name: f"{name} benchmark C"},
            }
        }
        sas_specs = {name: {f"{name}_SAS": bm["spec"]}}

        model = Model(
            data_df,
            sas_specs=sas_specs,
            solute_parameters=solute_parameters,
            debug=debug,
            verbose=verbose,
            dt=dt,
            n_substeps=n_substeps,
            jacobian=jacobian,
            max_age=max_age,
            warning=True,
            num_scheme=num_scheme,
        )
        tic = time.perf_counter()
        model.run()
        toc = time.perf_counter()
        data_df = model.data_df
        err01 = -(
            data_df[f"{name} benchmark C"].values - data_df[f"C --> {name}"].values
        )  # /data_df[f'{name} benchmark C'].values
        logging.info(f"{name} error01 = {RMS(err01)}")
        logging.info(f"{name} time 01 = {toc - tic}")
        assert RMS(err01) < 1e-2

        model = Model(
            data_df,
            sas_specs=sas_specs,
            solute_parameters=solute_parameters,
            debug=debug,
            verbose=verbose,
            dt=dt,
            n_substeps=n_substeps * 10,
            jacobian=jacobian,
            max_age=max_age,
            warning=True,
            num_scheme=num_scheme,
        )
        tic = time.perf_counter()
        model.run()
        toc = time.perf_counter()
        data_df = model.data_df
        err10 = -(
            data_df[f"{name} benchmark C"].values - data_df[f"C --> {name}"].values
        )  # /data_df[f'{name} benchmark C'].values
        logging.info(f"{name} error10 = {RMS(err10)}")
        logging.info(f"{name} time 10 = {toc - tic}")
        # assert RMS(err10)<1E-3
        if makefigure:
            # icol = int(figcount/nrow)
            # irow = figcount - nrow * icol
            if bm["subplot"] not in ax0_dict.keys():
                ax0 = plt.subplot2grid((nrow, ncol), (0, bm["subplot"]))
                ax0.set_title(name.split("(")[0].strip())
                ax0.set_ylim((-0.02, 0.6))
                # ax0.set_ytick([1000, 2000])
                ax0.set_xlabel(r"$S_T$")
                ax0.spines.top.set_visible(False)
                # ax0.spines.bottom.set_visible(False)
                ax0.spines.right.set_visible(False)
                ax0.spines.bottom.set_position(("outward", 5))
                if bm["subplot"] > 0:
                    ax0.get_yaxis().set_visible(False)
                    ax0.spines.left.set_visible(False)
                else:
                    ax0.spines.left.set_position(("outward", 10))
                    ax0.set_ylabel(r"SAS function $\omega(S_T)$")
                ax0_dict[bm["subplot"]] = ax0
                ax0.annotate(
                    letter[0 * ncol + bm["subplot"]],
                    (-0.12, 1.1),
                    xycoords="axes fraction",
                    fontsize="x-large",
                    fontweight="bold",
                )
            else:
                ax0 = ax0_dict[bm["subplot"]]
            print(name)
            ST = np.r_[
                [0, S_m - 0.0001, S_m],
                np.linspace(S_m, (S_0 + S_m), 1000),
                [S_0 + S_m],
                S_0 + S_m + 0.0001 + np.linspace(0, 1, 10),
            ]
            if name == "Uniform":
                SASpdf = np.zeros_like(ST)
                SASpdf[3:1003] = 1 / S_0
            else:
                SASpdf = model.sas_specs[name].components[f"{name}_SAS"].sas_fun[0].func.pdf(ST)
            SASpdf[2] = np.nan
            SASpdf[-11] = np.nan
            ax0.plot(ST, SASpdf, alpha=0.6, lw=2, ls="-", label=bm["distname"])
            ax0.legend(frameon=False)
            if bm["subplot"] not in ax1_dict.keys():
                ax1 = plt.subplot2grid((nrow, ncol), (1, bm["subplot"]))
                # ax1.set_title(name.split('(')[0].strip())
                # ax1.set_ylim((900, 2100))
                ax1.set_ylim((-0.25, 1.05))
                # ax1.set_ytick([1000, 2000])
                ax1.set_xlabel("Time")
                ax1.plot(
                    data_df[f"{name} benchmark t"],
                    data_df[f"{name} benchmark C"],
                    "k",
                    alpha=0.5,
                    ls="--",
                    lw=1.0,
                    zorder=100,
                    label="Benchmark",
                )
                ax1.legend(frameon=False)
                ax1.spines.top.set_visible(False)
                # ax1.spines.bottom.set_visible(False)
                ax1.spines.right.set_visible(False)
                ax1.spines.bottom.set_position(("outward", 10))
                if bm["subplot"] > 0:
                    ax1.get_yaxis().set_visible(False)
                    ax1.spines.left.set_visible(False)
                else:
                    ax1.spines.left.set_position(("outward", 10))
                    ax1.set_ylabel("Tracer conc.")
                ax1_dict[bm["subplot"]] = ax1
                ax1.annotate(
                    letter[1 * ncol + bm["subplot"]],
                    (-0.12, 1.1),
                    xycoords="axes fraction",
                    fontsize="x-large",
                    fontweight="bold",
                )
            else:
                ax1 = ax1_dict[bm["subplot"]]
            ax1.plot(data_df["t"], data_df[f"C --> {name}"], alpha=0.6, lw=2)
            ax1.legend(frameon=False)
            if bm["subplot"] not in ax2_dict.keys():
                ax2 = plt.subplot2grid((nrow, ncol), (2, bm["subplot"]))
                # ax2.set_ylim((500, 2500))
                # ax2.set_ytick([1000, 2000])
                ax2.set_xlabel("Time")
                ax2.spines.top.set_visible(False)
                # ax2.spines.bottom.set_visible(False)
                ax2.spines.right.set_visible(False)
                ax2.spines.bottom.set_position(("outward", 10))
                if bm["subplot"] > 0:
                    pass
                #    ax2.get_yaxis().set_visible(False)
                #    ax2.spines.left.set_visible(False)
                else:
                    ax2.set_ylabel("Error (1 substep)")
                #    ax2.spines.left.set_position(('outward', 10))
                ax2_dict[bm["subplot"]] = ax2
                ax2.annotate(
                    letter[2 * ncol + bm["subplot"]],
                    (-0.12, 1.1),
                    xycoords="axes fraction",
                    fontsize="x-large",
                    fontweight="bold",
                )
            else:
                ax2 = ax2_dict[bm["subplot"]]
            ax2.plot(data_df["t"], err01, alpha=0.6, lw=2, label=f" RMSE = {RMS(err01):.2e}")
            ax2.legend(frameon=False, loc="upper left")
            if bm["subplot"] not in ax3_dict.keys():
                ax3 = plt.subplot2grid((nrow, ncol), (3, bm["subplot"]))
                # ax3.set_ylim((500, 2500))
                # ax3.set_ytick([1000, 2000])
                ax3.set_xlabel("Time")
                ax3.spines.top.set_visible(False)
                # ax3.spines.bottom.set_visible(False)
                ax3.spines.right.set_visible(False)
                ax3.spines.bottom.set_position(("outward", 10))
                if bm["subplot"] > 0:
                    pass
                #    ax3.get_yaxis().set_visible(False)
                #    ax3.spines.left.set_visible(False)
                else:
                    ax3.set_ylabel("Error (10 substeps)")
                #    ax3.spines.left.set_position(('outward', 10))
                ax3_dict[bm["subplot"]] = ax3
                ax3.annotate(
                    letter[3 * ncol + bm["subplot"]],
                    (-0.12, 1.1),
                    xycoords="axes fraction",
                    fontsize="x-large",
                    fontweight="bold",
                )
            else:
                ax3 = ax3_dict[bm["subplot"]]
            ax3.plot(data_df["t"], err10, alpha=0.6, lw=2, label=f" RMSE = {RMS(err10):.2e}")
            ax3.legend(frameon=False, loc="upper left")
            figcount += 1
    if makefigure:
        fig.tight_layout()
        fig.savefig("test_steady.pdf")


# %%


# %%
# get analytical solution
def analytical_set(df, S_init=1000.0, C_old=50.0, dt=1):
    # get the time varying analytical solution
    total_t = len(df)
    C_J = df["C in"].values.ravel()
    J = df["J"].values.ravel()
    Q = df["Q"].values.ravel()
    ET = df["ET"].values.ravel()
    S = S_init + (J - Q - ET).cumsum() * dt  # storage at the end of each timestep
    S = np.append(S_init, S)  # storage at the start of each timestep

    # j - timestep
    # i - age step
    def analytical(total_t, dt, ET, Q, S, J, C_J, C_old):
        C_Q = np.zeros(total_t)
        # delta_j
        delta = dt * (Q + ET) / S[:-1]
        eta = S[1:] / S[:-1] - 1.0
        # phi
        phi = np.log(eta + 1) / eta
        phi[np.isnan(phi)] = 1.0  # when eta=0

        # at given t and T, exponential part when i > 0
        def expo(delta, phi):
            return np.exp(-np.sum(delta * phi))  # sum over age T at time t

        # for all timesteps
        for t in range(total_t):
            # now the maximum age in the system is t
            pq = np.zeros(t + 2)  # store cq that goes away
            # when T = 0
            pq[0] = (np.exp(-delta[t] * phi[t]) + delta[t] - 1) / delta[t]  # p cannot be smaller than 0
            if np.isnan(pq[0]):
                pq[0] = 0.0
            # get C_Q at T == 0
            # C_Q[t+1] += C_J[t]*pq[0]*dt
            C_Q[t] += C_J[t] * pq[0] * dt
            # when T > 0
            for T in range(1, t + 1):
                # calculate SAS
                pq[T] = (
                    S[t - T]
                    / S[t]
                    * expo(delta[t - T - 1 : t + 1], phi[t - T - 1 : t + 1])
                    * (np.exp(delta[t] * phi[t]) - 1)
                    * (np.exp((delta[t - T] + eta[t - T]) * phi[t - T]) - 1.0)
                    / delta[t]
                )
                # get C_Q
                C_Q[t] += C_J[t - T] * pq[T] * dt
            C_Q[t] += C_old * (1 - pq.sum() * dt)
        return C_Q

    C_Q = analytical(total_t, dt, ET, Q, S, J, C_J, C_old)
    return C_Q


def test_unsteady_uniform(makefigure=False, tmax=500):
    # %%
    data_df = pd.read_csv("./test/unsteady_data.csv")
    data_df = data_df[:tmax]
    data_df["Q"] = data_df["Q"] + data_df["ET"]
    data_df["ET"] = 0

    Storage_init = 1000.0
    C_old = 50.0
    dt = 1
    data_df["unsteady uniform benchark C"] = analytical_set(data_df, S_init=Storage_init, C_old=C_old, dt=dt)

    data_df["S0"] = (
        Storage_init + (data_df["J"] - data_df["Q"] - data_df["ET"]).cumsum() * dt
    )  # This should be the total storage volume, solve the mass balance equation
    # average stoage over the timestep
    data_df.loc[data_df.index[1:], "S0"] = data_df["S0"].rolling(2).mean()[1:]
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
    option = {"dt": dt, "influx": "J", "n_substeps": 1, "verbose": True}
    model = Model(data_df, sas_specs=sas_spec, solute_parameters=solute_parameter, **option)
    model.run()
    data_df = model.data_df
    # Throw out the spinup
    # data_df = data_df[-2923:]

    err = data_df["C in --> Q"].values - data_df["unsteady uniform benchark C"].values
    RMSE = np.sqrt(np.mean(err**2))
    logging.info(f"Unsteady uniform error = {RMSE}")
    assert RMSE < 1e-2

    if makefigure:
        fig = plt.figure()
        ax = plt.subplot()
        ax.plot(data_df["C in --> Q"], alpha=0.3, lw=2, label="mesas")
        ax.plot(data_df["unsteady uniform benchark C"], alpha=0.3, lw=2, label="benchmark")
        ax.plot(data_df["C_Q TranSAS"], "r--", alpha=0.3, lw=2, label="transas")
        plt.legend()
        ax.set_ylabel("Tracer conc.")
        ax.set_xlabel("Time")
        fig.tight_layout()
        fig.savefig("test_unsteady_uniform.pdf")
        fig.show()


# %%


def test_part_multiple(makefigure=False):
    # %%
    timeseries_length = 300
    max_age = timeseries_length
    dt = 0.1
    Q_0 = 1.0  # <-- steady-state flow rate
    C_J = 1000 * np.ones(timeseries_length)
    C_eq = 00.0
    C_old = 000.0
    J = Q_0
    f1, f2, f3 = 0.2, 0.2, 0.6
    S_0 = 10
    n_substeps = 1
    debug = False
    verbose = False
    jacobian = False

    data_df = pd.DataFrame(index=range(timeseries_length))
    data_df["t"] = data_df.index * dt
    data_df["J"] = J
    data_df["Q1"] = J * f1
    data_df["Q2"] = J * f2
    data_df["Q3"] = J * f3
    data_df["S_0"] = S_0
    data_df["C"] = C_J

    if makefigure:
        fig = plt.figure()
        name = "Uniform"
        bm = steady_benchmarks[name]
        j = np.arange(timeseries_length)
        pQdisc = np.zeros_like(j, dtype=float)
        delta = dt * Q_0 / (S_0)
        pQdisc[0] = bm["pQdisc0"](delta) / dt
        pQdisc[1:] = bm["pQdisc"](delta, j[1:]) / dt
        data_df[f"{name} benchmark C"] = np.convolve(C_J, pQdisc, mode="full")[:timeseries_length] * dt + C_old * (
            1 - np.cumsum(pQdisc)[:timeseries_length] * dt
        )
        data_df[f"{name} benchmark t"] = data_df["t"]

        solute_parameters = {
            "C": {"C_old": C_old, "alpha": {"Q1": 0.5, "Q2": 1.5, "Q3": 1.0}},
        }

        sas_specs = {
            "Q1": {"Q1_SAS": {"ST": [0, S_0 / 3, S_0 * 2 / 3, S_0]}},
            "Q2": {"Q2_SAS": {"ST": [0, S_0 / 3, S_0 * 2 / 3, S_0]}},
            "Q3": {
                "Q3_SAS": {
                    "ST": [0, S_0 / 2, S_0 * 3 / 4, S_0],
                    "P": [0, 1 / 2.0, 3 / 4.0, 1],
                }
            },
        }

        model = Model(
            data_df,
            sas_specs=sas_specs,
            solute_parameters=solute_parameters,
            debug=debug,
            verbose=verbose,
            dt=dt,
            n_substeps=n_substeps,
            jacobian=jacobian,
            max_age=max_age,
            warning=True,
        )
        model.run()
        data_df = model.data_df
        data_df[f"C --> {name}"] = f1 * data_df["C --> Q1"] + f2 * data_df["C --> Q2"] + f3 * data_df["C --> Q3"]
        err = (data_df[f"{name} benchmark C"].values - data_df[f"C --> {name}"].values) / data_df[
            f"{name} benchmark C"
        ].values
        logging.info(f"Part/multiple error = {err.mean()}")
        assert err.mean() < 1e-2
        if makefigure:
            ax = plt.subplot()
            ax.plot(data_df["t"], data_df[f"C --> {name}"], alpha=0.3, lw=2)
            ax.plot(data_df["t"], data_df["C --> Q1"], "c", alpha=0.3, lw=2)
            ax.plot(data_df["t"], data_df["C --> Q2"], "m", alpha=0.3, lw=2)
            ax.plot(data_df["t"], data_df["C --> Q3"], "k", alpha=0.3, lw=2)
            ax.plot(
                data_df[f"{name} benchmark t"],
                data_df[f"{name} benchmark C"],
                "k--",
                alpha=0.3,
                lw=2,
            )
            ax.set_title(name)
            ax.legend()
            # ax.set_ylim((500, 2500))
            ax.set_ylabel("Tracer conc.")
            ax.set_xlabel("Time")
    if makefigure:
        fig.tight_layout()
        fig.savefig("test_part_multiple.pdf")
        fig.show()


# %%


def test_reaction(makefigure=False):
    # %%
    u = 1  # No effect
    v = 1  # No effect
    timeseries_length = 500
    dt = 1 * u
    Q_0 = 1.0 / u * v  # <-- steady-state flow rate
    C_J = 0.0 * np.ones(timeseries_length)
    C_eq = 1000.0
    C_old = 1000.0
    J = Q_0
    S_0 = 100.0 * v
    k1 = 1 / 200.0 / u
    max_age = timeseries_length
    n_substeps = 1
    debug = False
    verbose = False
    jacobian = False

    data_df = pd.DataFrame(index=range(timeseries_length))
    data_df["t"] = data_df.index * dt
    data_df["J"] = J
    data_df["Q1"] = J
    data_df["S_0"] = S_0
    data_df["R"] = C_J

    if makefigure:
        fig = plt.figure()
        bm = steady_benchmarks["Uniform"]
        j = np.arange(timeseries_length)
        pQdisc = np.zeros_like(j, dtype=float)
        delta = dt * Q_0 / (S_0)
        pQdisc[0] = bm["pQdisc0"](delta) / dt
        pQdisc[1:] = bm["pQdisc"](delta, j[1:]) / dt
        kappa = dt * k1
        pkdisc = np.zeros_like(j, dtype=float)
        pkdisc[0] = bm["pQdisc0"](kappa) / dt
        pkdisc[1:] = bm["pQdisc"](kappa, j[1:]) / dt
        data_df["Reaction benchmark R"] = np.convolve((C_J / k1 + C_eq / (Q_0 / S_0)), pQdisc * pkdisc, mode="full")[
            :timeseries_length
        ] * dt + C_old * (1 - np.cumsum(pkdisc)[:timeseries_length] * dt) * (
            1 - np.cumsum(pQdisc)[:timeseries_length] * dt
        )
        data_df["Reaction benchmark t"] = data_df["t"]

        solute_parameters = {"R": {"C_old": C_old, "k1": k1, "C_eq": C_eq}}

        sas_specs = {"Q1": {"Q1_SAS": {"ST": [0, S_0 / 3, S_0 * 2 / 3, S_0]}}}

        model = Model(
            data_df,
            sas_specs=sas_specs,
            solute_parameters=solute_parameters,
            debug=debug,
            verbose=verbose,
            dt=dt,
            n_substeps=n_substeps,
            jacobian=jacobian,
            max_age=max_age,
            warning=True,
        )
        model.run()
        data_df = model.data_df
        err = (data_df["Reaction benchmark R"].values - data_df["R --> Q1"].values) / data_df[
            "Reaction benchmark R"
        ].values
        logging.info(f"Reaction error = {err.mean()}")
        assert err.mean() < 1e-2
        if makefigure:
            ax = plt.subplot()
            ax.plot(
                data_df["t"],
                data_df["R --> Q1"],
                "r",
                alpha=0.3,
                lw=2,
                label="mesas.py",
            )
            ax.plot(
                data_df["Reaction benchmark t"],
                data_df["Reaction benchmark R"],
                "r--",
                alpha=0.3,
                lw=2,
                label="benchmark",
            )
            ax.set_title("First order reaction")
            ax.legend()
            ax.set_ylim((0, 1100))
            ax.set_ylabel("Tracer conc.")
            ax.set_xlabel("Time")
            ax.set_title(data_df["R --> Q1"].values[-1])
    if makefigure:
        fig.tight_layout()
        fig.savefig("test_reaction.pdf")
        fig.show()


# %%

if __name__ == "__main__":
    test_steady(makefigure=True)
    # test_unsteady_uniform(makefigure=True)
    # test_part_multiple(makefigure=True)
    # test_reaction(makefigure=True)
