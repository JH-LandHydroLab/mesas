"""Visualization utilities for SAS transport models.

Provides functions for plotting the "transport column" — a visualization
of how water of different ages is stored and released, coloured by
solute concentration.

.. note::
   ``plot_SAS_cumulative`` references ``model.sas_specs`` for the
   per-timestep SAS function lookup tables.
"""

from __future__ import annotations

from collections import OrderedDict

import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def _require_recorded_state(model, i):
    """Raise a helpful error unless the model recorded state at timestep i."""
    result = getattr(model, "result", None)
    if result is None or "sT" not in result:
        raise ValueError("Model has no results. Call model.run() first.")
    n_recorded = result["sT"].shape[1]
    if i + 1 >= n_recorded:
        raise ValueError(
            f"Timestep i={i} requires recorded state at steps i and i+1, but only "
            f"{n_recorded} state snapshots were recorded. Re-run the model with "
            "record_state=True to record every timestep."
        )


def plot_transport_column(
    model,
    flux,
    sol,
    i=0,
    ax=None,
    dST=None,
    nST=20,
    cmap="cividis_r",
    TC_frac=0.3,
    vrange=None,
    ST_max=None,
    omega_max=None,
    valvegap=0.3,
    hspace=0.015,
    artists_dict=None,
    do_init=True,
):
    if artists_dict is None:
        artists_dict = OrderedDict()
    if i is None:
        i = 0
    _require_recorded_state(model, i)
    dt = model.options["dt"]
    Q = model.data_df[flux].iloc[i]
    C_old = model.solute_parameters[sol]["C_old"]

    sTs = np.r_[0, model.result["sT"][:-1, i]]
    mTs = np.r_[0, model.result["mT"][:-1, i, list(model._solorder).index(sol)]]
    sTe = model.result["sT"][:, i + 1]
    mTe = model.result["mT"][:, i + 1, list(model._solorder).index(sol)]
    sT = (sTs + sTe) / 2
    mT = (mTs + mTe) / 2
    pQ = model.result["pQ"][:, i, list(model._fluxorder).index(flux)]
    mQ = model.result["mQ"][:, i, list(model._fluxorder).index(flux), list(model._solorder).index(sol)]
    ST = np.r_[0, np.cumsum(sT)] * dt
    PQ = np.r_[0, np.cumsum(pQ)] * dt
    _MT = np.r_[0, np.cumsum(mT)] * dt  # noqa: F841 (kept for potential future use)
    MQ = np.r_[0, np.cumsum(mQ)] * dt

    model.sas_specs[flux].make_spec_ts()
    ST_mod = np.r_[0.0, model.sas_specs[flux].ST[i, :]]
    PQ_mod = np.r_[0.0, model.sas_specs[flux].P[i, :]]
    # Zero-length segments and empty age bins legitimately produce 0/0 here;
    # the resulting nan/inf entries are masked below.
    with np.errstate(divide="ignore", invalid="ignore"):
        omega_mod = np.diff(PQ_mod) / np.diff(ST_mod)
        if omega_max is None:
            omega_max = np.nanmax(omega_mod) * 1.1
        if ST_max is None:
            ST_max = ST_mod[-1] * 1.1
        if np.isinf(ST_max):
            ST_max = ST.max() * 1.1

        CS = np.where(sT > 0, mT / sT, 0)

        if dST is None:
            ST_reg = np.linspace(0, ST_max, nST + 1)
        else:
            nST = int(ST_max / dST) + 1
            ST_reg = np.arange(nST + 1) * dST
        sT_reg = np.diff(ST_reg) / dt
        spacer = sT_reg[0] * dt * valvegap

        PQ_regmod = np.interp(ST_reg, ST_mod, PQ_mod, right=np.nan)
        omega_reg = np.diff(PQ_regmod) / dt / sT_reg

        PQ_reg = np.interp(ST_reg, ST, PQ)
        pQ_reg = np.diff(PQ_reg)
        MQ_reg = np.interp(ST_reg, ST, MQ)
        any_old_reg = np.interp(ST_reg, ST, (PQ == PQ[-1]))[1:]
        f_old_reg = any_old_reg * np.diff(PQ_regmod - PQ_reg) / np.diff(PQ_regmod)

        CQ_reg_new = np.where(pQ_reg > 0, np.diff(MQ_reg) / (Q * pQ_reg), 0)
        CQ_reg_new[np.isnan(CQ_reg_new)] = 0
        CQ_reg = CQ_reg_new * (1 - f_old_reg) + C_old * f_old_reg

    cmap = plt.get_cmap(cmap)
    if vrange is None:
        vmin, vmax = min(CS.min(), C_old), max(CS.max(), C_old)
    else:
        vmin, vmax = vrange
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    if do_init:
        if ax is None:
            ax = plt.subplot(111)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        for loc, spine in ax.spines.items():
            spine.set_visible(False)
        ax.spines["right"].set_visible(False)
        ax1 = ax.inset_axes([0, 0, TC_frac, 1])
        ax2 = ax.inset_axes([TC_frac + hspace, 0, 1 - TC_frac - hspace, 1])
        ax1.set_visible(True)
        ax2.set_visible(True)

        ax1.spines["bottom"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.spines["left"].set_position(("outward", 10))
        ax1.spines["right"].set_visible(False)
        ax1.xaxis.set_visible(False)

        ax2.spines["bottom"].set_position(("outward", 10))
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.yaxis.set_visible(False)

    if do_init:
        for j in range(len(sT)):
            artists_dict[f"TCpatch {j}"] = patches.Rectangle(
                (0, 0), 0, 0, linewidth=0, edgecolor="None", visible=False
            )  # , clip_on=False)
            ax1.add_patch(artists_dict[f"TCpatch {j}"])
        artists_dict["TCpatch C_old"] = patches.Rectangle(
            (0, 0), 0, 0, linewidth=0, edgecolor="0.5", visible=False, hatch=r"///"
        )  # , clip_on=False)
        ax1.add_patch(artists_dict["TCpatch C_old"])

    for j in range(len(sT)):
        if sT[j] > 0:
            artists_dict[f"TCpatch {j}"].set_bounds((0, ST[j], 1, sT[j] * dt))
            artists_dict[f"TCpatch {j}"].set_facecolor(cmap(norm(CS[j])))
            artists_dict[f"TCpatch {j}"].set_visible(True)
    artists_dict["TCpatch C_old"].set_bounds((0, ST[-1], 1, ST_max - ST[-1]))
    artists_dict["TCpatch C_old"].set_facecolor(cmap(norm(C_old)))
    artists_dict["TCpatch C_old"].set_visible(True)

    if do_init:
        for n in range(nST):
            artists_dict[f"TQpatch {n}"] = patches.Rectangle(
                (0, 0), 0, 0, linewidth=0, edgecolor="0.8", visible=False
            )  # , clip_on=False)
            ax2.add_patch(artists_dict[f"TQpatch {n}"])

    for n in range(nST):
        artists_dict[f"TQpatch {n}"].set_bounds((0, ST_reg[n], omega_reg[n], sT_reg[n] * dt - spacer))
        artists_dict[f"TQpatch {n}"].set_facecolor(cmap(norm(CQ_reg[n])))
        artists_dict[f"TQpatch {n}"].set_visible(True)

    if do_init:
        omega_linestylestr = "k--"
        (artists_dict["omegaline v start"],) = ax2.plot([0, 0], [0, 0], omega_linestylestr)  # , clip_on=False)
        (artists_dict["omegaline h start"],) = ax2.plot([0, 0], [0, 0], omega_linestylestr)  # , clip_on=False)
        for ip in range(len(omega_mod)):
            (artists_dict[f"omegaline v {ip}"],) = ax2.plot([0, 0], [0, 0], omega_linestylestr)  # , clip_on=False)
            if ip > 0:
                (artists_dict[f"omegaline h {ip}"],) = ax2.plot([0, 0], [0, 0], omega_linestylestr)  # , clip_on=False)
        (artists_dict["omegaline h end"],) = ax2.plot([0, 0], [0, 0], omega_linestylestr)  # , clip_on=False)
        (artists_dict["omegaline v end"],) = ax2.plot([0, 0], [0, 0], omega_linestylestr)  # , clip_on=False)

    artists_dict["omegaline v start"].set_data([0, 0], [0, ST_mod[0]])
    artists_dict["omegaline h start"].set_data([0, omega_mod[0]], [ST_mod[0], ST_mod[0]])
    for ip in range(len(omega_mod)):
        artists_dict[f"omegaline v {ip}"].set_data([omega_mod[ip], omega_mod[ip]], [ST_mod[ip], ST_mod[ip + 1]])
        if ip > 0:
            artists_dict[f"omegaline h {ip}"].set_data([omega_mod[ip - 1], omega_mod[ip]], [ST_mod[ip], ST_mod[ip]])
    artists_dict["omegaline h end"].set_data([omega_mod[-1], 0], [ST_mod[-1], ST_mod[-1]])
    artists_dict["omegaline v end"].set_data([0, 0], [ST_mod[-1], ST_max])

    if do_init:
        ax1.set_ylim([0, ST_max])
        ax2.set_ylim([0, ST_max])
        ax2.set_xlim([0, omega_max])
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        ax1.set_ylabel("$S_T$")
        ax2.set_xlabel(rf"$\omega(S_T)$ for {flux}")

        return ax1, ax2


def plot_influx(model, ax=None, sharex=None, i=0, artists_dict=None, do_init=True):
    if artists_dict is None:
        artists_dict = OrderedDict()
    J = model.data_df[model.options["influx"]]
    if do_init:
        if ax is None:
            ax = plt.subplot(111, sharex=sharex)
        ax.plot(model.data_df.index, J, "b")
        ax.set_ylim(bottom=0)
        (artists_dict["plot_influx timeline"],) = ax.plot(2 * [model.data_df.index[0]], [0, 0], "k", lw=0.5)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_title(model.options["influx"])
    if i is not None:
        timeline = artists_dict["plot_influx timeline"]
        timeline.set_data(2 * [model.data_df.index[i]], timeline.axes.get_ylim())


def plot_outflux(model, flux, ax=None, sharex=None, i=0, artists_dict=None, do_init=True):
    if artists_dict is None:
        artists_dict = OrderedDict()
    Q = model.data_df[flux]
    if do_init:
        if ax is None:
            ax = plt.subplot(111, sharex=sharex)
        ax.plot(model.data_df.index, Q, "b")
        ax.set_ylim(bottom=0)
        (artists_dict["plot_outflux timeline"],) = ax.plot(2 * [model.data_df.index[0]], [0, 0], "k", lw=0.5)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_title(flux)
    if i is not None:
        timeline = artists_dict["plot_outflux timeline"]
        timeline.set_data(2 * [model.data_df.index[i]], timeline.axes.get_ylim())


def plot_influx_conc(model, sol, ax=None, sharex=None, i=0, artists_dict=None, do_init=True):
    if artists_dict is None:
        artists_dict = OrderedDict()
    C_J = model.data_df[sol]
    if do_init:
        if ax is None:
            ax = plt.subplot(111, sharex=sharex)
        ax.plot(model.data_df.index, C_J, "b")
        (artists_dict["plot_influx_conc timeline"],) = ax.plot(2 * [model.data_df.index[0]], [0, 0], "k", lw=0.5)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_title(sol)
    if i is not None:
        timeline = artists_dict["plot_influx_conc timeline"]
        timeline.set_data(2 * [model.data_df.index[i]], timeline.axes.get_ylim())


def plot_outflux_conc(model, flux, sol, ax=None, sharex=None, i=0, artists_dict=None, do_init=True):
    if artists_dict is None:
        artists_dict = OrderedDict()
    colname = sol + " --> " + flux
    C_Q = model.data_df[colname]
    if do_init:
        if ax is None:
            ax = plt.subplot(111, sharex=sharex)
        ax.plot(model.data_df.index, C_Q, "b")
        (artists_dict["plot_outflux_conc timeline"],) = ax.plot(2 * [model.data_df.index[0]], [0, 0], "k", lw=0.5)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_title(colname)
    if i is not None:
        timeline = artists_dict["plot_outflux_conc timeline"]
        timeline.set_data(2 * [model.data_df.index[i]], timeline.axes.get_ylim())


def plot_SAS_cumulative(model, flux, ax=None, sharex=None, i=0, artists_dict=None, do_init=True):
    if artists_dict is None:
        artists_dict = OrderedDict()
    # The padded ST/P matrices only exist after make_spec_ts() has run
    if getattr(model.sas_specs[flux], "ST", None) is None:
        model.sas_specs[flux].make_spec_ts()
    if do_init:
        if ax is None:
            ax = plt.subplot(111, sharex=sharex)
        if i is None:
            i = 0
        (artists_dict[f"plot_SAS {flux}"],) = ax.plot(
            model.sas_specs[flux].ST[i, :], model.sas_specs[flux].P[i, :], "bo-", lw=1.5
        )  # , clip_on=False)
        ax.set_ylim([0, 1])
        ax.set_xlim(xmin=0)
        ax.plot(ax.get_xlim(), [1, 1], color="0.1", lw=0.8, ls=":")
        ax.plot(ax.get_xlim(), [0, 0], color="0.1", lw=0.8, ls=":")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("$S_T$")
        ax.set_ylabel(r"$\Omega(S_T)$")
        ax.spines["left"].set_position(("outward", 10))
        ax.spines["bottom"].set_position(("outward", 10))
    if i is not None:
        artists_dict[f"plot_SAS {flux}"].set_data(model.sas_specs[flux].ST[i, :], model.sas_specs[flux].P[i, :])


def plot_transport_column_with_timeseries(model, flux, sol, i=0, fig=None, artists_dict=None, do_init=True, **kwargs):
    if artists_dict is None:
        artists_dict = OrderedDict()

    if do_init:
        if fig is None:
            fig = plt.figure(figsize=[11.5, 4.0], layout="tight")

        axTC = plt.subplot2grid((2, 3), (0, 1), rowspan=2, fig=fig)
        axJ = plt.subplot2grid((2, 3), (0, 0), fig=fig)
        axQ = plt.subplot2grid((2, 3), (0, 2), fig=fig)
        axCJ = plt.subplot2grid((2, 3), (1, 0), fig=fig)
        axCQ = plt.subplot2grid((2, 3), (1, 2), fig=fig)

        plot_transport_column(model, flux, sol, i=i, ax=axTC, artists_dict=artists_dict, **kwargs)
        plot_influx(model, ax=axJ, i=i, artists_dict=artists_dict)
        plot_outflux(model, flux, ax=axQ, sharex=axJ, i=i, artists_dict=artists_dict)
        plot_influx_conc(model, sol, ax=axCJ, sharex=axJ, i=i, artists_dict=artists_dict)
        plot_outflux_conc(model, flux, sol, ax=axCQ, sharex=axJ, i=i, artists_dict=artists_dict)

        return axTC, axJ, axQ, axCJ, axCQ

    # Update-only path: refresh the artists created during init in place,
    # without creating new axes or patches (used by the animation).
    plot_transport_column(model, flux, sol, i=i, artists_dict=artists_dict, do_init=False, **kwargs)
    plot_influx(model, i=i, artists_dict=artists_dict, do_init=False)
    plot_outflux(model, flux, i=i, artists_dict=artists_dict, do_init=False)
    plot_influx_conc(model, sol, i=i, artists_dict=artists_dict, do_init=False)
    plot_outflux_conc(model, flux, sol, i=i, artists_dict=artists_dict, do_init=False)
    return None


def make_transport_column_animation(model, flux, sol, fig=None, frames=None, **kwargs):
    if frames is None:
        frames = range(model._timeseries_length - 1)

    if fig is None:
        fig = plt.figure(figsize=[11.5, 4.0], layout="tight")

    from matplotlib.animation import FuncAnimation

    artists = OrderedDict()

    def init():
        plot_transport_column_with_timeseries(model, flux, sol, i=0, fig=fig, artists_dict=artists, **kwargs)
        return list(artists.values())

    def update(frame):
        plot_transport_column_with_timeseries(
            model, flux, sol, i=frame, fig=fig, artists_dict=artists, do_init=False, **kwargs
        )
        return list(artists.values())

    return FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)
