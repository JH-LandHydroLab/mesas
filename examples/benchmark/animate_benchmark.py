from collections import OrderedDict

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import norm as gaussian

from mesas.sas.model import Model


def plot_transport_column(
    model,
    flux,
    sol,
    i=None,
    axTC=None,
    axTQ=None,
    dST=None,
    nST=20,
    cmap="inferno",
    vrange=None,
    ST_max=None,
    omega_max=None,
    artists_dict=OrderedDict(),
    do_init=True,
    SASfun=None,
    bounded_ST=True,
    **kwargs,
):
    if i is None:
        i = 0

    dt = model.options["dt"]
    Q = model.data_df[flux].iloc[i]
    S = model.data_df["S_0"].iloc[i]
    if bounded_ST:
        ST_bound = S
    else:
        ST_bound = ST_max
    C_old = model.solute_parameters[sol]["C_old"]
    C_J = model.data_df[sol]

    sTs = np.r_[0, model.result["sT"][:-1, i]]
    mTs = np.r_[0, model.result["mT"][:-1, i, list(model._solorder).index(sol)]]
    sTe = model.result["sT"][:, i + 1]
    mTe = model.result["mT"][:, i + 1, list(model._solorder).index(sol)]
    sT = (sTs + sTe) / 2
    mT = (mTs + mTe) / 2
    pQ = model.result["pQ"][:, i, list(model._fluxorder).index(flux)]
    ST = np.r_[0, np.cumsum(sT)] * dt
    CS = np.where(sT > 0, mT / sT, 0)

    omega = pQ / sT

    cmap = plt.get_cmap(cmap)
    if vrange is None:
        vmin, vmax = min(C_J.min(), C_old), max(C_J.max(), C_old)
    else:
        vmin, vmax = vrange
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    if do_init:
        axTC.set_visible(True)
        axTQ.set_visible(True)
        axTC.yaxis.tick_right()
        axTC.yaxis.set_label_position("right")

        axTC.spines["bottom"].set_visible(False)
        axTC.spines["top"].set_visible(False)
        axTC.spines["right"].set_position(("outward", 10))
        axTC.spines["left"].set_visible(False)
        axTC.xaxis.set_visible(False)

        axTQ.spines["bottom"].set_position(("outward", 10))
        axTQ.spines["top"].set_visible(False)
        axTQ.spines["right"].set_visible(False)
        axTQ.spines["left"].set_visible(False)
        axTQ.yaxis.set_visible(False)

    ST_f = np.linspace(0.00000001, ST_max, 1000)
    omega_f = bm["SASfun"](ST_f, model.data_df.iloc[i, :])

    if do_init:
        omega_linestylestr = "-"
        (artists_dict["omegaline"],) = axTQ.plot(
            [0, 0], [0, 0], omega_linestylestr, color="blue", lw=2, clip_on=True, zorder=2
        )
        artists_dict["omegamask"] = Polygon([[0, 0], [1, 1]], closed=True, facecolor="none", edgecolor="none")
        axTQ.add_patch(artists_dict["omegamask"])
    artists_dict["omegaline"].set_data(omega_f, ST_f)
    artists_dict["omegamask"].set_xy(
        np.concatenate(
            (np.array([[0, 0]]).T, np.array([np.minimum(omega_max, omega_f), ST_f]), np.array([[0, ST_f[-1]]]).T),
            axis=1,
        ).T
    )

    if do_init:
        for j in range(len(sT)):
            artists_dict[f"TCpatch {j}"] = patches.Rectangle(
                (0, 0), 0, 0, linewidth=0, edgecolor="None", visible=False
            )  # , clip_on=False)
            axTC.add_patch(artists_dict[f"TCpatch {j}"])
        artists_dict["TCpatch C_old"] = patches.Rectangle(
            (0, 0), 0, 0, linewidth=0, edgecolor="0.5", visible=False, hatch=r"///"
        )  # , clip_on=False)
        axTC.add_patch(artists_dict["TCpatch C_old"])

    for j in range(len(sT)):
        if sT[j] > 0:
            artists_dict[f"TCpatch {j}"].set_bounds((0, ST[j], 1, sT[j] * dt))
            artists_dict[f"TCpatch {j}"].set_facecolor(cmap(norm(CS[j])))
            artists_dict[f"TCpatch {j}"].set_visible(True)
        else:
            if f"TCpatch {j}" in artists_dict:
                artists_dict[f"TCpatch {j}"].set_visible(False)
    artists_dict["TCpatch C_old"].set_bounds((0, ST[-1], 1, ST_bound - ST[-1]))
    artists_dict["TCpatch C_old"].set_facecolor(cmap(norm(C_old)))
    artists_dict["TCpatch C_old"].set_visible(True)

    if do_init:
        for j in range(len(sT)):
            artists_dict[f"TQpatch {j}"] = patches.Rectangle(
                (0, 0), 0, 0, linewidth=0, edgecolor="0.8", visible=False, clip_path=artists_dict["omegamask"]
            )  # , clip_on=False)
            axTQ.add_patch(artists_dict[f"TQpatch {j}"])
        artists_dict["TQpatch C_old"] = patches.Rectangle(
            (0, 0), 0, 0, linewidth=0, edgecolor="0.5", visible=False, hatch=r"///", clip_path=artists_dict["omegamask"]
        )  # , clip_on=False)
        axTQ.add_patch(artists_dict["TQpatch C_old"])

    for j in range(len(sT)):
        if sT[j] > 0:
            # artists_dict[f'TQpatch {j}'].set_bounds((0, ST[j], omega[j], sT[j] * dt))
            artists_dict[f"TQpatch {j}"].set_bounds((0, ST[j], omega_max, sT[j] * dt))
            artists_dict[f"TQpatch {j}"].set_facecolor(cmap(norm(CS[j])))
            artists_dict[f"TQpatch {j}"].set_visible(True)
        else:
            if f"TQpatch {j}" in artists_dict:
                artists_dict[f"TQpatch {j}"].set_visible(False)
    artists_dict["TQpatch C_old"].set_bounds((0, ST[-1], omega_max, ST_max - ST[-1]))
    artists_dict["TQpatch C_old"].set_facecolor(cmap(norm(C_old)))
    artists_dict["TQpatch C_old"].set_visible(True)

    if do_init:
        axTC.set_ylim([0, ST_max])
        axTQ.set_ylim([0, ST_max])
        axTQ.set_xlim([0, omega_max])
        axTC.invert_yaxis()
        axTQ.invert_yaxis()
        axTQ.invert_xaxis()
        axTC.set_ylabel("$S_T$ [mm]")
        axTQ.set_xlabel(r"SAS function $\omega(S_T,t)$ [mm$^-$$^1$]")
        axTQ.xaxis.label.set_color("blue")
        axTQ.tick_params(axis="x", colors="blue")
        axTQ.spines["bottom"].set_color("blue")
        return axTC, axTQ
    else:
        # Uncomment to anchor the storage column to the total storage, rather than to ST=0
        # axTC.set_ylim([S+0.05*ST_max, S-ST_max-0.05*ST_max])
        # axTQ.set_ylim([S+0.05*ST_max, S-ST_max-0.05*ST_max])
        # axTC.set_yticks([tick for tick in axTC.get_yticks() if tick>=0])
        # axTC.set_ylim([S+0.05*ST_max, S-ST_max-0.05*ST_max])
        # axTQ.set_ylim([S+0.05*ST_max, S-ST_max-0.05*ST_max])
        pass


def plot_influx(model, ax=None, sharex=None, i=None, artists_dict=OrderedDict(), do_init=True):
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
        artists_dict["plot_influx timeline"].set_data(2 * [model.data_df.index[i]], ax.get_ylim())


def plot_influx_and_conc(
    model,
    sol,
    ax=None,
    sharex=None,
    i=None,
    artists_dict=OrderedDict(),
    do_init=True,
    cmap="inferno",
    TCspan=None,
    t_span=None,
    **kwargs,
):
    J = model.data_df[model.options["influx"]]
    C_J = model.data_df[sol]
    t = model.data_df.index
    windowleft = t_span
    windowright = t_span * TCspan / (1 - TCspan)
    cmap = cm.get_cmap(cmap)
    norm = colors.Normalize(vmin=C_J.min(), vmax=C_J.max())
    color_fun = lambda C: cmap(norm(C))
    if do_init:
        if ax is None:
            ax = plt.subplot(111, sharex=sharex)
        ax.bar(t, J, color=color_fun(C_J), width=1)
        ax.set_ylim(bottom=0, top=J.max() * 1.1)
        ax.set_xlim(left=t[0] - windowleft, right=t[0] + windowright)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # jax.set_title(model.options['influx'])
        # ax.set_title("Inflow (e.g. precipitation)", loc="left")
        ax.yaxis.tick_right()
        bbox = ax.get_position()
        # dimR = Rectangle((bbox.x0, bbox.y0),
        #    bbox.width*(1-TCspan), bbox.height,
        #    transform=ax.get_figure().transFigure,
        #    color='w',alpha=0.7)
        # ax.add_patch(dimR)
        artists_dict["plot_influx"] = ax
        # plt.annotate(r"time t", xy=[bbox.x0+bbox.width*(1-TCspan)+0.015, bbox.y0+bbox.height*1.1],
        #        xycoords=ax.get_figure().transFigure, color='0.5', fontweight='bold')
        # plt.annotate("", xy=[bbox.x0+bbox.width*(1-TCspan), 0.11],
        #        xytext=[bbox.x0+bbox.width*(1-TCspan), bbox.y0+bbox.height*1.1],
        #        xycoords=ax.get_figure().transFigure,
        #        arrowprops=dict(arrowstyle='-', linewidth=2, color='0.5', ls='dashed'))
        artists_dict["inflow_arrow_1"] = plt.annotate(
            "",
            xy=[bbox.x0 + bbox.width * (1 - TCspan) + 0.235, bbox.y0 - bbox.height * 0.9 - 0.01],
            xytext=[bbox.x0 + bbox.width * (1 - TCspan) + 0.235, bbox.y0 - bbox.height * 0.9 - 0.01 + 0.01],
            xycoords=ax.get_figure().transFigure,
            arrowprops=dict(arrowstyle="simple,tail_width=0.7,head_width=2.5,head_length=1", linewidth=2, color="0.5"),
        )
        artists_dict["inflow_arrow_2"] = plt.annotate(
            "",
            xy=[bbox.x0 + bbox.width * (1 - TCspan) + 0.23, bbox.y0 - bbox.height * 0.9],
            xytext=[bbox.x0 + bbox.width * (1 - TCspan) + 0.12, bbox.y0 - bbox.height * 0.4],
            xycoords=ax.get_figure().transFigure,
            arrowprops=dict(
                arrowstyle="simple,tail_width=0.7,head_width=0.5",
                linewidth=2,
                color="0.5",
                connectionstyle="angle3,angleA=0,angleB=60",
            ),
        )
        artists_dict["inflow_arrow_3"] = plt.annotate(
            "",
            xy=[bbox.x0 + bbox.width * (1 - TCspan), bbox.y0],
            xytext=[bbox.x0 + bbox.width * (1 - TCspan) + 0.13, bbox.y0 - bbox.height * 0.4],
            xycoords=ax.get_figure().transFigure,
            arrowprops=dict(
                arrowstyle="wedge,tail_width=0.7",
                linewidth=2,
                color="0.5",
                connectionstyle="angle3,angleA=0,angleB=-45",
            ),
        )
    if i is not None:
        artists_dict["plot_influx"].set_xlim(left=t[i] - windowleft, right=t[i] + windowright)
        if J.iloc[i] > 0:
            col = color_fun(C_J.iloc[i])
        else:
            col = "0.8"
        for a in range(1, 4):
            artists_dict[f"inflow_arrow_{a}"].arrow_patch.set_color(col)


def plot_outflux_and_conc(
    model,
    flux,
    sol,
    ax=None,
    sharex=None,
    i=None,
    artists_dict=OrderedDict(),
    do_init=True,
    cmap="inferno",
    TCspan=None,
    t_span=None,
    **kwargs,
):
    Q = model.data_df[flux]
    C_J = model.data_df[sol]
    C_Q = model.data_df[f"{sol} --> {flux}"]
    pQ = model.result["pQ"][:, :, list(model._fluxorder).index(flux)]
    qQ = pQ * Q.values
    mQ = model.result["mQ"][:, :, list(model._fluxorder).index(flux), list(model._solorder).index(sol)]
    PQ = np.r_[np.zeros((1, len(model.data_df))), np.cumsum(pQ, axis=0)] * dt
    QQ = PQ * Q.values
    CQ = np.where(qQ > 0, mQ / qQ, 0)
    t = model.data_df.index
    N = pQ.shape[0]
    windowleft = t_span
    windowright = t_span * TCspan / (1 - TCspan)
    if do_init:
        if ax is None:
            ax = plt.subplot(111, sharex=sharex)
        cmap = cm.get_cmap(cmap)
        norm = colors.Normalize(vmin=C_J.min(), vmax=C_J.max())
        color_fun = lambda C: cmap(norm(C))
        deltat = t[1] - t[0]
        # ax.bar(t, Q, color=color_fun(C_Q), width=1)
        # for j in range(len(pQ)):
        #    print(j)
        #    ax.bar([t[j]]*N, qQ[:,j], bottom=Q.iloc[j]-QQ[:-1,j]-qQ[:,j], color=color_fun(CQ[:,j]), width=1)
        # ax.bar(t, Q-QQ[-1,:], color=color_fun(C_old), width=1)
        verts = [
            list(
                zip(
                    [t[j], t[j], t[j] + deltat, t[j] + deltat],
                    [
                        Q.iloc[j] - QQ[k, j] - qQ[k, j],
                        Q.iloc[j] - QQ[k, j],
                        Q.iloc[j] - QQ[k, j],
                        Q.iloc[j] - QQ[k, j] - qQ[k, j],
                    ],
                )
            )
            for j in range(len(pQ))
            for k in range(N)
            if qQ[k, j] > 0
        ]
        verts += [
            list(zip([t[j], t[j], t[j] + deltat, t[j] + deltat], [0, Q.iloc[j] - QQ[-1, j], Q.iloc[j] - QQ[-1, j], 0]))
            for j in range(len(pQ))
        ]
        evalcolors = color_fun(CQ)
        Pcolors = [evalcolors[k, j] for j in range(len(pQ)) for k in range(N) if qQ[k, j] > 0]
        Pcolors += [color_fun(C_old) for j in range(len(pQ))]
        P = PolyCollection(verts, facecolors=Pcolors, edgecolors="none")
        ax.add_collection(P)
        ax.set_ylim(bottom=0, top=Q.quantile(0.95))
        ax.set_xlim(left=t[0] - windowleft, right=t[0] + windowright)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # ax.set_title("Outflow (e.g. discharge)", loc="left")
        bbox = ax.get_position()
        tbox = ax.get_tightbbox(ax.get_figure().canvas.get_renderer())
        TCmask = Rectangle(
            (bbox.x0 + bbox.width * (1 - TCspan), bbox.y0 - bbox.height * 0.5),
            bbox.width * (TCspan) * 1.1,
            bbox.height * 1.5,
            transform=ax.get_figure().transFigure,
            color="w",
            alpha=1.0,
            zorder=100,
            clip_on=False,
        )
        ax.add_patch(TCmask)
        artists_dict["plot_outflux"] = ax
    if i is not None:
        artists_dict["plot_outflux"].set_xlim(left=t[i] - windowleft, right=t[i] + windowright)


def plot_outflux(model, flux, ax=None, sharex=None, i=None, artists_dict=OrderedDict(), do_init=True):
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
        artists_dict["plot_outflux timeline"].set_data(2 * [model.data_df.index[i]], ax.get_ylim())


def plot_influx_conc(model, sol, ax=None, sharex=None, i=None, artists_dict=OrderedDict(), do_init=True):
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
        artists_dict["plot_influx_conc timeline"].set_data(2 * [model.data_df.index[i]], ax.get_ylim())


def plot_outflux_conc(model, flux, sol, ax=None, sharex=None, i=None, artists_dict=OrderedDict(), do_init=True):
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
        artists_dict["plot_outflux_conc timeline"].set_data(2 * [model.data_df.index[i]], ax.get_ylim())


def plot_SAS_cumulative(model, flux, ax=None, sharex=None, i=None, artists_dict=OrderedDict(), do_init=True):
    if do_init:
        if ax is None:
            ax = plt.subplot(111, sharex=sharex)
        if i is None:
            i = 0
        (artists_dict[f"plot_SAS {flux}"],) = ax.plot(
            model.sas_blends[flux].ST[i, :], model.sas_blends[flux].P[i, :], "bo-", lw=1.5
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
        artists_dict[f"plot_SAS {flux}"].set_data(model.sas_blends[flux].ST[i, :], model.sas_blends[flux].P[i, :])


def plot_transport_column_with_combined_timeseries(
    model,
    flux,
    sol,
    i=None,
    fig=None,
    axs=None,
    artists_dict=OrderedDict(),
    do_init=True,
    omega_max=None,
    ST_max=None,
    TC_frac=0.3,
    valvegap=0.3,
    hspace=0.2,
    **kwargs,
):
    if fig is None:
        fig = plt.figure(figsize=[6.5, 4.0])

    TCspan = 0.3
    t_span = 100

    if do_init:
        axJ = plt.subplot2grid((4, 3), (0, 0), colspan=3, fig=fig)
        axQ = plt.subplot2grid((4, 3), (1, 0), rowspan=3, colspan=3, fig=fig)
        axTCQ = inset_axes(
            axQ,
            width="100%",
            height="100%",
            bbox_to_anchor=((1 - TCspan) + 0.05, 0.0, TCspan, 0.82),
            bbox_transform=axQ.transAxes,
            borderpad=0,
        )
        plt.subplots_adjust(left=0.03, bottom=0.15, right=0.85, top=0.9, wspace=0.4, hspace=0.4)
        axTCQ.xaxis.set_visible(False)
        axTCQ.yaxis.set_visible(False)
        for loc, spine in axTCQ.spines.items():
            spine.set_visible(False)
        axTCQ.spines["left"].set_visible(False)
        axTC = axTCQ.inset_axes([1 - TC_frac, 0, TC_frac, 1])
        axTQ = axTCQ.inset_axes([0, 0, 1 - TC_frac - hspace, 1])

        if omega_max is None:
            sTs = np.r_[np.zeros((1, len(model.data_df))), model.result["sT"][:-1, :-1]]
            sTe = model.result["sT"][:, 1:]
            sT = (sTs + sTe) / 2
            pQ = model.result["pQ"][:, :, list(model._fluxorder).index(flux)]
            omega = pQ / sT
            omega_max = np.nanmax(omega)
    else:
        axTCQ, axTC, axTQ, axJ, axQ = axs

    plot_transport_column(
        model,
        flux,
        sol,
        i=i,
        axTC=axTC,
        axTQ=axTQ,
        artists_dict=artists_dict,
        TCspan=TCspan,
        t_span=t_span,
        omega_max=omega_max,
        ST_max=ST_max,
        do_init=do_init,
        **kwargs,
    )
    plot_influx_and_conc(
        model, sol, ax=axJ, i=i, artists_dict=artists_dict, TCspan=TCspan, t_span=t_span, do_init=do_init, **kwargs
    )
    plot_outflux_and_conc(
        model,
        flux,
        sol,
        ax=axQ,
        sharex=axJ,
        i=i,
        artists_dict=artists_dict,
        TCspan=TCspan,
        t_span=t_span,
        do_init=do_init,
        **kwargs,
    )

    return axTCQ, axTC, axTQ, axJ, axQ


def plot_transport_column_with_timeseries(model, flux, sol, i=None, fig=None, artists_dict=OrderedDict(), **kwargs):
    if fig is None:
        fig = plt.figure(figsize=[11.5, 4.0])
        fig.set_tight_layout(True)

    axTC = plt.subplot2grid((2, 3), (0, 1), rowspan=2, fig=fig)
    axJ = plt.subplot2grid((2, 3), (0, 0), fig=fig)
    axQ = plt.subplot2grid((2, 3), (0, 2), fig=fig)
    axCJ = plt.subplot2grid((2, 3), (1, 0), fig=fig)
    axCQ = plt.subplot2grid((2, 3), (1, 2), fig=fig)

    if "omega_max" in kwargs:
        omega_max = kwargs.pop("omega_max")
    else:
        sTs = np.r_[np.zeros(1, len(model.data_df)), model.result["sT"][:-1, :-1]]
        sTe = model.result["sT"][:, 1:]
        sT = (sTs + sTe) / 2
        pQ = model.result["pQ"][:, :, list(model._fluxorder).index(flux)]
        omega = pQ / sT
        omega_max = omega.nanmax()

    plot_transport_column(model, flux, sol, i=i, ax=axTC, artists_dict=artists_dict, omega_max=omega_max, **kwargs)
    plot_influx(model, ax=axJ, i=i, artists_dict=artists_dict)
    plot_outflux(model, flux, ax=axQ, sharex=axJ, i=i, artists_dict=artists_dict)
    plot_influx_conc(model, sol, ax=axCJ, sharex=axJ, i=i, artists_dict=artists_dict)
    plot_outflux_conc(model, flux, sol, ax=axCQ, sharex=axJ, i=i, artists_dict=artists_dict)

    return axTC, axJ, axQ, axCJ, axCQ


def make_transport_column_animation(model, flux, sol, fig=None, frames=None, **kwargs):
    global axs

    if frames is None:
        frames = range(model._timeseries_length - 1)

    if fig is None:
        fig = plt.figure(figsize=[6.5, 4.0])

    from matplotlib.animation import FuncAnimation

    artists = OrderedDict()

    axs = []

    def init():
        global axs
        i = 0
        axs = plot_transport_column_with_combined_timeseries(
            model, flux, sol, i=i, fig=fig, artists_dict=artists, do_init=True, **kwargs
        )
        return [artists[x] for x in artists.keys()]

    def update(frame):
        global axs
        i = frame
        print(f"frame={frame}")
        plot_transport_column_with_combined_timeseries(
            model, flux, sol, i=i, fig=fig, axs=axs, artists_dict=artists, do_init=False, **kwargs
        )
        return [artists[x] for x in artists.keys()]

    return FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)


# %%
def clip(f, x, xmin, xmax):
    return np.where((xmin <= x) & (x <= xmax), f, 0)


from scipy.stats import gamma

# if __name__=="__main__":
if True:
    scenarios = {
        "Uniform": {
            "spec": {"ST": ["S_m", "S_m0"]},
            "pQdisc": lambda delta, i: (-1 + np.exp(delta)) ** 2 / (np.exp((1 + i) * delta) * delta),
            "pQdisc0": lambda delta: (1 + np.exp(delta) * (-1 + delta)) / (np.exp(delta) * delta),
            "subplot": 0,
            "distname": "Uniform",
            "bounded_ST": True,
            "SASfun": lambda ST, df: clip(1 / (df["S_m0"] - df["S_m"]), ST, df["S_m"], df["S_m0"]),
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
            "bounded_ST": False,
            "distname": "Gamma(1.0)",
        },
        "Gamma(0.25)": {
            "spec": {
                "func": "gamma",
                "args": {"a": 0.25, "scale": "S_0", "loc": "S_m"},
            },
            "distname": "Gamma(0.5)",
            "SASfun": lambda ST, df: gamma.pdf(ST, a=0.25, scale=df["S_0"], loc=df["S_m"]),
            "bounded_ST": False,
        },
        "Biased old (Beta)": {
            "spec": {
                "func": "beta",
                "args": {"a": 2.0 - 0.00001, "b": 1.0 - 0.00001, "scale": "S_0", "loc": "S_m"},
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
            "bounded_ST": True,
            "distname": "Beta(2,1)",
        },
        "Biased young (Beta)": {
            "spec": {
                "func": "beta",
                "args": {"a": 1.0 - 0.00001, "b": 2.0 - 0.00001, "scale": "S_0", "loc": "S_m"},
            },
            "pQdisc": lambda delta, i: (
                (2 * delta) / ((1 + (-1 + i) * delta) * (1 + i * delta) * (1 + delta + i * delta))
            ),
            "pQdisc0": lambda delta: delta / (1 + delta),
            "subplot": 3,
            "bounded_ST": True,
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
            "bounded_ST": True,
            "distname": "Beta(1/2,1)",
        },
        "Partial piston (Beta)": {
            "spec": {
                "func": "beta",
                "args": {"a": 1.0, "b": 1.0 / 2, "scale": "S_0", "loc": "S_m"},
            },
            "pQdisc": lambda delta, i: delta / 2,
            "pQdisc0": lambda delta: delta / 4,
            "subplot": 5,
            "bounded_ST": True,
            "distname": "Beta(1,1/2)",
        },
        "Biased old (Kumaraswamy)": {
            "spec": {
                "func": "kumaraswamy",
                "args": {"a": 2.0 - 0.00001, "b": 1.0 - 0.00001, "scale": "S_0", "loc": "S_m"},
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
            "bounded_ST": True,
            "distname": "Kumaraswamy(2,1)",
        },
        "Biased young (Kumaraswamy)": {
            "spec": {
                "func": "kumaraswamy",
                "args": {"a": 1.0 - 0.00001, "b": 2.0 - 0.00001, "scale": "S_0", "loc": "S_m"},
            },
            "pQdisc": lambda delta, i: (
                (2 * delta) / ((1 + (-1 + i) * delta) * (1 + i * delta) * (1 + delta + i * delta))
            ),
            "pQdisc0": lambda delta: delta / (1 + delta),
            "subplot": 3,
            "bounded_ST": True,
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
            "bounded_ST": True,
            "distname": "Kumaraswamy(1/2,1)",
            "SASfun": lambda ST, df: clip(
                1 / ((ST - df["S_m"]) / (df["S_0"] - df["S_m"])) ** 0.5 / 2 / (df["S_0"] - df["S_m"]),
                ST,
                df["S_m"],
                df["S_0"],
            ),
        },
        "Partial piston (Kumaraswamy)": {
            "spec": {
                "func": "kumaraswamy",
                "args": {"a": 1.0, "b": 1.0 / 2, "scale": "S_0", "loc": "S_m"},
            },
            "pQdisc": lambda delta, i: delta / 2,
            "pQdisc0": lambda delta: delta / 4,
            "subplot": 5,
            "bounded_ST": True,
            "distname": "Kumaraswamy(1,1/2)",
            "SASfun": lambda ST, df: clip(
                1 / ((df["S_0"] - ST) / (df["S_0"] - df["S_m"])) ** 0.5 / 2 / (df["S_0"] - df["S_m"]),
                ST,
                df["S_m"],
                df["S_0"],
            ),
        },
        "Uniform and Linear reservoir": {
            "spec": {"ST": [0, "S_0"]},
            "pQdisc": lambda J, Q, delta, i, j: J[j - i] / Q[j] * (1 - np.exp(-delta)),
            "pQdisc0": lambda J, Q, delta, j: J[j] / Q[j] * (delta - (1 - np.exp(-delta))) / delta,
        },
        "Inverse Storage Effect": {
            "spec": {
                "func": "kumaraswamy",
                "args": {"a": "a_IS", "b": 1.0, "scale": "S_0", "loc": "S_m"},
            },
            "distname": "Kumaraswamy(a,1)",
            "bounded_ST": True,
            "SASfun": lambda ST, df: clip(
                df["a_IS"] * ((ST - df["S_m"]) / (df["S_0"] - df["S_m"])) ** (df["a_IS"] - 1) / (df["S_0"] - df["S_m"]),
                ST,
                df["S_m"],
                df["S_0"],
            ),
        },
    }

    # %%
    timeseries_length = 200
    data_df = pd.read_csv("../lower_hafren/data.csv")
    data_df = data_df[2025 : 2025 + timeseries_length]
    dt = 1

    # %%
    data_df["S_m"] = 0
    S_init = 300
    S0factor = (data_df["J"] - data_df["Q"]).sum() / data_df["ET"].sum()
    data_df["ET"] = data_df["ET"] * S0factor
    S_0 = S_init + (data_df["J"] - data_df["Q"] - data_df["ET"]).cumsum()
    Q = data_df["Q"]
    data_df["S_0"] = S_0
    data_df["S_m0"] = S_0
    z = (S_0 - S_0.min()) / (S_0.max() - S_0.min())
    data_df["a_IS"] = 1.05 - z

    C_J = np.zeros(timeseries_length)
    C_J[0] = 0
    inno = 0.02
    np.random.seed(1)
    for i in range(timeseries_length - 1):
        C_J[i + 1] = (1 - inno) * C_J[i] + np.random.randn(1)
    data_df["C"] = gaussian.cdf((C_J - C_J.mean()) / C_J.std())

    C_old = 0
    solute_parameters = {
        "C": {
            "C_old": C_old,
        }
    }

    n_substeps = 4
    debug = False
    verbose = True
    jacobian = False

    # for name in ['Inverse Storage Effect']:
    for name in [
        "Partial bypass (Kumaraswamy)",
        "Partial piston (Kumaraswamy)",
        "Uniform",
        "Inverse Storage Effect",
        "Gamma(0.25)",
    ]:
        bm = scenarios[name]
        print(name)
        i = np.arange(timeseries_length)

        sas_specs = {"Q": {f"{name}_SAS": bm["spec"]}, "ET": {"ET_SAS": {"ST": ["S_m", "S_m0"]}}}

        model = Model(
            data_df,
            sas_specs=sas_specs,
            solute_parameters=solute_parameters,
            debug=debug,
            verbose=verbose,
            dt=dt,
            n_substeps=n_substeps,
            jacobian=jacobian,
            max_age=timeseries_length,
            warning=True,
        )
        model.run()
        data_df = model.data_df

        F = make_transport_column_animation(
            model, "Q", "C", cmap="turbo", omega_max=0.01, ST_max=500, bounded_ST=bm["bounded_ST"]
        )
        from matplotlib import animation

        F.save(f"{name}.mp4", dpi=300, writer=animation.FFMpegWriter(fps=5))

        # plot_transport_column_with_combined_timeseries(model, 'Q', 'C', i=103, cmap='turbo', omega_max=0.01, ST_max=500)
        # plt.savefig(f"{name}.png", dpi=300)
