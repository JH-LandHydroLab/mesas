"""Smoke tests for mesas.utils.vis and SAS spec plotting.

These exercise every public plotting function on a small model using the
Agg backend. Regression coverage for:
- Component.plot / SAS_Spec.plot (was: AttributeError, sas_fun is a list)
- mutable default artists_dict shared across calls
- missing record_state guard (was: bare IndexError)
- animation re-creating all axes/patches every frame (memory leak)
- np.NaN removed in NumPy 2.0
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mesas.sas.model import Model
from mesas.utils import vis


@pytest.fixture()
def model():
    n = 40
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "J": 1.0 + 0.5 * rng.random(n),
            "Q": np.ones(n),
            "C": 2.0 + rng.random(n),
        }
    )
    m = Model(
        data_df=df,
        sas_specs={"Q": {"u": {"ST": [0.0, 5.0]}}},
        solute_parameters={"C": {"C_old": 2.0}},
        verbose=False,
        record_state=True,
    )
    m.run()
    return m


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_plot_transport_column(model):
    vis.plot_transport_column(model, "Q", "C", i=10)


def test_plot_transport_column_requires_recorded_state():
    n = 30
    df = pd.DataFrame({"J": np.ones(n), "Q": np.ones(n), "C": np.ones(n)})
    m = Model(
        data_df=df,
        sas_specs={"Q": {"u": {"ST": [0.0, 5.0]}}},
        solute_parameters={"C": {"C_old": 1.0}},
        verbose=False,
    )  # record_state defaults to False
    m.run()
    with pytest.raises(ValueError, match="record_state"):
        vis.plot_transport_column(m, "Q", "C", i=10)


def test_timeseries_cursor_plots(model):
    vis.plot_influx(model, i=5)
    vis.plot_outflux(model, "Q", i=5)
    vis.plot_influx_conc(model, "C", i=5)
    vis.plot_outflux_conc(model, "Q", "C", i=5)
    vis.plot_SAS_cumulative(model, "Q", i=5)


def test_no_shared_state_between_calls(model):
    """Mutable default argument regression: two independent calls must not
    share artists."""
    d1, d2 = {}, {}
    vis.plot_influx(model, i=1, artists_dict=d1)
    plt.close("all")
    vis.plot_influx(model, i=2, artists_dict=d2)
    assert d1["plot_influx timeline"] is not d2["plot_influx timeline"]
    # and calls relying on the default must not blow up after closes
    vis.plot_influx(model, i=3)
    plt.close("all")
    vis.plot_influx(model, i=4)


def test_dashboard(model):
    axes = vis.plot_transport_column_with_timeseries(model, "Q", "C", i=10)
    assert len(axes) == 5


def test_animation_does_not_leak_axes(model):
    """The animation must update artists in place, not rebuild the figure."""
    anim = vis.make_transport_column_animation(model, "Q", "C", frames=range(3))
    fig = anim._fig
    anim._init_draw()
    n_axes_after_init = len(fig.axes)
    n_artists = sum(len(ax.patches) + len(ax.lines) for ax in fig.axes)
    for frame in [1, 2]:
        anim._draw_frame(frame)
    assert len(fig.axes) == n_axes_after_init
    assert sum(len(ax.patches) + len(ax.lines) for ax in fig.axes) == n_artists


def test_component_and_spec_plot(model):
    """specs.py Component.plot / SAS_Spec.plot regression (sas_fun is a list)."""
    fig, ax = plt.subplots()
    model.sas_specs["Q"].components["u"].plot(ax=ax)
    model.sas_specs["Q"].plot()


def test_sas_function_plot(model):
    fig, ax = plt.subplots()
    model.sas_specs["Q"].components["u"].sas_fun[0].plot(ax=ax)
