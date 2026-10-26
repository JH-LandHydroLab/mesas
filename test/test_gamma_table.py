"""Accuracy tests for the gamma CDF lookup table in the Numba solver.

Regression tests for two bugs in the original lookup-table implementation:

1. The table domain was fixed at [0, 40] and returned 1.0 for x >= 40,
   which is badly wrong for large shape parameters (gammainc(50, 40) = 0.07).
2. The grid was uniform in x, which cannot resolve the integrable pdf
   singularity at x -> 0 for shape < 1 (the typical hydrologic case),
   misallocating young-water TTD mass between adjacent age bins.

The fixed implementation sizes the domain from the shape parameter and
tabulates on a grid uniform in y = x**min(a/2, 1).
"""

import numpy as np
import pytest
from scipy.special import gammainc

from mesas.sas import _solve_numba as sn

SHAPES = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0, 5.0, 25.0, 50.0, 100.0, 500.0]


@pytest.mark.parametrize("a", SHAPES)
def test_table_matches_scipy(a):
    """Table CDF agrees with scipy.special.gammainc across the full domain."""
    table = sn._build_gamma_table(a)
    x_max, dy, a_exp = sn._gamma_table_meta_for_shape(a)
    x = np.concatenate(
        [
            np.geomspace(1e-12, 1e-3, 200),
            np.linspace(1e-3, min(5.0, x_max), 2000),
            np.linspace(0.0, x_max * 1.05, 5000),
        ]
    )
    approx = np.array([sn._gamma_cdf_table(xi, 0.0, 1.0, table, x_max, dy, a_exp) for xi in x])
    exact = gammainc(a, x)
    assert np.max(np.abs(approx - exact)) < 1e-5


@pytest.mark.parametrize("a", SHAPES)
def test_saturation_beyond_x_max_is_valid(a):
    """Returning 1.0 beyond x_max is justified: the true CDF is within 1e-12 of 1."""
    x_max, _, _ = sn._gamma_table_meta_for_shape(a)
    assert 1.0 - gammainc(a, x_max) < 1e-12


def test_large_shape_not_saturated():
    """Regression: old table returned 1.0 at x=45 for a=50 (truth: 0.2468)."""
    a = 50.0
    table = sn._build_gamma_table(a)
    x_max, dy, a_exp = sn._gamma_table_meta_for_shape(a)
    v = sn._gamma_cdf_table(45.0, 0.0, 1.0, table, x_max, dy, a_exp)
    assert v == pytest.approx(gammainc(a, 45.0), abs=1e-6)


def test_small_shape_near_zero_differences():
    """Regression: pQ-style CDF differences near x=0 for shape<1.

    With the old uniform grid (dx=0.002) a difference taken inside the first
    cell was off by up to 94 % relative for a=0.3.
    """
    a = 0.3
    table = sn._build_gamma_table(a)
    x_max, dy, a_exp = sn._gamma_table_meta_for_shape(a)
    x0, x1 = 0.0004, 0.0016
    num = sn._gamma_cdf_table(x1, 0.0, 1.0, table, x_max, dy, a_exp) - sn._gamma_cdf_table(
        x0, 0.0, 1.0, table, x_max, dy, a_exp
    )
    den = gammainc(a, x1) - gammainc(a, x0)
    assert num == pytest.approx(den, rel=1e-3)


def test_loc_scale_handling():
    """Location/scale shift and stretch are applied before the table lookup."""
    a = 0.7
    loc, scale = 10.0, 30.0
    table = sn._build_gamma_table(a)
    x_max, dy, a_exp = sn._gamma_table_meta_for_shape(a)
    for ST in [0.0, 10.0, 11.0, 40.0, 400.0, 1e6]:
        expected = 0.0 if ST <= loc else gammainc(a, (ST - loc) / scale)
        v = sn._gamma_cdf_table(ST, loc, scale, table, x_max, dy, a_exp)
        assert v == pytest.approx(expected, abs=1e-6)


def test_model_gamma_large_shape_end_to_end():
    """End-to-end: model with gamma shape=50 must match the exact-CDF path.

    The old table gave mean C_Q = 2.18 vs 6.60 exact for this configuration.
    The exact path is forced by making the shape time-varying at the 1e-9
    level (constant-shape detection then rejects the table, while the
    results are perturbed far below the comparison tolerance).
    """
    import pandas as pd

    from mesas.sas.model import Model

    n = 200
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "J": np.full(n, 1.0),
            "Q": np.full(n, 1.0),
            "C": 5.0 + 3.0 * rng.random(n),
            "a_col": 50.0 + 1e-9 * np.arange(n),
        }
    )
    spec_table = {"Q": {"g": {"func": "gamma", "args": {"a": 50.0, "scale": 2.0, "loc": 0.0}}}}
    spec_exact = {"Q": {"g": {"func": "gamma", "args": {"a": "a_col", "scale": 2.0, "loc": 0.0}}}}
    sol = {"C": {"C_old": 5.0}}

    out = {}
    for name, spec in [("table", spec_table), ("exact", spec_exact)]:
        m = Model(
            data_df=df.copy(),
            sas_specs=spec,
            solute_parameters=sol,
            verbose=False,
        )
        m.run()
        out[name] = m.data_df["C --> Q"].to_numpy()

    np.testing.assert_allclose(out["table"], out["exact"], atol=1e-5)
