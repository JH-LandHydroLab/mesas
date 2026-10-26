"""Pure-Python + Numba replacement for the Fortran ``solve.f90`` solver.

This module implements the SAS (StorAge Selection) transport solver using
the method of characteristics with Runge-Kutta time integration.  The
algorithm is numerically equivalent to the original Fortran implementation,
but written in Python and accelerated with Numba ``@njit`` for performance
parity.

The public entry point is :func:`solve`, which has the same signature and
return values as the Fortran ``solvesas`` subroutine (via f2py).
"""

from __future__ import annotations

import math
import os

import numpy as np

try:
    from numba import njit, prange
except ImportError:
    # Fallback: run without JIT compilation
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def decorator(func):
            return func

        return decorator


# Set to False to disable JIT compilation during development/debugging
_USE_NUMBA = True

# Names of the recorded state arrays users can select via record_arrays
RECORDABLE_ARRAYS = frozenset({"sT", "pQ", "mT", "mQ", "mR", "water_balance", "solute_balance"})


def _maybe_jit(func):
    """Apply @njit if Numba is available and enabled."""
    if _USE_NUMBA:
        return njit(cache=True)(func)
    return func


def _maybe_jit_fastmath(func):
    """Apply @njit(fastmath=True) for hot numerical loops."""
    if _USE_NUMBA:
        return njit(cache=True, fastmath=True)(func)
    return func


def _maybe_jit_parallel(func):
    """Apply @njit(parallel=True, fastmath=True) for prange loops.

    Set the environment variable ``MESAS_PARALLEL=0`` (before import) to
    compile the hot loop serially instead — useful when parallelising at
    the process level (e.g. multiprocessing calibration), where oversubscribing
    cores with numba threads hurts throughput. Thread count can also be
    limited with ``NUMBA_NUM_THREADS``.
    """
    if _USE_NUMBA:
        parallel = os.environ.get("MESAS_PARALLEL", "1") != "0"
        return njit(cache=True, fastmath=True, parallel=parallel)(func)
    return func


def _maybe_jit_inline(func):
    """Apply @njit with forced inlining and fastmath for innermost hot functions."""
    if _USE_NUMBA:
        return njit(cache=True, fastmath=True, inline="always")(func)
    return func


# ---------------------------------------------------------------------------
# Special-function helpers (Numba-compatible, no scipy dependency)
# These use inline='always' + fastmath=True since they are called millions
# of times in the inner loop. Inlining eliminates function call overhead.
# ---------------------------------------------------------------------------


@_maybe_jit_inline
def _alngam(x):
    """Log-gamma function (Stirling-based, matches AS 245 in solve.f90)."""
    if x <= 0.0:
        return 0.0
    # Use Stirling's series for lgamma
    # For small x, use recurrence relation gamma(x+1) = x*gamma(x)
    result = 0.0
    y = x
    while y < 8.0:
        result -= math.log(y)
        y += 1.0
    # Stirling approx for y >= 8
    r = 1.0 / y
    r2 = r * r
    result += (y - 0.5) * math.log(y) - y + 0.9189385332046727  # 0.5*ln(2*pi)
    result += r * (1.0 / 12.0 - r2 * (1.0 / 360.0 - r2 * (1.0 / 1260.0 - r2 * (1.0 / 1680.0 - r2 / 1188.0))))
    return result


@_maybe_jit_inline
def _gammad(x, p):
    """Incomplete gamma integral P(p, x) = gammainc(p, x).

    Faithful port of the AS 239 / solve.f90 ``gammad`` implementation.
    """
    tol = 1.0e-14
    elimit = -88.0
    oflo = 1.0e37

    if x < 0.0 or p <= 0.0:
        return 0.0
    if x == 0.0:
        return 0.0

    # Pearson's series expansion (x <= 1 or x < p)
    if x <= 1.0 or x < p:
        arg = p * math.log(x) - x - _alngam(p + 1.0)
        cc = 1.0
        result = 1.0
        a = p
        while True:
            a += 1.0
            cc = cc * x / a
            result += cc
            if cc <= tol:
                break
        arg = arg + math.log(result)
        if arg >= elimit:
            return math.exp(arg)
        return 0.0
    else:
        # Continued fraction expansion
        arg = p * math.log(x) - x - _alngam(p)
        a = 1.0 - p
        b = a + x + 1.0
        cc = 0.0
        pn1 = 1.0
        pn2 = x
        pn3 = x + 1.0
        pn4 = x * b
        result = pn3 / pn4
        while True:
            a += 1.0
            b += 2.0
            cc += 1.0
            an = a * cc
            pn5 = b * pn3 - an * pn1
            pn6 = b * pn4 - an * pn2
            if abs(pn6) > 0.0:
                rn = pn5 / pn6
                if abs(result - rn) <= min(tol, tol * rn):
                    arg = arg + math.log(result)
                    if arg >= elimit:
                        return 1.0 - math.exp(arg)
                    return 1.0
                result = rn
            pn1 = pn3
            pn2 = pn4
            pn3 = pn5
            pn4 = pn6
            # Rescale to prevent overflow
            if abs(pn5) >= oflo:
                pn1 /= oflo
                pn2 /= oflo
                pn3 /= oflo
                pn4 /= oflo
        # Should not reach here
        return 1.0 - math.exp(arg) * result


@_maybe_jit_inline
def _betain(x, p, q):
    """Incomplete beta function I_x(p, q).

    Matches AS 63 / solve.f90 ``betain`` implementation.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Use the continued fraction in the region where it converges faster
    psq = p + q
    if p < psq * x:
        xx = 1.0 - x
        cx = x
        pp = q
        qq = p
        indx = True
    else:
        xx = x
        cx = 1.0 - x
        pp = p
        qq = q
        indx = False

    beta = _alngam(p) + _alngam(q) - _alngam(p + q)
    term = 1.0
    ai = 1.0
    result = 1.0
    ns = int(qq + cx * psq)

    # Soper's reduction
    rx = xx / cx
    temp = qq - ai
    if ns == 0:
        rx = xx

    for _ in range(1000):
        term = term * temp * rx / (pp + ai)
        result += term
        temp = abs(term)
        if temp <= 1e-14 and temp <= 1e-14 * abs(result):
            result = result * math.exp(pp * math.log(xx) + (qq - 1.0) * math.log(cx) - beta) / pp
            if indx:
                result = 1.0 - result
            return result
        ai += 1.0
        ns -= 1
        if ns >= 0:
            temp = qq - ai
            if ns == 0:
                rx = xx
        else:
            temp = psq
            psq += 1.0

    # Should not reach here, but return best estimate
    result = result * math.exp(pp * math.log(xx) + (qq - 1.0) * math.log(cx) - beta) / pp
    if indx:
        result = 1.0 - result
    return result


# ---------------------------------------------------------------------------
# SAS function evaluators
# ---------------------------------------------------------------------------


@_maybe_jit_inline
def _piecewise_linear_cdf(ST, SAS_args, P_list, grad_precalc, ai, jt_c, nargs):
    """Evaluate a piecewise-linear SAS function at cumulative storage ST.

    Uses full arrays + offset ``ai`` and column ``jt_c`` to avoid Numba
    issues with dynamically-sized array slices.
    """
    if ST <= SAS_args[ai, jt_c]:
        return P_list[ai, jt_c]
    for i in range(nargs):
        if ST < SAS_args[ai + i, jt_c]:
            ia = i - 1
            return P_list[ai + ia, jt_c] + (ST - SAS_args[ai + ia, jt_c]) * grad_precalc[ai + ia, jt_c]
    return P_list[ai + nargs - 1, jt_c]


@_maybe_jit_inline
def _kumaraswamy_cdf(ST, loc, scale, a, b):
    """Kumaraswamy CDF: F(x) = 1 - (1 - x^a)^b."""
    x = (ST - loc) / scale
    x = min(max(0.0, x), 1.0)
    return 1.0 - (1.0 - x**a) ** b


@_maybe_jit_inline
def _beta_cdf(ST, loc, scale, a, b):
    """Beta CDF via incomplete beta function."""
    x = (ST - loc) / scale
    x = min(max(0.0, x), 1.0)
    return _betain(x, a, b)


@_maybe_jit_inline
def _gamma_cdf(ST, loc, scale, a):
    """Gamma CDF via incomplete gamma integral."""
    if ST <= loc:
        return 0.0
    x = (ST - loc) / scale
    return _gammad(x, a)


# ---------------------------------------------------------------------------
# Gamma CDF lookup table for fast evaluation
# ---------------------------------------------------------------------------

_GAMMA_TABLE_SIZE = 20000


@_maybe_jit
def _gamma_table_meta_for_shape(a):
    """Compute (x_max, dy, a_exp) lookup-table metadata for shape ``a``.

    The table domain [0, x_max] must cover the CDF up to where it is
    indistinguishable from 1: gammainc(a, a + 12*sqrt(a) + 40) < 1e-12 from
    1 for all a, so beyond x_max the lookup may safely return 1.0.

    The table is uniform in the transformed variable y = x**a_exp with
    a_exp = min(a/2, 1). Near x = 0 the CDF behaves like x**a, i.e. like
    y**2 in the transformed variable, so linear interpolation stays accurate
    even for a < 1 where the gamma pdf has an integrable singularity at 0
    that a uniform-in-x grid cannot resolve.
    """
    x_max = a + 12.0 * np.sqrt(a) + 40.0
    a_exp = min(a / 2.0, 1.0)
    y_max = x_max**a_exp
    dy = y_max / _GAMMA_TABLE_SIZE
    return x_max, dy, a_exp


@_maybe_jit
def _build_gamma_table(a):
    """Build lookup table for gammainc(a, x), uniform in y = x**a_exp.

    Returns a 1D array of size ``_GAMMA_TABLE_SIZE + 1`` containing
    gammainc(a, x) at the grid points x = (i*dy)**(1/a_exp), where
    (x_max, dy, a_exp) come from :func:`_gamma_table_meta_for_shape`.
    """
    x_max, dy, a_exp = _gamma_table_meta_for_shape(a)
    inv_a_exp = 1.0 / a_exp
    table = np.empty(_GAMMA_TABLE_SIZE + 1)
    table[0] = 0.0
    for i in range(1, _GAMMA_TABLE_SIZE + 1):
        x = (i * dy) ** inv_a_exp
        table[i] = _gammad(x, a)
    return table


@_maybe_jit_inline
def _gamma_cdf_table(ST, loc, scale, table, x_max, dy, a_exp):
    """Gamma CDF via precomputed lookup table + linear interpolation.

    The table is uniform in y = x**a_exp (see _build_gamma_table).
    """
    if ST <= loc:
        return 0.0
    x = (ST - loc) / scale
    if x >= x_max:
        return 1.0
    if a_exp == 1.0:
        y = x
    else:
        y = x**a_exp
    fy = y / dy
    i = int(fy)
    if i >= _GAMMA_TABLE_SIZE:
        return table[_GAMMA_TABLE_SIZE]
    frac = fy - i
    return table[i] + frac * (table[i + 1] - table[i])


# ---------------------------------------------------------------------------
# Pre-computation helpers
# ---------------------------------------------------------------------------


@_maybe_jit_fastmath
def _precompute_gradients(
    SAS_args,
    P_list,
    component_type,
    component_index_list,
    args_index_list,
    numargs_list,
    numflux,
    timeseries_length,
    numargs_total,
):
    """Pre-compute slopes for piecewise-linear SAS function segments."""
    grad_precalc = np.zeros((numargs_total, timeseries_length))
    for iq in range(numflux):
        ic_start = int(component_index_list[iq])
        ic_end = int(component_index_list[iq + 1])
        for ic_ in range(ic_start, ic_end):
            if component_type[ic_] == -1:
                na = int(numargs_list[ic_])
                ai = int(args_index_list[ic_])
                for ia in range(na - 1):
                    ii = ai + ia
                    for jt in range(timeseries_length):
                        s1 = float(SAS_args[ii + 1, jt])
                        s0 = float(SAS_args[ii, jt])
                        dS = s1 - s0
                        if dS != 0.0:
                            p1 = float(P_list[ii + 1, jt])
                            p0 = float(P_list[ii, jt])
                            grad_precalc[ii, jt] = (p1 - p0) / dS
    return grad_precalc


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------


@_maybe_jit
def _solve_core(
    J_fullstep,  # (timeseries_length,)
    Q_fullstep,  # (timeseries_length, numflux)
    SAS_args,  # (numargs_total, timeseries_length) -- note: transposed
    P_list,  # (numargs_total, timeseries_length) -- note: transposed
    weights_fullstep,  # (timeseries_length, numcomponent_total)
    sT_init_fullstep,  # (max_age,)
    dt,  # scalar
    verbose,  # bool
    mT_init_fullstep,  # (max_age, numsol)
    C_J_fullstep,  # (timeseries_length, numsol)
    alpha_fullstep,  # (timeseries_length, numflux, numsol)
    k1_fullstep,  # (timeseries_length, numsol)
    C_eq_fullstep,  # (timeseries_length, numsol)
    C_old,  # (numsol,)
    n_substeps,  # int
    component_type,  # (numcomponent_total,)
    numcomponent_list,  # (numflux,)
    numargs_list,  # (numcomponent_total,)
    output_these_fullsteps,  # (num_output_fullsteps,)
    num_scheme,  # int
    numflux,  # int
    numsol,  # int
    max_age,  # int
    timeseries_length,  # int
    num_output_fullsteps,  # int
    numcomponent_total,  # int
    numargs_total,  # int
    sT_outputstep,  # (max_age, num_output_fullsteps + 1), age-first; (1, 1) if not recorded
    mT_outputstep,  # (max_age, num_output_fullsteps + 1, numsol); (1, 1, 1) if not recorded
    pQ_outputstep,  # (max_age, num_output_fullsteps, numflux); (1, 1, 1) if not recorded
    mQ_outputstep,  # (max_age, num_output_fullsteps, numflux, numsol); (1, 1, 1, 1) if not recorded
    mR_outputstep,  # (max_age, num_output_fullsteps, numsol); (1, 1, 1) if not recorded
    rec_sT,  # bool: write into sT_outputstep
    rec_mT,  # bool
    rec_pQ,  # bool
    rec_mQ,  # bool
    rec_mR,  # bool
):
    """Core SAS solver — Numba-compiled equivalent of solveSAS in solve.f90.

    Output arrays are allocated by the caller (`solve`) in age-first layout
    and may be RAM ndarrays or disk-backed memmaps of float64 or float32;
    the kernels only ever write to them (accumulate), never read.
    """

    total_num_substeps = timeseries_length * n_substeps
    dt_substep = dt / n_substeps
    norm = 1.0 / n_substeps / n_substeps

    # RK coefficients
    rk4_stepfraction = np.array([0.0, 0.5, 0.5, 1.0, 1.0])
    rk4_coeff = np.array([1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0])
    rk2_stepfraction = np.array([0.0, 1.0, 1.0])
    rk2_coeff = np.array([0.5, 0.5])

    # Pre-compute index mappings
    args_index_list = np.zeros(numcomponent_total + 1, dtype=np.int64)
    component_index_list = np.zeros(numflux + 1, dtype=np.int64)
    for iq in range(numflux):
        component_index_list[iq + 1] = component_index_list[iq] + numcomponent_list[iq]
        for ic in range(component_index_list[iq], component_index_list[iq + 1]):
            args_index_list[ic + 1] = args_index_list[ic] + numargs_list[ic]

    # Pre-compute gradients for piecewise-linear SAS functions
    grad_precalc = _precompute_gradients(
        SAS_args,
        P_list,
        component_type,
        component_index_list,
        args_index_list,
        numargs_list,
        numflux,
        timeseries_length,
        numargs_total,
    )

    # Pre-compute gamma CDF lookup tables (one per gamma component).
    # The table is valid when the shape parameter 'a' is constant across
    # timesteps. If 'a' varies, we set gamma_use_table[ic]=False and fall
    # back to direct evaluation via _gammad.
    gamma_tables = np.zeros((numcomponent_total, _GAMMA_TABLE_SIZE + 1))
    gamma_use_table = np.zeros(numcomponent_total, dtype=np.int64)
    # Per-component table metadata: columns are (x_max, dy, a_exp)
    gamma_table_meta = np.zeros((numcomponent_total, 3))
    for ic in range(numcomponent_total):
        if component_type[ic] == 1:
            ai = int(args_index_list[ic])
            a_shape = float(SAS_args[ai + 2, 0])
            # Check if 'a' is constant across all timesteps
            a_is_constant = True
            for jt in range(1, timeseries_length):
                if float(SAS_args[ai + 2, jt]) != a_shape:
                    a_is_constant = False
                    break
            if a_is_constant:
                gamma_tables[ic, :] = _build_gamma_table(a_shape)
                x_max, dy, a_exp = _gamma_table_meta_for_shape(a_shape)
                gamma_table_meta[ic, 0] = x_max
                gamma_table_meta[ic, 1] = dy
                gamma_table_meta[ic, 2] = a_exp
                gamma_use_table[ic] = 1

    # Allocate the always-in-RAM outputs (O(N), small); the O(N^2) recorded
    # state arrays are allocated by the caller and passed in.
    C_Q_fullstep = np.zeros((timeseries_length, numflux, numsol))

    P_old_fullstep = np.ones((timeseries_length, numflux))

    # Working arrays
    STcum_topbot_start = np.zeros((total_num_substeps + 1, 2))
    pQ_temp = np.zeros((total_num_substeps, numflux))
    pQ_aver = np.zeros((total_num_substeps, numflux))
    mQ_temp = np.zeros((total_num_substeps, numflux, numsol))
    mQ_aver = np.zeros((total_num_substeps, numflux, numsol))
    mR_temp = np.zeros((total_num_substeps, numsol))
    mR_aver = np.zeros((total_num_substeps, numsol))
    sT_start = np.zeros(total_num_substeps)
    mT_start = np.zeros((total_num_substeps, numsol))
    mT_temp = np.zeros((total_num_substeps, numsol))
    jt_fullstep_at = np.zeros(total_num_substeps, dtype=np.int64)
    jt_substep_at = np.zeros(total_num_substeps, dtype=np.int64)

    # Initial conditions for the recorded state are set by the caller
    # (column 0 of the age-first sT/mT arrays).

    # Unified RK stage tables: stagefrac[rk] is the evaluation fraction of
    # stage rk; stagefrac[rk+1] is the propagation fraction to the next stage
    # state (the last entry, 1.0, is the final full-step update from averages).
    if num_scheme == 1:
        nstage = 1
        stagefrac = np.array([0.0, 1.0])
        rk_coeffs = np.array([1.0])
    elif num_scheme == 2:
        nstage = 2
        stagefrac = rk2_stepfraction
        rk_coeffs = rk2_coeff
    else:
        nstage = 4
        stagefrac = rk4_stepfraction
        rk_coeffs = rk4_coeff

    iT_prev_fullstep = -1

    # If the initial age-ranked storage (and mass) are identically zero, then
    # at age step iT the trailing iT characteristics (the "wrapped" slots that
    # carry initial-condition water) are identically zero: their sT, mT, pQ,
    # mQ, mR are all zero and contribute nothing. We can restrict the hot
    # loops to the active front [0, total_num_substeps - iT_substep).
    zero_init = True
    for t in range(max_age):
        if sT_init_fullstep[t] != 0.0:
            zero_init = False
            break
        for s in range(numsol):
            if mT_init_fullstep[t, s] != 0.0:
                zero_init = False
                break
        if not zero_init:
            break

    # =========================================================================
    # MAIN LOOP: iterate over water age (outer) and substeps (inner)
    # =========================================================================
    for iT_fullstep in range(max_age):
        for substep in range(n_substeps):
            iT_substep = iT_fullstep * n_substeps + substep

            if zero_init:
                n_active = total_num_substeps - iT_substep
            else:
                n_active = total_num_substeps

            # Map each characteristic to its current full timestep
            for c in range(n_active):
                jt_sub = (c + iT_substep) % total_num_substeps
                jt_substep_at[c] = jt_sub
                jt_is_which_substep = jt_sub % n_substeps
                jt_fullstep_at[c] = (jt_sub - jt_is_which_substep) // n_substeps

            # Reset RK averages (n_active slots plus the just-retired slot)
            n_reset = min(n_active + 1, total_num_substeps)
            for c in range(n_reset):
                for iq in range(numflux):
                    pQ_aver[c, iq] = 0.0
                    for s in range(numsol):
                        mQ_aver[c, iq, s] = 0.0
                for s in range(numsol):
                    mR_aver[c, s] = 0.0

            # Initial condition for newest characteristic
            if iT_substep > 0:
                idx = total_num_substeps - iT_substep
                sT_start[idx] = sT_init_fullstep[iT_prev_fullstep]
                for s in range(numsol):
                    mT_start[idx, s] = mT_init_fullstep[iT_prev_fullstep, s]

            # ---- Fused Runge-Kutta integration over characteristics ----
            _rk_substep_fused(
                sT_start,
                mT_start,
                mT_temp,
                pQ_temp,
                mQ_temp,
                mR_temp,
                pQ_aver,
                mQ_aver,
                mR_aver,
                stagefrac,
                rk_coeffs,
                nstage,
                iT_substep,
                n_active,
                dt_substep,
                jt_fullstep_at,
                jt_substep_at,
                STcum_topbot_start,
                Q_fullstep,
                alpha_fullstep,
                k1_fullstep,
                C_eq_fullstep,
                J_fullstep,
                C_J_fullstep,
                weights_fullstep,
                SAS_args,
                P_list,
                grad_precalc,
                component_type,
                component_index_list,
                args_index_list,
                numargs_list,
                numflux,
                numsol,
                gamma_tables,
                gamma_use_table,
                gamma_table_meta,
            )

            # Update cumulative storage along characteristics
            for c in range(total_num_substeps + 1):
                STcum_topbot_start[c, 0] = STcum_topbot_start[c, 1]
            for c in range(n_active):
                jt_c = jt_substep_at[c]
                if jt_c < total_num_substeps:
                    STcum_topbot_start[jt_c + 1, 1] = STcum_topbot_start[jt_c + 1, 0] + sT_start[c] * dt_substep
            STcum_topbot_start[0, 1] = STcum_topbot_start[0, 0] + sT_init_fullstep[iT_fullstep] * dt_substep

            # Update records
            _update_records(
                iT_fullstep,
                iT_substep,
                substep,
                total_num_substeps,
                n_active,
                n_substeps,
                dt_substep,
                dt,
                norm,
                jt_fullstep_at,
                jt_substep_at,
                Q_fullstep,
                alpha_fullstep,
                pQ_aver,
                mQ_aver,
                mR_aver,
                sT_start,
                mT_start,
                C_Q_fullstep,
                P_old_fullstep,
                pQ_outputstep,
                mQ_outputstep,
                mR_outputstep,
                sT_outputstep,
                mT_outputstep,
                output_these_fullsteps,
                num_output_fullsteps,
                numflux,
                numsol,
                max_age,
                timeseries_length,
                rec_sT,
                rec_mT,
                rec_pQ,
                rec_mQ,
                rec_mR,
            )

            iT_prev_fullstep = iT_fullstep

        if verbose and iT_fullstep % 10 == 0:
            print(" ...Done", iT_fullstep, "of", max_age)

    # Finalization: add contribution from water older than max_age
    for s in range(numsol):
        for iq in range(numflux):
            for jt in range(timeseries_length):
                if Q_fullstep[jt, iq] > 0:
                    C_Q_fullstep[jt, iq, s] += alpha_fullstep[jt, iq, s] * C_old[s] * P_old_fullstep[jt, iq]

    # Mass balances are computed by the caller (`solve`) as a post-pass.

    if verbose:
        print(" ...Finished...")

    return C_Q_fullstep


@_maybe_jit_parallel
def _rk_substep_fused(
    sT_start,
    mT_start,
    mT_temp,
    pQ_temp,
    mQ_temp,
    mR_temp,
    pQ_aver,
    mQ_aver,
    mR_aver,
    stagefrac,
    rk_coeffs,
    nstage,
    iT_substep,
    n_active,
    dt_substep,
    jt_fullstep_at,
    jt_substep_at,
    STcum_topbot_start,
    Q_fullstep,
    alpha_fullstep,
    k1_fullstep,
    C_eq_fullstep,
    J_fullstep,
    C_J_fullstep,
    weights_fullstep,
    SAS_args,
    P_list,
    grad_precalc,
    component_type,
    component_index_list,
    args_index_list,
    numargs_list,
    numflux,
    numsol,
    gamma_tables,
    gamma_use_table,
    gamma_table_meta,
):
    """Fused RK substep: all stages for each characteristic in one sweep.

    Characteristics are independent within a substep (STcum_topbot_start is
    frozen), so the c loop is embarrassingly parallel.
    """
    for c in prange(n_active):
        jt_sub = jt_substep_at[c]
        jt_c = jt_fullstep_at[c]
        top0 = STcum_topbot_start[jt_sub, 0]
        top1 = STcum_topbot_start[jt_sub + 1, 1]
        sT_c = sT_start[c]
        for s in range(numsol):
            mT_temp[c, s] = mT_start[c, s]

        for rk in range(nstage):
            sf = stagefrac[rk]

            # ---- pQ from SAS functions at current stage state ----
            if sT_c > 0.0 and not (iT_substep == 0 and sf == 0.0):
                if iT_substep == 0:
                    top = 0.0
                else:
                    top = top0 * (1.0 - sf) + top1 * sf
                bot = top + sT_c * dt_substep
                for iq in range(numflux):
                    PQt = 0.0
                    PQb = 0.0
                    for ic in range(int(component_index_list[iq]), int(component_index_list[iq + 1])):
                        ctype = int(component_type[ic])
                        ai = int(args_index_list[ic])
                        nargs = int(numargs_list[ic])
                        w = float(weights_fullstep[jt_c, ic])
                        if ctype == -1:
                            PQt += w * _piecewise_linear_cdf(top, SAS_args, P_list, grad_precalc, ai, jt_c, nargs)
                            PQb += w * _piecewise_linear_cdf(bot, SAS_args, P_list, grad_precalc, ai, jt_c, nargs)
                        elif ctype == 1:
                            loc = float(SAS_args[ai, jt_c])
                            scale = float(SAS_args[ai + 1, jt_c])
                            if gamma_use_table[ic]:
                                g_xmax = gamma_table_meta[ic, 0]
                                g_dy = gamma_table_meta[ic, 1]
                                g_aexp = gamma_table_meta[ic, 2]
                                PQt += w * _gamma_cdf_table(top, loc, scale, gamma_tables[ic], g_xmax, g_dy, g_aexp)
                                PQb += w * _gamma_cdf_table(bot, loc, scale, gamma_tables[ic], g_xmax, g_dy, g_aexp)
                            else:
                                a = float(SAS_args[ai + 2, jt_c])
                                PQt += w * _gamma_cdf(top, loc, scale, a)
                                PQb += w * _gamma_cdf(bot, loc, scale, a)
                        elif ctype == 2:
                            loc = float(SAS_args[ai, jt_c])
                            scale = float(SAS_args[ai + 1, jt_c])
                            a = float(SAS_args[ai + 2, jt_c])
                            b = float(SAS_args[ai + 3, jt_c])
                            PQt += w * _beta_cdf(top, loc, scale, a, b)
                            PQb += w * _beta_cdf(bot, loc, scale, a, b)
                        elif ctype == 3:
                            loc = float(SAS_args[ai, jt_c])
                            scale = float(SAS_args[ai + 1, jt_c])
                            a = float(SAS_args[ai + 2, jt_c])
                            b = float(SAS_args[ai + 3, jt_c])
                            PQt += w * _kumaraswamy_cdf(top, loc, scale, a, b)
                            PQb += w * _kumaraswamy_cdf(bot, loc, scale, a, b)
                    pQ_temp[c, iq] = (PQb - PQt) / dt_substep
            else:
                for iq in range(numflux):
                    pQ_temp[c, iq] = 0.0

            # ---- mQ, mR from current stage state ----
            if sT_c > 0.0:
                inv_sT = 1.0 / sT_c
                for iq in range(numflux):
                    Q_pQ_inv = Q_fullstep[jt_c, iq] * pQ_temp[c, iq] * inv_sT
                    for s in range(numsol):
                        mQ_temp[c, iq, s] = mT_temp[c, s] * alpha_fullstep[jt_c, iq, s] * Q_pQ_inv
            else:
                for iq in range(numflux):
                    for s in range(numsol):
                        mQ_temp[c, iq, s] = 0.0
            for s in range(numsol):
                if k1_fullstep[jt_c, s] > 0.0:
                    mR_temp[c, s] = k1_fullstep[jt_c, s] * (C_eq_fullstep[jt_c, s] * sT_c - mT_temp[c, s])
                else:
                    mR_temp[c, s] = 0.0

            # ---- accumulate RK averages ----
            co = rk_coeffs[rk]
            for iq in range(numflux):
                pQ_aver[c, iq] += co * pQ_temp[c, iq]
                for s in range(numsol):
                    mQ_aver[c, iq, s] += co * mQ_temp[c, iq, s]
            for s in range(numsol):
                mR_aver[c, s] += co * mR_temp[c, s]

            # ---- propagate: next stage state, or final update from averages ----
            last = rk == nstage - 1
            sf_next = stagefrac[rk + 1]
            dt_num = dt_substep * sf_next
            sT_c = sT_start[c]
            if last:
                for s in range(numsol):
                    mT_temp[c, s] = mT_start[c, s] + mR_aver[c, s] * dt_num
            else:
                for s in range(numsol):
                    mT_temp[c, s] = mT_start[c, s] + mR_temp[c, s] * dt_num
            if iT_substep == 0:
                sT_c += J_fullstep[jt_c] * sf_next
                for s in range(numsol):
                    mT_temp[c, s] += J_fullstep[jt_c] * C_J_fullstep[jt_c, s] * sf_next
            total_Q_pQ = 0.0
            if last:
                for iq in range(numflux):
                    total_Q_pQ += Q_fullstep[jt_c, iq] * pQ_aver[c, iq]
            else:
                for iq in range(numflux):
                    total_Q_pQ += Q_fullstep[jt_c, iq] * pQ_temp[c, iq]
            sT_c -= total_Q_pQ * dt_num
            if sT_c < 0.0:
                sT_c = 0.0
            if last:
                for s in range(numsol):
                    total_mQ = 0.0
                    for iq in range(numflux):
                        total_mQ += mQ_aver[c, iq, s]
                    mT_temp[c, s] -= total_mQ * dt_num
            else:
                for s in range(numsol):
                    total_mQ = 0.0
                    for iq in range(numflux):
                        total_mQ += mQ_temp[c, iq, s]
                    mT_temp[c, s] -= total_mQ * dt_num

        # ---- commit ----
        sT_start[c] = sT_c
        for s in range(numsol):
            mT_start[c, s] = mT_temp[c, s]


@_maybe_jit
def _update_records(
    iT_fullstep,
    iT_substep,
    substep,
    total_num_substeps,
    n_active,
    n_substeps,
    dt_substep,
    dt,
    norm,
    jt_fullstep_at,
    jt_substep_at,
    Q_fullstep,
    alpha_fullstep,
    pQ_aver,
    mQ_aver,
    mR_aver,
    sT_start,
    mT_start,
    C_Q_fullstep,
    P_old_fullstep,
    pQ_outputstep,
    mQ_outputstep,
    mR_outputstep,
    sT_outputstep,
    mT_outputstep,
    output_these_fullsteps,
    num_output_fullsteps,
    numflux,
    numsol,
    max_age,
    timeseries_length,
    rec_sT,
    rec_mT,
    rec_pQ,
    rec_mQ,
    rec_mR,
):
    """Accumulate results into output arrays (age-first layout, write-only)."""

    # Update output concentration (iterate over active characteristics;
    # inactive ones have mQ_aver == 0 and contribute nothing)
    for c in range(n_active):
        jt_fullstep = jt_fullstep_at[c]
        for iq in range(numflux):
            if Q_fullstep[jt_fullstep, iq] > 0.0:
                for s in range(numsol):
                    C_Q_fullstep[jt_fullstep, iq, s] += (
                        mQ_aver[c, iq, s] * dt_substep / Q_fullstep[jt_fullstep, iq] / n_substeps
                    )

    # Update old water fraction
    for c in range(n_active):
        jt_c = jt_fullstep_at[c]
        for iq in range(numflux):
            P_old_fullstep[jt_c, iq] -= pQ_aver[c, iq] * dt_substep / n_substeps

    # Timestep-averaged transit time distribution
    if (rec_pQ or rec_mQ or rec_mR) and iT_fullstep < max_age - 1:
        for outputstep in range(num_output_fullsteps):
            jt_fullstep = output_these_fullsteps[outputstep]
            for jt_ws in range(n_substeps):
                if jt_ws < substep:
                    c = (total_num_substeps + jt_fullstep * n_substeps + jt_ws - iT_substep) % total_num_substeps
                    for iq in range(numflux):
                        if rec_pQ:
                            pQ_outputstep[iT_fullstep + 1, outputstep, iq] += pQ_aver[c, iq] * norm
                        if rec_mQ:
                            for s in range(numsol):
                                mQ_outputstep[iT_fullstep + 1, outputstep, iq, s] += mQ_aver[c, iq, s] * norm
                    if rec_mR:
                        for s in range(numsol):
                            mR_outputstep[iT_fullstep + 1, outputstep, s] += mR_aver[c, s] * norm

    if rec_pQ or rec_mQ or rec_mR:
        for outputstep in range(num_output_fullsteps):
            jt_fullstep = output_these_fullsteps[outputstep]
            for jt_ws in range(n_substeps):
                if jt_ws >= substep:
                    c = (total_num_substeps + jt_fullstep * n_substeps + jt_ws - iT_substep) % total_num_substeps
                    for iq in range(numflux):
                        if rec_pQ:
                            pQ_outputstep[iT_fullstep, outputstep, iq] += pQ_aver[c, iq] * norm
                        if rec_mQ:
                            for s in range(numsol):
                                mQ_outputstep[iT_fullstep, outputstep, iq, s] += mQ_aver[c, iq, s] * norm
                    if rec_mR:
                        for s in range(numsol):
                            mR_outputstep[iT_fullstep, outputstep, s] += mR_aver[c, s] * norm

    # Extract substep state at output timesteps
    if rec_sT or rec_mT:
        for outputstep in range(num_output_fullsteps):
            jt_fullstep = output_these_fullsteps[outputstep]
            jt_ws = n_substeps - 1
            c = (total_num_substeps + jt_fullstep * n_substeps + jt_ws - iT_substep) % total_num_substeps
            if rec_sT:
                sT_outputstep[iT_fullstep, outputstep + 1] += sT_start[c] / n_substeps
            if rec_mT:
                for s in range(numsol):
                    mT_outputstep[iT_fullstep, outputstep + 1, s] += mT_start[c, s] / n_substeps


@_maybe_jit
def _calculate_water_balance(
    max_age,
    num_output_fullsteps,
    output_these_fullsteps,
    J_fullstep,
    Q_fullstep,
    sT_outputstep,
    pQ_outputstep,
    WaterBalance_outputstep,
    dt,
    numflux,
):
    """Compute water balance residuals (age-first layout)."""
    for iT_fullstep in range(max_age):
        for outputstep in range(num_output_fullsteps):
            jt_fullstep = output_these_fullsteps[outputstep]
            if iT_fullstep == 0:
                wb = J_fullstep[jt_fullstep] - sT_outputstep[iT_fullstep, outputstep + 1]
            else:
                wb = sT_outputstep[iT_fullstep - 1, outputstep] - sT_outputstep[iT_fullstep, outputstep + 1]
            for iq in range(numflux):
                wb -= Q_fullstep[jt_fullstep, iq] * pQ_outputstep[iT_fullstep, outputstep, iq] * dt
            WaterBalance_outputstep[iT_fullstep, outputstep] = wb


@_maybe_jit
def _calculate_solute_balance(
    max_age,
    num_output_fullsteps,
    output_these_fullsteps,
    J_fullstep,
    C_J_fullstep,
    mT_outputstep,
    mQ_outputstep,
    mR_outputstep,
    SoluteBalance_outputstep,
    dt,
    numflux,
    numsol,
):
    """Compute solute mass balance residuals (age-first layout)."""
    for iT_fullstep in range(max_age):
        for outputstep in range(num_output_fullsteps):
            jt_fullstep = output_these_fullsteps[outputstep]
            for s in range(numsol):
                if iT_fullstep == 0:
                    sb = (
                        C_J_fullstep[jt_fullstep, s] * J_fullstep[jt_fullstep]
                        - mT_outputstep[iT_fullstep, outputstep + 1, s]
                    )
                else:
                    sb = mT_outputstep[iT_fullstep - 1, outputstep, s] - mT_outputstep[iT_fullstep, outputstep + 1, s]
                for iq in range(numflux):
                    sb -= mQ_outputstep[iT_fullstep, outputstep, iq, s] * dt
                sb += mR_outputstep[iT_fullstep, outputstep, s] * dt
                SoluteBalance_outputstep[iT_fullstep, outputstep, s] = sb


# ---------------------------------------------------------------------------
# Public API — matches the f2py interface
# ---------------------------------------------------------------------------


def solve(
    J_fullstep,
    Q_fullstep,
    SAS_args,
    P_list,
    weights_fullstep,
    sT_init_fullstep,
    dt,
    verbose,
    debug,
    warning,
    jacobian,
    mT_init_fullstep,
    C_J_fullstep,
    alpha_fullstep,
    k1_fullstep,
    C_eq_fullstep,
    C_old,
    n_substeps,
    component_type,
    numcomponent_list,
    numargs_list,
    output_these_fullsteps,
    num_scheme,
    numflux,
    numsol,
    max_age,
    timeseries_length,
    num_output_fullsteps,
    numcomponent_total,
    numargs_total,
    *,
    record_to=None,
    record_arrays=None,
    record_dtype="float64",
):
    """SAS transport solver — drop-in replacement for the Fortran solvesas.

    Positional parameters and the return tuple match the f2py interface.
    The recorded state arrays are returned in age-first layout
    ``(max_age, n_recorded_steps, ...)``. The Jacobian slots of the return
    tuple (``dsTdSj``, ``dmTdSj``, ``dCdSj``) are singleton placeholders —
    the Numba solver does not compute Jacobians.

    Keyword parameters
    ------------------
    record_to : str or None
        Directory in which to allocate the recorded state arrays as
        disk-backed ``.npy`` memmaps (created if needed). ``None`` (default)
        allocates in RAM.
    record_arrays : iterable of str or None
        Which arrays to record: subset of {"sT", "pQ", "mT", "mQ", "mR",
        "water_balance", "solute_balance"}. ``None`` records all.
        Unrecorded arrays are returned as singleton placeholders.
    record_dtype : {"float64", "float32"}
        Storage dtype of the recorded arrays. Computation is float64 always.
    """
    # Ensure contiguous 2D float64 arrays.
    # SAS_args and P_list may arrive as 3D from _create_sas_lookup;
    # the Fortran f2py binding auto-squeezed them, so we reshape here.
    J_fullstep = np.ascontiguousarray(J_fullstep, dtype=np.float64)
    Q_fullstep = np.ascontiguousarray(Q_fullstep, dtype=np.float64)
    SAS_args = np.ascontiguousarray(SAS_args, dtype=np.float64).ravel().reshape(numargs_total, timeseries_length)
    P_list = np.ascontiguousarray(P_list, dtype=np.float64).ravel().reshape(numargs_total, timeseries_length)
    weights_fullstep = np.ascontiguousarray(weights_fullstep, dtype=np.float64)
    sT_init_fullstep = np.ascontiguousarray(sT_init_fullstep, dtype=np.float64)
    mT_init_fullstep = np.ascontiguousarray(mT_init_fullstep, dtype=np.float64)
    C_J_fullstep = np.ascontiguousarray(C_J_fullstep, dtype=np.float64)
    alpha_fullstep = np.ascontiguousarray(alpha_fullstep, dtype=np.float64)
    k1_fullstep = np.ascontiguousarray(k1_fullstep, dtype=np.float64)
    C_eq_fullstep = np.ascontiguousarray(C_eq_fullstep, dtype=np.float64)
    C_old = np.ascontiguousarray(C_old, dtype=np.float64)
    component_type = np.ascontiguousarray(component_type, dtype=np.int64)
    numcomponent_list = np.ascontiguousarray(numcomponent_list, dtype=np.int64)
    numargs_list = np.ascontiguousarray(numargs_list, dtype=np.int64)
    output_these_fullsteps = np.ascontiguousarray(output_these_fullsteps, dtype=np.int64)

    # --- Allocate the recorded state arrays (age-first layout) ---
    if record_arrays is None:
        rec = RECORDABLE_ARRAYS
    else:
        rec = frozenset(record_arrays)
    if record_dtype not in ("float64", "float32"):
        raise ValueError(f"record_dtype must be 'float64' or 'float32', got {record_dtype!r}")
    out_dtype = np.float32 if record_dtype == "float32" else np.float64
    A = int(max_age)
    T_out = int(num_output_fullsteps)
    q = int(numflux)
    ns = int(numsol)
    # Balance arrays can only be computed from recorded ingredients
    wb_on = {"water_balance", "sT", "pQ"} <= rec
    sb_on = {"solute_balance", "mT", "mQ", "mR"} <= rec and ns > 0

    if record_to is not None:
        record_to = os.fspath(record_to)
        os.makedirs(record_to, exist_ok=True)

    def _alloc(name, shape, on):
        if not on:
            return np.zeros((1,) * len(shape), dtype=out_dtype)
        if record_to is None:
            return np.zeros(shape, dtype=out_dtype)
        return np.lib.format.open_memmap(
            os.path.join(record_to, name + ".npy"), mode="w+", dtype=out_dtype, shape=shape
        )

    sT_out = _alloc("sT", (A, T_out + 1), "sT" in rec)
    mT_out = _alloc("mT", (A, T_out + 1, ns), "mT" in rec)
    pQ_out = _alloc("pQ", (A, T_out, q), "pQ" in rec)
    mQ_out = _alloc("mQ", (A, T_out, q, ns), "mQ" in rec)
    mR_out = _alloc("mR", (A, T_out, ns), "mR" in rec)
    WB_out = _alloc("water_balance", (A, T_out), wb_on)
    SB_out = _alloc("solute_balance", (A, T_out, ns), sb_on)

    # Initial conditions: column 0 of the age-first sT/mT arrays
    if "sT" in rec:
        sT_out[:, 0] = sT_init_fullstep[:A]
    if "mT" in rec:
        mT_out[:, 0, :] = mT_init_fullstep[:A, :]

    C_Q_fullstep = _solve_core(
        J_fullstep,
        Q_fullstep,
        SAS_args,
        P_list,
        weights_fullstep,
        sT_init_fullstep,
        float(dt),
        bool(verbose),
        mT_init_fullstep,
        C_J_fullstep,
        alpha_fullstep,
        k1_fullstep,
        C_eq_fullstep,
        C_old,
        int(n_substeps),
        component_type,
        numcomponent_list,
        numargs_list,
        output_these_fullsteps,
        int(num_scheme),
        int(numflux),
        int(numsol),
        int(max_age),
        int(timeseries_length),
        int(num_output_fullsteps),
        int(numcomponent_total),
        int(numargs_total),
        sT_out,
        mT_out,
        pQ_out,
        mQ_out,
        mR_out,
        "sT" in rec,
        "mT" in rec,
        "pQ" in rec,
        "mQ" in rec,
        "mR" in rec,
    )

    # --- Mass balance post-passes (read the recorded state sequentially) ---
    if wb_on:
        _calculate_water_balance(
            int(max_age),
            int(num_output_fullsteps),
            output_these_fullsteps,
            J_fullstep,
            Q_fullstep,
            sT_out,
            pQ_out,
            WB_out,
            float(dt),
            int(numflux),
        )
    if sb_on:
        _calculate_solute_balance(
            int(max_age),
            int(num_output_fullsteps),
            output_these_fullsteps,
            J_fullstep,
            C_J_fullstep,
            mT_out,
            mQ_out,
            mR_out,
            SB_out,
            float(dt),
            int(numflux),
            int(numsol),
        )

    # For disk-backed outputs: flush writes, then hand back read-only views
    if record_to is not None:

        def _reopen(arr, name, on):
            if not on:
                return arr
            arr.flush()
            return np.load(os.path.join(record_to, name + ".npy"), mmap_mode="r")

        sT_out = _reopen(sT_out, "sT", "sT" in rec)
        mT_out = _reopen(mT_out, "mT", "mT" in rec)
        pQ_out = _reopen(pQ_out, "pQ", "pQ" in rec)
        mQ_out = _reopen(mQ_out, "mQ", "mQ" in rec)
        mR_out = _reopen(mR_out, "mR", "mR" in rec)
        WB_out = _reopen(WB_out, "water_balance", wb_on)
        SB_out = _reopen(SB_out, "solute_balance", sb_on)

    # Jacobian slots: singleton placeholders (not computed by this solver)
    ds_placeholder = np.zeros((1, 1, 1))
    dm_placeholder = np.zeros((1, 1, 1, 1))
    dC_placeholder = np.zeros((1, 1, 1, 1))

    return (
        sT_out,
        pQ_out,
        WB_out,
        mT_out,
        mQ_out,
        mR_out,
        C_Q_fullstep,
        ds_placeholder,
        dm_placeholder,
        dC_placeholder,
        SB_out,
    )


# ---------------------------------------------------------------------------
# Warm up the Numba parallel threading layer from Python at import time.
# Initializing it lazily from within jit-compiled code can segfault.
# ---------------------------------------------------------------------------
if _USE_NUMBA and os.environ.get("MESAS_PARALLEL", "1") != "0":
    # Initialise numba's threading layer from Python at import time.
    # Without this, the first invocation of a parallel=True function from
    # inside another njit function can segfault intermittently on some
    # platforms (observed with numba 0.61 on macOS arm64) due to lazy
    # threading-layer initialisation.
    @njit(parallel=True, cache=False)
    def _warmup_threads(n):
        acc = 0.0
        for i in prange(n):
            acc += i * 0.5
        return acc

    _warmup_threads(64)
