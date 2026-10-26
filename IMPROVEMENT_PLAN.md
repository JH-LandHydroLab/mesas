# MESAS Improvement Plan (post-refactor review)

Date: 2026-07-05. Branch: `stochastic` (after the 6-stage refactor recorded in
`REFACTORING_PROGRESS.md`).

This plan is the product of a systematic review of the codebase against the published
description of the software (Harman & Xu Fei, 2024, *mesas.py v1.0*, GMD 17:477,
doi:10.5194/gmd-17-477-2024) and the in-repo documentation. Four independent review
passes were run: (A) numerical correctness of the solver (line-by-line against the
retired Fortran reference `solve.f90` and the paper's equations, with numerical
cross-validation), (B) solver performance (with measured prototypes), (C) the
visualization library, and (D) API robustness / documentation accuracy / packaging.

Verification status labels: **CONFIRMED** = reproduced numerically or by unambiguous
logic; **SUSPECTED** = strong code-reading evidence, not yet reproduced.

Reference baseline: the full test suite currently passes except
`test/test_time.py::test_unsteady_uniform` and its `test_time2.py` twin, which fail on
a file path (bug A7 below), i.e. 1 failed / 123 passed / 4 perf tests deselected.
The Numba solver was cross-validated against the compiled Fortran solver on three
nontrivial configurations: Euler and RK2 agree to ≤7e-15 across all outputs; RK4
differs by ~1e-8 only because the Fortran RK4 weights were single-precision (A8) —
the port itself is faithful.

---

## A. Bugs and errors

### Critical

**A1. Gamma CDF lookup table saturates at x ≥ 40 — wrong results for shape ≳ 25.**
`mesas/sas/_solve_numba.py:273,296-298`. The lookup table introduced in commit
`d51756c` returns 1.0 for all `x ≥ 40`, on the claim that `gammainc(a, 40) ≈ 1` for
all practical `a`. That is false: gammainc(30,40)=0.957, gammainc(50,40)=0.070,
gammainc(100,40)≈1e-15. The table path is taken whenever the gamma shape is constant
in time (the common case), so Ω_Q silently saturates to 1 at S_T = loc + 40·scale.
CONFIRMED end-to-end: gamma(shape=50, scale=2) gives mean C_Q = 2.18 vs 6.60 exact.
*Fix:* size the table domain from the shape (x_max ≈ a + 12√a + 40), or bypass the
table whenever `gammainc(a, table_xmax)` is not within tolerance of 1.

**A2. Gamma lookup table inaccurate near x → 0 for shape < 1 (the typical hydrologic
case).** `_solve_numba.py:284-303`. The uniform grid (Δx = 0.002) cannot resolve the
integrable singularity of the gamma pdf at x→0 when a<1. CONFIRMED: max CDF error
0.072 at a=0.3; pQ obtained by differencing within the first table cell is off by up
to 94 % relative. Misallocates young-water TTD mass between adjacent age bins.
*Fix:* log-spaced or singularity-adapted grid near 0 (e.g. tabulate against x^a), with
an accuracy test against `scipy.special.gammainc` over shapes 0.05–100.

**A3. Parameter updates only reach timestep 0 — calibration machinery silently
broken.** `mesas/sas/specs.py:171,189-190`. `Component._sas_funs` is a list of N
distinct per-timestep function objects, but `update_from_parameter_list` /
`get_parameter_list` read and write only `sas_fun[0]`, while the solver consumes all
timesteps via `Component.argsS`. CONFIRMED: after `update_from_parameter_list(2*p)`,
`argsS[:,0]` is updated and `argsS[:,1:]` still holds the old values. Every
optimization workflow (`components_to_learn`, `mesas.me.recursive_split`) trains a
SAS function that applies only at t=0. This is fallout from the Stage-6 list-of-N
change. *Fix:* for time-invariant components store one function (fast path, see B5);
for time-varying ones, propagate parameter updates to all timesteps. Add a regression
test that runs the model after an update and checks outputs actually changed at t>0.

**A4. `Model`/`ModelResult` cannot be pickled or deep-copied (RecursionError).**
`mesas/sas/model.py:219-225`. `ModelResult.__getattr__` references `self._data`;
during unpickling/copy protocol probing `_data` does not exist yet, so the lookup
re-enters `__getattr__` infinitely. CONFIRMED for `pickle.dumps` and `copy.deepcopy`
of both `Model` and `ModelResult`. This blocks all multiprocessing-based calibration
and uncertainty workflows. *Fix:* raise `AttributeError` for missing dunder/underscore
names in `__getattr__` and add `__getstate__`/`__setstate__`; test round-trip pickle
of a run model.

**A5. `mesas.me.recursive_split` default mode calls a method that doesn't exist.**
`mesas/me/recursive_split.py:343` calls `Model.trim_unused_ST()`, which is defined
nowhere; the default `search_mode="leftfirst"` hits it at the first accepted
subdivision. CONFIRMED. *Fix:* implement `trim_unused_ST` on `Model` (drop piecewise
segments whose ST range is never visited) or remove the call; add a `run()` test for
the leftfirst path.

### Moderate

**A6. `sT_init` longer than the timeseries bypasses the `max_age` guard → garbage
output.** `mesas/sas/model.py:356-363`. The `max_age > timeseries_length` check runs
*before* `max_age = len(sT_init)` unconditionally overrides it (also silently ignoring
an explicit user `max_age`). CONFIRMED: N=20 with `sT_init` of length 40 runs to
completion; in the solver, negative indices wrap and characteristics recycle,
producing garbage with no error. *Fix:* re-validate after the override; error on
conflicting explicit settings.

**A7. Two analytical benchmark tests can never pass (path bug).**
`test/conftest.py:6-10` chdirs tests to the repo root, but `test/test_time.py:421` and
`test/test_time2.py:556` read `'unsteady_data.csv'` without the `test/` prefix.
CONFIRMED — this is the only current suite failure, and it means the unsteady-uniform
analytical benchmark in those files is not actually exercised. *Fix:* resolve data
paths relative to `__file__`.

**A8. `Model.set_sas_fun` always raises** (`model.py:517` assigns to the getter-only
`Component.sas_fun` property). CONFIRMED. Fix or remove the public method.

**A9. Jacobian option returns silent zeros.** `_solve_numba.py:442-445` fills the
Jacobian outputs with placeholder zeros, yet `ModelOptions.jacobian=True` is accepted
and `Model.get_jacobian` post-processes them. Additionally `model.py:816` indexes the
flux axis with the wrong enumeration (`_comp2learn_fluxorder` index vs `_fluxorder`),
and `model.py:828` advances `param_offset` only by the last inner-loop
`n_breakpoints` (and `NameError`s if the first solflux has no observations).
CONFIRMED by inspection. *Fix:* raise `NotImplementedError` for `jacobian=True` now;
repair the index bugs if/when the analytical Jacobian is revived. (`mesas.me` works
because it defaults to numerical jacobians.)

**A10. `SAS_Spec.plot()` / `Component.plot()` broken.** `mesas/sas/specs.py:411` calls
`self.sas_fun.plot(...)` where `sas_fun` is a list. CONFIRMED
(`AttributeError: 'list' object has no attribute 'plot'`). One more missed caller from
the Stage-6 `sas_fun[0]` sweep. Fix together with C-track work.

**A11. Silent acceptance of invalid inputs.** All CONFIRMED:
- NaN in `J` or a solute column → NaN propagates through results with no warning
  (`inputs.rst` promises otherwise implicitly).
- Negative fluxes run silently.
- Non-monotonic piecewise `ST` (e.g. `[0, 5, 3]`) accepted at construction —
  `Piecewise.__init__` (`functions.py:345-353`) checks only length and `ST[0]>=0`;
  monotonicity lives only in the property setter; `sasspec.rst:210` promises an error.
  Repeated breakpoints (`[0,50,50,100]`) silently drop the CDF jump
  (`_solve_numba.py:337-341` grad-0 guard).
- Empty DataFrame → cryptic reshape ValueError deep in the solver.
*Fix:* a `Model.validate()` pass run at construction/`run()` (opt-out flag), with
specific messages naming the offending column/flux/indices.

**A12. Bad error messages for missing columns / typo'd options.** CONFIRMED:
- A string `k1`/`C_eq`/`alpha` value that isn't a data column falls into the
  scalar-multiply branch of `_get_array` (`model.py:615-619`) → `UFuncTypeError`.
- Typo'd `Model(...)` option kwargs are silently discarded (`model.py:322`), while
  `ModelOptions.update()` raises — inconsistent.
- Solute named in `solute_parameters` with no data column → bare `KeyError` at run
  time instead of a construction-time check.
- `model.sas_specs = <raw dict>` stores the dict unparsed (`model.py:498-502`) and
  breaks `.run()`; the constructor path parses. Make the setter parse.
- `ModelOptions.from_dict` docstring says unknown keys warn; code raises.

### Minor

**A13.** `_betain` 1000-iteration cap silently returns an unconverged fallback for
very large beta `b` (SUSPECTED, accuracy verified fine for p,q ≤ 50) — add a
convergence flag or raise.
**A14.** `SAS_Spec.make_spec_ts` pads `ST` rows with a *decreasing* sentinel sequence
(`specs.py:116`) consumed by plotting code — plot artifacts only; fix with C-track.
**A15.** `mT_init` given as a column name crashes when `max_age != timeseries_length`
(`model.py:615-630`); partial `alpha` dicts KeyError for missing fluxes (SUSPECTED,
crash not corruption).
**A16.** `Model` mutates the caller's `sas_specs` dict; `data_df` reassignment leaves
stale `max_age`/interpolators (`model.py:465-473`).
**A17.** `_get_result(timestep=...)` indexes the output-step axis, which equals the
model timestep only when `record_state=True` — document/validate.
**A18.** `mesas/utils/vis.py:82` uses `np.NaN` → hard break on NumPy ≥ 2.0 (which
`pyproject.toml` currently permits). One-character fix; part of C1.

## B. Solver acceleration (all measured on Apple Silicon, N=2000 benchmark: gamma+piecewise SAS, 1 solute, RK4)

Baseline: raw `solve()` = 0.605 s; Python-side marshalling is only ~2 % (not worth
optimizing). SAS CDF evaluation is ~33 % of runtime; the rest is loop scaffolding.
Prototypes and the correctness sweep live in the session scratchpad; results:

**B1. Wetted-front skip — 1.70× speedup, bit-identical.** When `sT_init`/`mT_init`
are all zero (the default), the trailing "not yet wetted" characteristic slots are
identically zero, yet the solver still evaluates CDFs and sweeps them at every age
step. Restricting hot loops to `n_active = total_num_substeps - iT_substep` (gated by
a one-time zero-init check, full range otherwise) measured 0.605 → 0.353 s with max
abs diff exactly 0.0. Zero numerical risk. Subtleties: keep the RK-average reset
covering the just-retired slot and the per-substep re-init write.

**B2. Fused per-characteristic RK substep with `prange` — 3.8× cumulative at N=2000
(5.1× at N=5000, 1.5× at N=500), diffs ≤ 1e-14.** Characteristics are independent
within a substep (the only coupling, `STcum_topbot_start`, is frozen during the
substep). Fusing all RK stages + flux + averaging + state-commit into one
per-characteristic sweep and parallelizing with `numba.prange` gives the headline win.
**Landmine (reproduced):** numba 0.61.2 on macOS arm64 segfaults intermittently when a
`parallel=True` function is first invoked from inside another njit function (lazy
threading-layer init). Mandatory mitigation: call a trivial `@njit(parallel=True)`
warmup from Python at module import. With that, `parallel=True, fastmath=True,
cache=True` was stable across a 12-run stress matrix and an 11-config correctness
sweep (Euler/RK2/RK4, substeps 1–3, all SAS types, 0–2 solutes, nonzero `sT_init`,
`record_state=True`, `max_age < N`; worst rel diff 3.3e-15). Expose a Model option /
env var to fall back to serial (process-level parallel calibration should set
`NUMBA_NUM_THREADS`).

**B3. Loop restructuring / allocation hoisting — ~1.03×, essentially free.** Hoist
per-RK-stage temporaries out of `_calculate_pQ`, iterate the `C_Q_fullstep` record
loop via the existing characteristic map (drops a per-step modulo), fuse mQ zero+fill
in memory order. Fold into the B2 rewrite.

**B4. True `numsol=0` path.** `model.py:692` passes `numsol=max(_numsol,1)`, so
water-only runs execute all solute machinery with a dummy solute. Passing 0 (numba
handles zero-size arrays) is worth an estimated 15–30 % for no-solute runs. Requires
checking result-unpacking guards in `run()`.

**B5. Time-invariant component fast path (calibration hot loop).**
`Component.__init__` builds N per-timestep `Piecewise` objects and `make_spec_ts`
rebuilds 2N scipy `interp1d` objects on every parameter update (`specs.py:80-106,328`)
even when nothing varies in time. Detect time-invariance, build one function, and
broadcast in `argsS`. This also simplifies the A3 fix and speeds up every optimizer
iteration.

**B6. Beta CDF lookup table** (constant-parameter gating like gamma, est. 2–5× on the
CDF share for beta users) — do only after A1/A2 establish an accuracy-tested table
design. Kumaraswamy is closed-form and needs nothing.

Not recommended: float32 accumulators (mass balance is a core output); RK stage CDF
reuse (≤ 8 % serial, breaks time-varying gating); caching the gamma table across runs
(shape changes during calibration).

## C. Visualization library

### C1. Fix what's broken (P0)

- `specs.py:411` `Component.plot` → `self.sas_fun[i].plot(...)` with `i=0` default;
  `SAS_Spec.plot` gains the same and optionally draws the blended CDF (A10/A14).
- `vis.py:82` `np.NaN` → `np.nan` (NumPy 2 break, A18).
- Mutable default `artists_dict=OrderedDict()` in seven signatures
  (`vis.py:39,191,206,221,235,250,273`) — CONFIRMED cross-figure artist leakage;
  replace with `None` sentinel.
- No guard for the default `record_state=False`: `plot_transport_column` dies with a
  bare IndexError (`vis.py:48-51`). Add an upfront shape check with the actionable
  message "re-run with record_state=True".
- `make_transport_column_animation` re-creates all 5 axes and every Rectangle patch
  per frame (`vis.py:310-313`) — CONFIRMED leak (5 → 15 axes after 2 frames),
  quadratic in frames. Thread `do_init=False` through and update artists in place.
- Remove module-level `np.seterr(...)` (`vis.py:21`) — mutates global numpy state on
  import; `fig.set_tight_layout` deprecation; dead `_MT` variable.

### C2. New plotting module (P1)

Create `mesas/plotting.py` (functions take a `Model`, accept/return `Axes`, never call
`plt.show()`), plus a thin `model.plot` accessor (`ModelPlotter`) for discoverability:

- `plot_timeseries(model, flux, sol, obs=True)` — inputs/outputs, observations from
  `solute_parameters[sol]['observations']` overlaid (the calibration view).
- `plot_sas(model, flux, i=None, kind="cdf"|"pdf", envelope=True)` — single timestep
  or median + 5–95 % envelope across timesteps for time-varying specs.
- `plot_ttd(model, flux, times, cumulative=False, logx=True)` — p_Q/P_Q snapshots,
  marking the max-resolved-age / C_old cutoff.
- `plot_ttd_heatmap(model, flux, percentiles=(10,50,90))` — pcolormesh of pQ over
  (time, age) with transit-time percentile overlays.
- `plot_storage(model, by_age=...)` — S_T evolution and age-cohort partitions.
- `plot_balance(model, sol=None)` — water/solute balance residuals vs time (QC).

Supporting accessor: add `Model.get_PQ(flux)` (cumulative TTD = cumsum(pQ)·dt);
only `get_pQ` exists today.

### C3. Consolidate the transport column (P2)

Keep `plot_transport_column` + animation as the flagship pedagogy plot, but redraw the
column as a single `PolyCollection`/`pcolormesh` instead of per-age Rectangles
(~100× fewer artists), and delete the diverged fork in
`examples/benchmark/animate_benchmark.py` (contains an undefined global `bm`; import
from the library instead). Deprecate the four near-identical
`plot_influx/outflux/influx_conc/outflux_conc` cursor functions into one
`plot_timeseries(..., cursor=i)` with warning-emitting aliases.

### C4. Docs + tests (P3)

`doc/visualization.rst` gallery page (index.rst advertises `mesas.utils.vis` but no
page demonstrates it); Agg-backend smoke tests calling every public plot function on
the quickstart model (vis.py currently has zero test coverage — every C1 bug would
have been caught).

## D. Other improvements

### D1. Packaging and build

- **Drop meson-python.** Nothing is compiled anymore; the meson files only install
  `.py` sources. Move to hatchling (or setuptools): universal wheels, normal
  `pip install -e .`, no ninja/meson toolchain, simpler CI and RTD.
- Version: `v1.20230427` is stale and duplicated in `meson.build`; bump, single-source
  from package metadata, expose `mesas.__version__` (currently AttributeError).
- NumPy 2: pyproject allows it, CI pins `<2`, `vis.py` breaks on it. Fix A18, add a
  numpy≥2 CI job.
- `sklearn` is imported unconditionally by `mesas/me/recursive_split.py` but only
  listed in the `test` extra → ImportError on normal installs. Add an `estimation`
  extra + lazy import with a clear message.
- RTD config is at `doc/readthedocs.yml`; Read the Docs only reads root
  `.readthedocs.yaml` — move/rename or the config is silently ignored.
- Delete stale `requirements.txt` / `environment.yml` (python 3.9, no numba).
- CI: add ruff lint step, `sphinx-build -W` docs check, Python 3.13; remove conda once
  meson is gone; make coverage `fail_under` meaningful.
- pytest config duplicated (`pytest.ini` wins over `[tool.pytest.ini_options]`) —
  keep one.

### D2. Repo hygiene

- Retire `mesas/sas/solve.f90` and local build leftovers (`*.so`, `*.pyc`,
  `UNKNOWN.egg-info/`, `build/`, `sh.log`, `test/test.log`, `test/test_steady.pdf`,
  `doc/questions.html`, `examples/Archive.zip`); untrack `mesas/dev/troubleshoot.py`.
- `.gitignore`: add `.hypothesis/`, `.ruff_cache/`, `.pytest_cache/`, `.DS_Store`.
- Add `CITATION.cff` (Harman & Xu Fei 2024, doi:10.5194/gmd-17-477-2024) and the DOI
  to README; refresh the thin/stale README (Fortran claim at line 15, no quickstart,
  no badges); update LICENSE year.
- Move `REFACTORING_*.md` (and this file, once executed) to `doc/dev/`.
- Mark/segregate legacy test scripts (`test_time*.py`, `test_benchmark*.py`,
  `test_evapoconcentration.py`) from the curated suites; fix A7 while at it.
- `examples/quickstart.py:21` pandas FutureWarning (int column assigned 0.5);
  hyporheic/lower_hafren examples need runner scripts or READMEs.

### D3. Documentation corrections (all verified against current code)

- `doc/inputs.rst:14,26` — `from mesas.sas import Model` fails (`mesas/sas/__init__.py`
  is empty); either re-export or fix the snippet.
- `doc/sasspec.rst:97,171,223` — misspelled top-level key `"sas_specss"` in three JSON
  examples (copy-paste yields "No SAS specification found!"); `:226-227` invalid JSON
  (missing comma).
- `doc/config.rst:37-53` — invalid JSON in the flagship example.
- `doc/solspec.rst:87` — `set_solute_parameters("C1", C_old=22.5)` doesn't match the
  actual `(sol, params: dict)` signature; the "leaves the remainder unchanged" claim
  is false (setter rebuilds from scratch); `observations` default is `{}` not None.
- `doc/sasspec.rst` — loc/scale "defaults" that are actually required
  (`functions.py:674-684` raises KeyError; either add real defaults — preferred — or
  fix the docs); the monotonicity-error promise at `:210` (see A11); "strictly
  increasing" vs non-decreasing at `:212`.
- Stale Fortran references: `README.md:15`, `CONTRIBUTING.md:11,17,53`, and
  user-facing docstrings in `model.py:280,634-636,682`, `functions.py:248,542,612,804`.
- `doc/installation.rst:10-17` — clarify that PyPI/conda currently serve the old
  Fortran release.
- index.rst "Multiresolution" vs pyproject "Multiscale" naming mismatch.
- Result keys: stop storing duplicate camelCase keys in `ModelResult._data`
  (`model.py:764-780`) — attribute access to `WaterBalance` currently bypasses the
  deprecation warning; translate in `__getitem__` instead.

### D4. New user-facing features (ranked value/effort)

1. **First-class calibration API** (high/medium): `model.fit(...)` wrapper over the
   existing `get_residuals`/`mesas.me` machinery — blocked on A3, A4, A5; document
   `mesas.me` (no doc page mentions it today); replace `print`/globals with `logging`.
2. **Steady-state / spin-up helper** (high/low): `mesas.utils.spinup(model)` looping
   the record until `sT` converges, or analytic steady-state `sT_init` from mean
   fluxes.
3. **`Model.validate()` + input-prep utilities** (high/low): subsumes A11/A12;
   gap-filling/disaggregation helpers for tracer inputs (paper "future work").
4. **`result.to_xarray()`/`to_netcdf()`** (medium-high/low-medium): labeled
   (age, time, flux, solute) axes; eliminates the axis-order confusion the docs
   currently spend a section on.
5. **Linearized parameter uncertainty** (medium/medium): covariance from the numerical
   Jacobian; an emcee/bootstrap example once A4 lands.

---

## Implementation phases

Ordering principle: correctness first (silent-wrong-answer bugs), then the measured
performance rewrite (needs the correctness tests in place as its safety net), then
visualization, then packaging/docs/features. Every phase ends with the full suite
green (`conda run -n mesas11 python -m pytest test/`).

- **Phase 1 — critical correctness (A1, A2, A3, A4, A5, A6, A7, A8, A9)**
  Gamma-table accuracy fix with a dedicated accuracy test (shapes 0.05–100);
  parameter-update propagation (+ regression test); pickling; `trim_unused_ST`;
  sT_init/max_age validation; test-path fix; `set_sas_fun`; NotImplementedError for
  `jacobian=True`.
- **Phase 2 — solver acceleration (B1 → B3 → B2, then B4, B5)**
  Land the bit-identical front skip first, then the fused/parallel rewrite with the
  threading warmup, serial fallback option, nonzero-`sT_init` regression test, and a
  large-N smoke test. Wire the N=2000 benchmark numbers into
  `test_stage0_performance.py` thresholds.
- **Phase 3 — visualization (C1 → C2 → C3 → C4)** with Agg smoke tests from the start.
- **Phase 4 — robustness & API (A11, A12, A15, A16, A17, D4.3 validate())**.
- **Phase 5 — packaging & hygiene (D1, D2)**: hatchling switch, version/`__version__`,
  RTD config, CITATION.cff, prune dead files.
- **Phase 6 — documentation (D3, C4, mesas.me page)**.
- **Phase 7 — features (D4.1, D4.2, D4.4, D4.5)** as follow-on work.
