# MESAS Staged Refactoring Plan

This document outlines a staged approach to refactoring the `mesas` package. Each stage is designed to be independently testable and releasable, with clear essential and optional improvements.

The guiding philosophy is: **stabilise first, then modernise, then optimise**. Every stage adds tests before making changes, so regressions are caught early.

---

## Codebase Evaluation

A file-by-file assessment of the current state, covering code quality, architecture, and specific deficiencies.

### Python Code Quality

#### `mesas/sas/model.py` (581 lines) — The central class

**Structural issues:**
- The `Model.__init__` method does too much: it parses config files, validates options, parses SAS specs, and sets up solute parameters all in one 55-line constructor. This makes it hard to understand the initialization sequence and impossible to construct a Model in stages.
- The options system uses a mutable default dict (`_default_options`) that is shared across the class. The `options` setter silently mutates `self._options` via `.update()` rather than replacing it, which means setting `model.options = {...}` doesn't actually replace options — it merges. This is surprising behaviour.
- `_processinputs()` is a module-level function that doubles as both a JSON file loader and a dict passthrough. The name is vague, and it's called from multiple places where the acceptable types differ.
- Uses `assert` for input validation throughout (e.g., `assert flux in self.data_df.columns`, `assert isinstance(spec, dict)`). Asserts are stripped in optimized mode (`python -O`), making these checks disappear silently in production.

**API design issues:**
- Results are stored as a raw dict with string keys (`'sT'`, `'pQ'`, `'mQ'`, etc.) and accessed through multiple getter methods (`get_sT()`, `get_pQ()`, `get_mQ()`). Each getter repeats the same pattern of looking up a flux/solute index from a list and calling `_get_result()`. This is boilerplate-heavy and error-prone.
- The `run()` method has a 45-line body that manually unpacks every option and array into local variables, then passes ~30 positional arguments to the Fortran solver in a single line (line 421-426). This call is extremely fragile — a single argument in the wrong position silently produces wrong results.
- `copy_without_results()` references `self._components_to_learn` which may not exist if `components_to_learn` was never set (it's only set if passed via `kwargs`). This will raise `AttributeError`.
- `np.NaN` usage (line 559) is deprecated in modern NumPy — should be `np.nan`.

**Naming inconsistencies:**
- Mix of `camelCase` (`WaterBalance`, `SoluteBalance`), `snake_case` (`sas_specs`, `solute_parameters`), and `SCREAMING_CASE` in the results dict
- Private attributes use inconsistent underscore conventions: `_result`, `_data_df`, `_fluxorder`, `_numsol`, `_max_age` — some are clearly private internals, others are public-facing data hidden behind properties

#### `mesas/sas/specs.py` (235 lines) — SAS specification classes

**Issues:**
- `SAS_Spec.make_spec_ts()` builds N separate `interp1d` interpolators in a Python loop (one per timestep). For a 10,000-timestep model, this creates 10,000 scipy interpolator objects. This is the kind of operation that should be vectorised or at minimum lazy-evaluated.
- The `_NoneList` class at the bottom (lines 227-232) is a hack to avoid `None` checks — it returns `None` for any index. It's used to handle the case where `ST` or `P` aren't provided. A sentinel pattern or Optional type would be clearer.
- `Component.__init__` modifies its `spec` argument in place via `spec.pop()` (lines 155-168), which mutates the caller's data. The `.copy()` on line 151 is shallow, so nested dicts aren't protected.
- `Component.plot()` calls `self.sas_fun.plot()` but `sas_fun` is a property that returns a list, not a single function. This will fail at runtime.
- Variables named `repr` shadow the Python builtin in `__repr__` methods.

#### `mesas/sas/functions.py` (541 lines) — SAS function implementations

**Issues:**
- The `Continuous` class docstring says "Base function for SAS functions" — copy-pasted from `_SASFunctionBase`.
- `Continuous.__init__` has a complex branching structure (`if use=='builtin'` / `elif use=='scipy.stats'`) that handles two fundamentally different initialization paths in one method. The `builtin` path hardcodes distribution-specific parameter extraction (lines 438-449) with repeated `if func == 'gamma'` / `if func == 'beta'` / `if func == 'kumaraswamy'` blocks that don't share logic.
- The `kumaraswamy_gen` class is defined at module level between `Piecewise` and `Continuous` (lines 399-406), with an unnecessary duplicate `from scipy.stats import rv_continuous` import right above it. This class should either be in its own module or at the top of the file.
- `Continuous.func` setter has a typo: `selfs` instead of `self` (line 479). This method would never actually work if called.
- The `_SASFunctionBase.ST.setter` pattern uses Python descriptor protocol overriding, which is correct but unusual and poorly documented. A reader encountering `@_SASFunctionBase.ST.setter` in a subclass would need to understand property descriptor inheritance.
- Several `assert` statements used for validation (same issue as model.py).
- `Piecewise.__init__` docstring has `ST_max` and `ST_min` descriptions swapped (lines 194-195).

#### `mesas/me/recursive_split.py` (487 lines) — Model estimation

**Issues:**
- Uses **global mutable state** for verbosity (`VERBOSE`) and iteration counting (`ITERATION`). This makes the module non-thread-safe and means importing it has side effects.
- References `model.sas_blends` in multiple places (e.g., lines 95, 100, 207-208, 253, 338) but the `Model` class has no `sas_blends` attribute — it's called `sas_specs`. This means the entire `lookfor_new_components`, `increase_resolution_scanning`, and `increase_resolution_leftfirst` functions are **broken** and would raise `AttributeError` at runtime.
- `search_mode` error handling (line 71): `print('Never be here')` — this should be a `ValueError`.
- The `fit_model` function attaches arbitrary attributes to the model object (`new_model.x_prev`, `new_model.rmse`), violating encapsulation.
- `cross_validation_rmse` attaches `model.rmse_cv` — same problem.
- Variable named `iter` shadows the Python builtin (line 474).

#### `mesas/utils/vis.py` (279 lines) — Visualization

**Issues:**
- `np.seterr(divide='ignore', invalid='ignore')` at module import time (line 2) — this globally suppresses NumPy floating-point warnings for the entire process, not just this module. This masks real errors elsewhere.
- `plot_transport_column` is a 153-line function with 12 parameters, no docstring, and deeply nested matplotlib configuration. It mixes data computation (lines 17-63) with rendering (lines 65-151).
- References `model.sas_blends` in `plot_SAS_cumulative` (line 220, 232) — same broken attribute name as `recursive_split.py`.
- Uses mutable default arguments (`artists_dict=OrderedDict()`) which is a classic Python gotcha — all callers share the same default dict.

### Fortran Code: `solve.f90` (1597 lines)

**Architecture:**
- The entire solver is a single 1597-line subroutine (`solveSAS`) with 6 contained subroutines. This is a monolith — there's no separation between the numerical integration engine and the SAS-specific physics.
- Takes 30+ arguments, making the f2py interface extremely fragile. A misalignment between Python and Fortran argument ordering produces silent wrong results.
- ~200 lines of commented-out Jacobian/sensitivity code scattered throughout (lines 57-68, 146-158, 203-206, 237-243, 384-426, 509-592). This dead code constitutes roughly 12% of the file and obscures the actual logic.

**Numerical concerns:**
- `real(8)` is used throughout instead of `selected_real_kind`. While `real(8)` is commonly double precision, it's technically non-portable — the Fortran standard doesn't guarantee `real(8)` is 64-bit.
- The `do concurrent` constructs (lines 231, 330, 344, 352, 412, etc.) suggest intent for parallelisation, but `do concurrent` semantics are subtle and compiler support varies. Without testing, it's unclear these actually parallelise correctly.
- Memory allocation is entirely stack-based (consistent with `-fno-stack-arrays` compile flag), and array sizes are `timeseries_length*n_substeps` in multiple dimensions. For a 10,000-step model with 10 substeps, this means arrays of 100,000 elements per flux/solute, which can blow the stack.

**Readability:**
- Almost no comments explaining the algorithm. The only comments are section headers ("Start by declaring...", "Useful constants", "Loop over ages"). The actual numerical method — characteristic method with piecewise-linear SAS interpolation, RK4 integration — requires reading the GMD paper to understand.
- Variable names are cryptic even by Fortran standards: `jt_fullstep_at_`, `leftbreakpt_topbot`, `STcum_topbot_start`, `jt_is_which_substep`. The trailing underscore on `jt_fullstep_at_` is especially unusual.

### Build System

- **`setup.py`** uses `numpy.distutils.core.Extension` which was removed in NumPy 2.0. This means the package **cannot be built from source** with NumPy >= 2.0.
- **`pyproject.toml`** declares `setuptools` as the build backend, but the Fortran extension requires `numpy.distutils` — these are fundamentally incompatible. The `pyproject.toml` is effectively a lie about how the package is built.
- Classifiers list `"Programming Language :: Cython"` but there is no Cython anywhere in the project.
- Python version classifiers stop at 3.9, which reached EOL in October 2025. No 3.10, 3.11, 3.12, or 3.13 support is declared.
- The `recompile.sh` script (`rm -rf build && pip install .`) is the only documented way to rebuild after Fortran changes.
- `mesas/sas/__init__.py` contains Windows-specific DLL loading code using `ctypes` and `glob` to find `.libs/*.dll` — this is brittle and undocumented.

### Test Suite

- **`pytest.ini` points to `mesas/test`** but no such directory exists. The tests are in `test/` at the repo root. Pytest discovery is broken out of the box.
- `test_time.py` is the most substantial test file, but:
  - The `test_steady` function only runs 3 of the 10+ defined benchmarks (the `steady_benchmarks` dict has 3 entries; `other` has 7 more that are never tested).
  - The test function has `makefigure=False` as a parameter, mixing test logic with plotting code. When `makefigure=False`, significant portions of some tests are skipped (e.g., `test_part_multiple` and `test_reaction` are effectively no-ops without `makefigure=True`).
  - `err10` is referenced on line 363 but the code that computes it (lines 254-261) is commented out. The test would crash if `makefigure=True`.
  - No explicit assertions in `test_steady` — the function runs the model and computes errors but never asserts they're within tolerance (unlike `test_unsteady_uniform` which does `assert RMSE<1E-2`).
- `test/test_examples.py` exists but was not checked — likely tests the example configs.
- There are **no tests at all** for `recursive_split.py`, `vis.py`, or the `SAS_Spec`/`Component`/`Piecewise`/`Continuous` classes in isolation.
- Test data files (`MeSASInputHourly.csv` at ~1.5MB, `unsteady_data.csv` at ~500KB) are committed directly to the repo rather than generated or stored in a fixtures directory.

### Documentation

- `doc/conf.py` targets Sphinx with `sphinx_rtd_theme`, but:
  - `doc/readthedocs.yml` is inside the `doc/` directory rather than the repo root where ReadTheDocs expects it (`.readthedocs.yml` or `.readthedocs.yaml`).
  - The docs build likely fails on RTD due to the Fortran compilation requirement.
- Most `.rst` files in `doc/` contain minimal content — they're skeleton pages from initial setup.
- The module-level docstring in `model.py` (lines 1-7) says `"Text here"` — a placeholder that was never filled in.
- `functions.py` has the most complete module docstring, but it references the wrong class name at one point.

### Architectural Concerns

1. **Tight coupling between Python and Fortran**: The `Model.run()` method manually constructs ~30 arrays in specific shapes and orderings to match what the Fortran subroutine expects. There's no interface contract or schema — if either side changes, the other breaks silently.

2. **No separation of concerns in the solver**: The Fortran code handles water flux, solute transport, reactions, mass balance, RK4 integration, output recording, and state management all in one subroutine. Extracting or replacing any one piece requires understanding the whole.

3. **State mutation throughout**: `Model` methods freely mutate `self._data_df` (e.g., `run()` adds columns like `'C --> Q'` directly to the input DataFrame). The user's original data is modified as a side effect of running the model.

4. **Mixed responsibilities in `Component.__init__`**: This single constructor handles piecewise functions, builtin parametric distributions, and arbitrary scipy.stats distributions through a branching if/elif/else with 20+ lines of conditional logic. Each branch constructs objects differently.

5. **The `recursive_split` module is broken**: Multiple references to `model.sas_blends` (which doesn't exist) mean the entire model estimation module crashes at runtime. This suggests it hasn't been tested since the attribute was renamed to `sas_specs`.

---

## Stage 0: Establish a Testing Baseline (Foundation)

**Goal:** Before changing anything, lock down the current behaviour so that every subsequent stage can be validated against known-good results.

### Essential

- [ ] **Set up CI** (GitHub Actions) that runs tests on push/PR for Linux, macOS, and Windows
- [ ] **Write regression tests** that capture current numerical output for the two bundled examples (`lower_hafren`, `hyporheic`) — save reference outputs as fixtures
- [ ] **Write steady-state analytical benchmark tests** — formalise the existing `test_time.py` benchmarks into proper pytest cases with explicit pass/fail thresholds (RMS error tolerances)
- [ ] **Add mass-balance tests** — verify water balance and solute balance closures for a range of configurations
- [ ] **Add edge-case tests** — no solutes, single timestep, very short / very long timeseries, zero fluxes
- [ ] **Fix pytest discovery** — `pytest.ini` currently points to `mesas/test` but the actual test directory is `test/` at the repo root. Unify this.

### Optional

- [x] Add code coverage reporting (e.g. `pytest-cov`) and set a baseline percentage
- [x] Add a performance benchmark test that records wall-clock time, so future changes can be checked for performance regressions
- [x] Set up pre-commit hooks (ruff/black for formatting, mypy for type checking)

---

## Stage 1: Code Quality and Documentation (Readability)

**Goal:** Make the existing Python code understandable, well-documented, and consistently styled — without changing any behaviour.

### Essential

- [ ] **Add docstrings** to all public classes and methods in `model.py`, `specs.py`, `functions.py`, `recursive_split.py`, and `vis.py` — use NumPy-style docstrings
- [ ] **Add inline comments** to `model.py:run()` and `model.py:_create_sas_lookup()` explaining the data marshalling between Python and Fortran
- [ ] **Comment the Fortran code** — add block-level comments to `solve.f90` explaining each major section (initialisation, RK4 stepping, flux calculation, mass balance)
- [ ] **Add type hints** to all public Python APIs
- [ ] **Rename unclear variables** in Python code (e.g. single-letter names, abbreviations that aren't domain-standard)
- [ ] **Run a linter/formatter** (ruff + black) across the Python codebase and commit the result as a single formatting-only commit

### Optional

- [x] Remove dead code: the commented-out Jacobian stubs in `solve.f90`, unused imports, the `mesas/dev/` directory (or move to a `devtools/` folder outside the package)
- [x] Add a `CONTRIBUTING.md` with setup instructions for developers
- [x] Populate `__init__.py` files with convenient top-level imports (e.g. `from mesas import Model`)

### Testing

- [ ] All Stage 0 tests still pass with zero tolerance change
- [ ] Docstring coverage check (e.g. `interrogate`) passes at >90%

---

## Stage 2: Fix the Build System (Packaging)

**Goal:** Replace the broken `numpy.distutils` + `setup.py` build with a modern, working build pipeline.

### Context

`numpy.distutils` was deprecated in NumPy 1.24 and removed in NumPy 2.0. The current `setup.py` uses `numpy.distutils.core.Extension` which is already broken on newer NumPy. The `pyproject.toml` declares `setuptools` as the build backend but doesn't know about the Fortran extension. The classifiers also incorrectly say "Cython".

### Essential

- [ ] **Choose a build backend that supports Fortran**: either `meson-python` (recommended, well-supported by NumPy/SciPy ecosystem) or `scikit-build-core` (CMake-based)
- [ ] **Write a `meson.build`** (or `CMakeLists.txt`) that compiles `solve.f90` into a Python extension via f2py
- [ ] **Update `pyproject.toml`** to use the new build backend and remove the broken `setup.py`
- [ ] **Fix classifiers** — remove "Cython", add Python 3.10-3.12
- [ ] **Update `requires-python`** to `>=3.10` (3.8 and 3.9 are EOL)
- [ ] **Remove `recompile.sh`** — the build system should handle this
- [ ] **Test the build** on Linux, macOS (both Intel and ARM), and Windows in CI
- [ ] **Update conda-forge feedstock** recipe to use the new build system

### Optional

- [x] Add `editable install` support (`pip install -e .`) for development
- [x] Pin Fortran compiler in CI to ensure reproducible builds
- [x] Add a `noxfile.py` or `tox.ini` for standardised test/build/docs environments

### Testing

- [ ] `pip install .` works from a clean checkout on all three platforms
- [ ] `conda build` succeeds with the updated recipe
- [ ] All Stage 0 tests pass against the newly-built extension

---

## Stage 3: Replace the Fortran Solver (Performance-First)

**Goal:** Replace `solve.f90` with a maintainable implementation that is **at least as fast** as the current Fortran, while eliminating the Fortran compilation dependency.

### Why

The Fortran code is the single biggest barrier to:
- **conda-forge builds** — cross-compilation with gfortran is fragile, especially on Windows and ARM macOS
- **User installation from source** — most users don't have a Fortran compiler
- **Contributor onboarding** — few potential contributors read Fortran 90
- **Debugging and extending** — Python is far easier to iterate on

### Performance Constraint

Speed must not regress. The solver is called in inner loops during optimisation (`recursive_split` calls `fit_model`, which calls `model.run()` hundreds of times). Any slowdown here multiplies across every calibration run. The target is **performance parity or better** with the current Fortran.

### Approach

This is the largest and riskiest stage. A phased sub-approach is recommended:

#### Phase 3a: Create a Python Reference Implementation

- [ ] Write a **pure Python/NumPy** version of `solveSAS()` in a new module `mesas/sas/_solve_py.py`
- [ ] Match the Fortran interface exactly — same inputs, same outputs, same array shapes
- [ ] Implement all numerical schemes (Euler, RK2, RK4) and substep support
- [ ] **Do not optimise yet** — prioritise correctness and readability
- [ ] This reference implementation serves as a **specification** and **test oracle**, not as the production solver

**Testing:**
- [ ] Run every Stage 0 test against the Python solver and confirm numerical equivalence (within floating-point tolerance, e.g. `rtol=1e-10`)
- [ ] Benchmark and record the performance gap vs Fortran (expect 10-100x slower — that's fine at this stage)

#### Phase 3b: Build the High-Performance Solver

The reference implementation from 3a tells us exactly what needs to be fast. Now optimise:

- [ ] **Profile** the reference solver to identify hotspots (likely: the inner age-step loop, `get_flux`/`calculate_pQ`, RK4 state updates)
- [ ] **Primary strategy: Numba `@njit`** for the core solver loop. Numba compiles Python to LLVM machine code at import time — no compiler needed at install time, no compilation step during `pip install`. Numba achieves Fortran-comparable speed for numerical loops and is already standard in the scientific Python ecosystem (used by NumPy, SciPy users extensively).
- [ ] **Structure for vectorisation**: where possible, restructure the age-step loop to operate on arrays rather than scalar elements. NumPy vectorised operations + Numba `prange` for parallel outer loops can exceed single-threaded Fortran.
- [ ] **Alternative if Numba is insufficient**: Cython with typed memoryviews. More complex to write but gives C-level control. Would require a C compiler but not a Fortran compiler (much more widely available).
- [ ] **Consider algorithmic improvements** while rewriting:
  - The current Fortran allocates arrays of size `timeseries_length * n_substeps` on the stack. A more memory-efficient implementation could use rolling buffers.
  - The `precalculate_useful_things` pattern (precomputing gradients) is good — preserve and extend it.
  - The characteristic indexing (`jt_fullstep_at_`, `jt_substep_at_`) is clever but opaque. Document the scheme clearly in the rewrite.

**Testing:**
- [ ] Numerical equivalence with reference implementation (rtol=1e-12)
- [ ] **Performance target: within 1x of Fortran** (i.e., same speed or faster) for typical use cases
- [ ] Run the performance benchmark from Stage 0 on both implementations and record results
- [ ] Test with large timeseries (50,000+ steps) to ensure no memory blowup

#### Phase 3c: Swap the Default and Deprecate Fortran

- [ ] Make the new solver the **default** backend
- [ ] Keep the Fortran solver as an optional backend, selectable via `model.options['solver'] = 'fortran'`
- [ ] Deprecation warning when using the Fortran solver
- [ ] Update `pyproject.toml` to make the Fortran extension optional (extra: `pip install mesas[fortran]`)

#### Phase 3d: Remove the Fortran Code

- [ ] After one or two releases with the new solver as default, remove `solve.f90` and the Fortran build machinery entirely
- [ ] Simplify the build system (pure Python package — no Fortran compiler needed at install time)

### Optional Performance Enhancements

- [x] **JAX backend** for automatic differentiation — this could replace the partially-implemented Jacobian code in `solve.f90` and enable gradient-based optimisation (currently the ~200 lines of commented-out Jacobian code suggest this was attempted and abandoned). JAX's `jit` + `grad` would give both speed and analytical gradients for free.
- [x] **Parallel model runs** — when `recursive_split` evaluates multiple candidate subdivisions, these could run in parallel (multiprocessing or Numba `prange`)
- [x] **GPU acceleration** via JAX or CuPy for very large models — likely unnecessary for current use cases but would come naturally with a JAX backend

---

## Stage 4: API Cleanup and Modernisation

**Goal:** Clean up the public API, improve usability, and adopt modern Python patterns.

### Essential

- [ ] **Use dataclasses or attrs** for `Model` options and solute parameters instead of nested dicts
- [ ] **Replace the JSON config file approach** with a proper Python API (the JSON config can remain as a convenience loader)
- [ ] **Standardise the results interface** — currently results are a dict with inconsistent access patterns; consider a `Results` dataclass or xarray Dataset
- [ ] **Improve error messages** — validate inputs early with descriptive errors (e.g. mismatched timeseries lengths, missing columns in data_df)
- [ ] **Consistent naming** — the API mixes camelCase (`SAS_Spec`, `WaterBalance`) and snake_case; standardise on snake_case per PEP 8
- [ ] **Deprecation path** — keep old names as deprecated aliases for one release cycle

### Optional

- [x] Add an `xarray` integration for results (natural fit for the multi-dimensional age/flux/solute data)
- [x] Add a `__repr__` to `Model` that summarises the configuration
- [x] Support passing xarray Datasets or polars DataFrames as input (not just pandas)

### Testing

- [ ] All existing tests pass (using deprecated aliases where needed)
- [ ] New tests cover the updated API
- [ ] Deprecation warnings are emitted for old API usage

---

## Stage 5: Documentation Overhaul

**Goal:** Bring the ReadTheDocs documentation up to a standard that lets a new user go from install to results without reading the source code.

### Essential

- [ ] **Fix the ReadTheDocs build** — update `doc/readthedocs.yml` to use the current build system and Python version
- [ ] **Rewrite the quickstart tutorial** — a complete, runnable example from loading data to plotting results
- [ ] **Add a "Concepts" page** explaining SAS theory at a level appropriate for users (not just the GMD paper audience)
- [ ] **Document all configuration options** with examples and default values
- [ ] **Document the SAS function types** with mathematical definitions and plots
- [ ] **Auto-generate API docs** from the docstrings added in Stage 1
- [ ] **Add a gallery of examples** using `sphinx-gallery` or similar

### Optional

- [x] Add a "How it works" page explaining the numerical solver
- [x] Add a "Migration guide" for users of older versions
- [x] Add Jupyter notebook tutorials (rendered via `nbsphinx`)
- [x] Publish the docs build in CI so PRs get a preview link

### Testing

- [ ] `sphinx-build` completes without warnings
- [ ] All code examples in docs are tested via `doctest` or `sphinx-gallery`

---

## Stage 6: Extended Testing and Validation

**Goal:** Build confidence in the package through comprehensive testing.

### Essential

- [ ] **Property-based tests** (using `hypothesis`) — e.g. mass balance must always close, output probabilities must sum to 1
- [ ] **Cross-validation against the GMD paper results** — reproduce key figures
- [ ] **Test all SAS function types** systematically: piecewise, gamma, beta, kumaraswamy, arbitrary scipy.stats
- [ ] **Test time-varying parameters** — parameters that reference data columns
- [ ] **Test the model estimation module** (`recursive_split`) — currently untested

### Optional

- [ ] Benchmark against `tran-SAS` (the other SAS implementation mentioned in the paper) for cross-validation
- [x] Fuzz testing of the config parser
- [x] Memory profiling for large timeseries

---

## Suggested Execution Order and Timeline

```
Stage 0 (Testing baseline)     ──── Do this FIRST, before anything else
  │
  ├── Stage 1 (Code quality)   ──── Can start immediately after Stage 0
  │
  ├── Stage 2 (Build system)   ──── Can run in parallel with Stage 1
  │
  └── Stage 5 (Docs)           ──── Can start in parallel (fix build, write concepts)
        │
        v
Stage 3 (Fortran → Python)     ──── Biggest effort; depends on Stage 0 tests existing
  │
  v
Stage 4 (API cleanup)          ──── Depends on Stage 3 (don't change API while swapping solver)
  │
  v
Stage 6 (Extended testing)     ──── Final validation after all changes
```

Stages 1, 2, and the early parts of 5 can be worked on concurrently. Stage 3 is the critical path and should start as soon as Stage 0 is complete.

---

## Release Strategy

| Release | Contents | Breaking Changes |
|---------|----------|-----------------|
| **v2.0.0a1** | Stages 0-2 complete. New build system, better docs, same API | Build system only |
| **v2.0.0b1** | Stage 3c complete. Python solver is default, Fortran optional | Solver backend change (should be transparent) |
| **v2.0.0** | Stages 3d, 4, 5, 6 complete. Fortran removed, new API | API naming changes (with deprecation aliases) |

Each release should be published to conda-forge and PyPI (pure Python releases to PyPI become trivial after Stage 3d).

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| New solver is too slow | Low | High | Numba `@njit` achieves Fortran-parity in practice; Cython fallback; keep Fortran as optional backend until parity confirmed |
| Numerical differences between Python and Fortran solvers | Medium | High | Extensive regression tests with tight tolerances in Stage 0 |
| Breaking conda-forge builds during transition | Low | Medium | Test in CI with `conda build` before submitting feedstock PR |
| API changes break downstream users | Medium | Medium | Deprecation aliases for one full release cycle |
| Stage 3 takes much longer than expected | High | Medium | The phased sub-approach means partial progress is still valuable |
