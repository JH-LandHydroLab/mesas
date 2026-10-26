# Spec: reducing memory overhead of `record_state=True` runs

Status: **O1–O5 implemented 2026-07-07** (O6/O7 held in reserve as planned).
Shipped: `record_every`, `record_arrays`, `record_dtype`, `record_to`
options; `Model.load_state`; dead Jacobian placeholders removed; solver
output layout is now natively age-first (the `moveaxis` in `model.py` is
gone). All open questions resolved — see end of document.

## Problem

With `record_state=True`, the solver records the full age-ranked state at
every timestep. The recorded arrays scale as **O(N²)** in the timeseries
length N (each of N timesteps records a value for each of up to N ages), and
for multi-decade daily records they exhaust RAM long before compute time
becomes a problem. Users must currently choose between full state output and
long runs.

## Confirmed: the arrays are write-only during the run

Verified against `mesas/sas/_solve_numba.py` (v2.0.0a1):

1. **The integration loop never reads the recorded arrays.** The solver's
   state is carried in separate O(N·n_substeps) working arrays (`sT_start`,
   `mT_start`, `STcum_topbot_start`, `pQ_aver`, …). The recording function
   `_update_records` only accumulates (`+=`) into the output arrays
   (`sT_outputstep`, `mT_outputstep`, `pQ_outputstep`, `mQ_outputstep`,
   `mR_outputstep`) and never reads them back.
2. **Writes are age-local.** The outer solver loop is over age `iT` (not
   time); iteration `iT` writes only age slices `iT` and `iT+1` across the
   recorded timesteps. Age slice `iT` is therefore *finalized* once outer
   iteration `iT` completes — the property that makes streaming/eviction
   designs possible.
3. **One read pass after the loop.** `_calculate_balances` runs once after
   integration, reading adjacent age slices (`iT-1`, `iT`) and adjacent
   recorded steps sequentially to fill `WaterBalance`/`SoluteBalance`. It is
   a streaming-friendly access pattern (and could be fused into the main
   loop's slice-finalization if needed).
4. `C_Q` (the predicted outflow concentrations — the thing most runs are
   actually for) is accumulated from the working arrays, **not** from the
   recorded arrays. Skipping state recording loses no concentration output.

So the premise holds: recorded state is a pure output stream, and nothing
about the algorithm requires it to live in RAM.

## Memory model

Let T = number of recorded timesteps (= N for `record_state=True`),
A = `max_age` (= N by default), q = numflux, s = numsol, p = numargs_total.
All arrays are float64 (8 B).

| Array | Shape | Bytes |
|---|---|---|
| `sT` | (T+1, A) | 8·T·A |
| `mT` | (T+1, s, A) | 8·T·A·s |
| `pQ` | (T, q, A) | 8·T·A·q |
| `mQ` | (T, q, s, A) | 8·T·A·q·s |
| `mR` | (T, s, A) | 8·T·A·s |
| `water_balance` | (T, A) | 8·T·A |
| `solute_balance` | (T, s, A) | 8·T·A·s |
| **Real outputs total** | | **8·T·A·(2 + q + 3s + q·s)** |
| `dsTdSj` (unused) | (T+1, p, A) | 8·T·A·p |
| `dmTdSj` (unused) | (T+1, p, s, A) | 8·T·A·p·s |
| **Dead placeholders total** | | **8·T·A·p·(1 + s)** |

Concrete examples (q=1, s=1, gamma SAS ⇒ p=3, `record_state=True`):

| Record length | N | Real outputs | Dead placeholders | Total today |
|---|---|---|---|---|
| 10 y daily | 3,652 | 0.75 GB | 0.64 GB | 1.4 GB |
| 20 y daily | 7,305 | 3.0 GB | 2.6 GB | 5.6 GB |
| 50 y daily | 18,262 | 18.7 GB | 16.0 GB | 34.7 GB |
| 100 y daily | 36,525 | 74.7 GB | 64.0 GB | 138.7 GB |

## Options

### O1 — Stop allocating the dead Jacobian placeholders (ship in 2.0.0)

`ds_outputstep`, `dm_outputstep`, `dC_fullstep` are zero-filled, **never
written** (the Numba solver raises `NotImplementedError` if `jacobian=True`),
and returned as zeros. Allocate them with singleton dimensions
(`(1, 1, 1)`-shaped) and have `model.py` expose empty results.

- Impact: removes ~45 % of current allocation for typical p. Free.
- Effort: S. Risk: none (nothing consumes these arrays' contents).

### O2 — `record_dtype` option: record in float32

Computation stays float64; only the recorded copies are cast on write.

- Impact: 2× on everything recorded. Composes with every other option.
- Cost: balance diagnostics and recorded state lose half the significant
  digits (~7 decimal digits — ample for plotting/analysis; document that
  mass-balance closure checks should use float64 runs).
- Effort: M (dtype-parameterizing the Numba kernels via `.astype` at the
  write sites; numba specializes per dtype automatically).

### O3 — Promote the existing time-subsampling (docs only)

`record_state="colname"` (a boolean column) already records only flagged
timesteps — T drops from N to however many you flag; memory falls
proportionally. This shipped with the refactor but is effectively
undocumented. Add a docs section + example (e.g. record month-ends:
365× reduction for daily data), and optionally sugar like
`record_every=30`.

- Impact: user-controlled, potentially huge. Age axis (A) unaffected.
- Effort: S. Risk: none.

### O4 — Per-array selection: `record_arrays={"sT", "pQ"}`

Most workflows want `sT` (TTDs) and maybe `pQ`; `mQ` — the *largest* array
(q·s·T·A) — is rarely inspected. Allocate singleton dummies for arrays not
requested (same mechanism as O1); skip their `_update_records` writes and
their balance calculation.

- Impact: an sT-only run drops from (2+q+3s+qs)·8·T·A to ~2·8·T·A — 3.5×
  for q=s=1, more for multi-solute/multi-flux models.
- Effort: M. Risk: low (flag-guarded writes; slight kernel branching).

### O5 — Disk-backed recording via memmap: `record_to="rundir/"` (recommended core)

Allocate the recorded arrays as `np.lib.format.open_memmap(...)` `.npy`
files instead of RAM, and pass them into the solver unchanged.
**Prototype-validated 2026-07-07: Numba `@njit` kernels write directly into
`np.memmap` arrays with zero code changes, and data round-trips through
disk.** The OS page cache absorbs writes and evicts under pressure, so
resident memory stays bounded regardless of array size.

Design points:

- Allocation currently happens *inside* jitted `_solve_core`. Move the seven
  output allocations up into the plain-Python `solve()` wrapper and pass
  them as arguments (mechanical change; also enables O1/O4 cleanly).
- **Store age-first on disk** (`(A, T, …)` rather than the solver's current
  `(T, …, A)`). The solver writes age slices `iT`/`iT+1` per outer
  iteration; age-first layout makes each iteration's writes two contiguous
  runs instead of T scattered strides — the difference between streaming
  and page-cache thrash when the file exceeds RAM. `model.py` already
  presents age-first arrays to users (today via `np.moveaxis` views), so
  the user-facing layout is unchanged and the moveaxis simply disappears.
  (Kernel index order changes accordingly; a one-time mechanical edit.)
- Results in `model.result` become memmap-backed arrays — same API,
  lazily paged, and they *persist*: a `rundir/` of standard `.npy` files
  reloadable later with `np.load(..., mmap_mode="r")` without rerunning.
- In-RAM behavior (`record_to=None`) unchanged; same code path, allocator
  switches between `np.zeros` and `open_memmap`.

- Impact: removes RAM as the constraint entirely; limit becomes disk.
  100 y daily with O1+O4(sT,pQ)+O2: ~10 GB of files, ~zero resident.
- Effort: M–L (allocation hoist + index-order flip + tests).
- Risks: slower on network/spinning disks (document SSD expectation);
  Windows file-lock semantics on cleanup (test in CI matrix); user must
  manage the run directory's lifetime.

**Measured performance impact** (2026-07-07, Apple Silicon laptop NVMe;
microbenchmark reproducing the solver's exact write pattern — age-major
accumulation into slices `iT`/`iT+1`, 1.4 GB across 7 channels, the same
volume as an N=5000 full-record run):

| Strategy | Time | vs RAM |
|---|---|---|
| RAM (`np.zeros`) | 0.285 s | 1.0× |
| memmap, lazy OS write-back | 0.576 s | 2.0× |
| memmap + incremental flush (256-slice) | 1.08 s | 3.8× |
| memmap + full forced `flush()` | 1.73 s | 6.1× (≈1.1 GB/s SSD) |

Interpretation. The overhead applies only to the *recording* portion of a
run, and lazy write-back is the default user experience (dirty pages flush
in background after `run()` returns). Two regimes:

1. *Recorded state fits in page cache* (all currently-possible runs): the
   worst case is a uniform-SAS run, which is maximally recording-dominated —
   there the ~0.3 s excess on 1.4 GB projects to **~25–30 % slowdown**
   (0.89 s → ~1.15 s at N=5000). Gamma/substep runs are compute-dominated:
   **low single-digit %**. If `run()` should not return until data is
   durable, forced flush adds disk-bandwidth time (~1.1 GB/s here).
2. *Recorded state exceeds RAM* (the runs this feature exists for): write-
   back is on the critical path, so throughput is bounded by SSD bandwidth.
   Full-float64-everything generation peaks at ~1.6 GB/s (uniform case) vs
   ~1.1 GB/s disk → **up to ~1.5–2× slower** worst case. With O4 (sT+pQ) +
   O2 (float32) the generated volume drops ~7× to ~0.2–0.3 GB/s and the
   SSD keeps up — **overhead becomes negligible exactly in the recommended
   configuration**. And the baseline for these runs is "impossible" (OOM),
   so even the worst case is a strict win.

Mitigation worth implementing: periodic `msync` of *finalized* slices (the
incremental strategy above) bounds the dirty-page backlog so the OS never
stalls the process in a write-back storm; it costs ~2× on the recording
portion but makes throughput predictable. Make it automatic when the
predicted recorded volume exceeds ~25 % of physical RAM.

### O6 — True streaming with compression (zarr/HDF5), O(N) resident

Restructure the outer age loop so each finalized age slice is flushed
through a Python-side writer (compressed zarr/HDF5 chunk), keeping a
two-slice rolling window in RAM. Balance calculation fuses into the
flush (it only needs slices `iT-1`, `iT`).

- Impact: resident memory O(N); smooth SAS state compresses well (5–20×
  on disk plausible with zstd).
- Effort: L (loop restructuring around the fused parallel kernel, objmode
  or chunked-call architecture; performance regression risk for the Phase 2
  fusion; new optional dependency).
- Performance expectation (not measured): compression throughput is
  ~0.5–1 GB/s/core (zstd) and competes with the parallel solver for the
  same cores — estimate **10–30 % slowdown** in steady state, *plus* the
  unquantified risk that restructuring the outer loop degrades the Phase 2
  fused-kernel speedup itself. Disk bandwidth stops mattering (compressed
  volume is 5–20× smaller), so this trades a predictable I/O bound for a
  CPU tax on every run.
- Verdict: hold in reserve — only if O5 proves insufficient in practice.

### O7 — Age-axis coarsening (log-spaced age bins)

Record age-aggregated state (e.g. daily resolution for young water,
log-widening bins for old). A → n_bins turns O(N²) into O(N·n_bins).

- Impact: potentially the largest asymptotic win.
- Cost: changes the *meaning* of returned arrays (binned, not per-age);
  balance diagnostics need binned formulations; API/vis implications.
- Effort: L. Verdict: scientifically attractive but a separate feature
  discussion, not a drop-in memory fix.

## Recommendation

Phased:

1. **2.0.0 (now):** O1 (dead placeholder removal — ~45 % for free) and O3
   (document time-subsampling). No API additions; nothing to deprecate.
2. **2.1 feature — "state recording, grown up":** O5 (`record_to=` memmap
   backing, age-first layout) + O4 (`record_arrays=`) + O2
   (`record_dtype=`). Together these compose into arbitrary-length runs:
   100 y daily sT+pQ float32 ≈ 10 GB on disk, ~zero resident.
3. **Later, only if demanded:** O6 compression streaming; O7 age binning as
   its own scientific feature.

### API sketch (2.1)

```python
model = Model(
    data_df,
    sas_specs=...,
    record_state=True,          # unchanged: True | False | "bool_column"
    record_to="results/run1/",  # NEW: directory for .npy memmaps (None = RAM)
    record_arrays={"sT", "pQ"}, # NEW: subset of {"sT","pQ","mQ","mR","mT",
                                #      "water_balance","solute_balance"}; "all" default
    record_dtype="float32",     # NEW: "float64" (default) | "float32"
)
```

`model.result` keys, shapes, and age-first orientation are unchanged in all
modes; only the backing storage differs.

## Verification plan

- Equivalence: in-RAM vs memmap vs float32 runs on the Stage 0 benchmark
  problems; sT/pQ/mQ checksums equal (float64) or within float32 tolerance.
- Balance closure: `water_balance`/`solute_balance` near machine precision
  in float64 modes (existing tests), documented looser bound for float32.
- Memory ceiling test: N=20,000 synthetic run with `record_to=` under an
  RSS assertion (psutil), proving resident memory stays < some bound while
  files reach the predicted size.
- Performance: Stage 0 perf tests within noise for in-RAM path; memmap path
  benchmarked on SSD, documented.
- Windows/macOS/Linux CI coverage for memmap file lifecycle.

## Open questions — resolved 2026-07-07

1. **Yes** — implemented. `record_to` directories contain `mesas_run.json`
   (version, dt, orders, shapes, dtype), `index_ts.npy`, and `C_Q.npy`
   alongside the state arrays, and reload via `Model.load_state("rundir/")`
   as read-only memmaps.
2. **Yes** — `record_arrays` defaults to `"all"` through the 2.x series;
   revisit the `{"sT","pQ"}` default with a deprecation note in 3.0.
3. **Yes, float32 is acceptable for state arrays** — experiment run
   2026-07-07 (strong evapoconcentration: half of influx evaporates with
   alpha=0, input concentrations spanning 6 orders of magnitude, small
   storage, n_substeps=4): max relative error of float32-recorded sT/mT/mQ
   vs float64 is 1–3×10⁻⁷, i.e. float32 machine epsilon — no error
   amplification from the accumulation pattern. The balance *diagnostics*
   are the only casualty (water 2×10⁻⁷ vs 2×10⁻¹⁶; solute 1×10⁻⁴ vs
   4×10⁻¹³): documented — use float64 runs for closure checks. `C_Q` is
   bit-identical (computed independently in float64). Regression-tested in
   `test/test_record_options.py`.
