# Deployment plan: mesas v2 (pure-Python/Numba)

State as of 2026-07-05: the post-refactor work is merged to `develop` (renamed
from `stochastic`). Build is hatchling, pure Python, Numba-JIT solver; 205
tests green. Published artifacts today: conda-forge `mesas` up to `1.20240418`
(Fortran build, `main` label). **Nothing is on PyPI — the name `mesas` is
unclaimed.**

## Step 0 — fix the version scheme (done: `2.0.0a1`)

The original `1.2026.0a1` could not ship: both conda and pip compare the
second segment numerically, and `2026 < 20240418`, so `1.2026.x` would sort
*below* the already-published conda `1.20240418` and never be selected as the
latest release.

Resolved 2026-07-05: version set to `2.0.0a1` (in `pyproject.toml` and the
`+dev` fallback in `mesas/__init__.py`). The Fortran→Numba rewrite is a
genuine major version; `2.x` sorts above every historical `1.2024…` release
and gives clean semver going forward.

## Step 1 — push and light up infrastructure

1. Push `develop` to GitHub; confirm the new pip-based CI matrix
   (`.github/workflows/tests.yml`) is green on all OS/Python combinations.
2. Read the Docs: activate `develop` as an *additional, non-default* version
   (Admin → Versions), served at `/en/develop/`. Do NOT change the default
   branch or default version — the landing docs must keep describing the
   stable v1.0 that default installs deliver, just as the beta packages sit
   behind an opt-in label/flag. At GA, point the default at the release tag.
3. Fix the README claim that PyPI hosts v1.0 (it hosts nothing).

## Step 2 — beta release on PyPI

The name is unclaimed, so the first upload registers it.

1. Configure **trusted publishing** on PyPI (project `mesas` → the
   `JH-LandHydroLab/mesas` repo + a `publish.yml` workflow) — no API tokens needed.
2. Add a publish workflow triggered on GitHub release / tag: `python -m build`
   then `pypa/gh-action-pypi-publish`. Dry-run against TestPyPI first.
3. Tag and publish `v2.0.0a1`.

Beta testers: `pip install --pre mesas` (or pin exactly:
`pip install mesas==2.0.0a1`). Once a final `2.0.0` exists, plain
`pip install mesas` never sees prereleases.

## Step 3 — beta release on conda-forge (non-default label)

Yes — conda-forge supports exactly this via **channel labels**. Packages
uploaded to a label other than `main` are invisible to normal
`conda install -c conda-forge mesas`; only users who explicitly add the label
channel get them. This is the documented conda-forge pre-release mechanism
(labels named `<package>_dev` or `<package>_rc`, versions carrying a PEP 440
pre-release suffix so they sort below the eventual final).

In `conda-forge/mesas-feedstock`, PR a recipe update:

```yaml
package:
  name: mesas
  version: 2.0.0a1        # from the PyPI sdist

build:
  noarch: python           # pure Python now — one build instead of a compiler matrix
  script: pip install . -vv

requirements:
  host: [python >=3.10, hatchling, pip]
  run:  [python >=3.10, numpy, scipy, pandas, matplotlib, numba]

extra:
  channel_targets:
    - conda-forge mesas_dev   # <- uploads go to the mesas_dev label, not main
```

(Also drop the Fortran compiler/meson machinery from the old recipe, and add
the `test:` imports/pytest section.)

Beta testers then run:

```bash
conda install -c conda-forge/label/mesas_dev -c conda-forge mesas
```

Regular users are unaffected: `conda install -c conda-forge mesas` still
resolves to `1.20240418` until a final release hits the `main` label.

**Faster alternative while iterating:** upload builds to a personal
anaconda.org channel with a label
(`anaconda upload --user charman2 --label beta …`), installable via
`conda install -c charman2/label/beta mesas`. No feedstock PR/review latency;
switch to the conda-forge label once the recipe stabilizes.

## Step 4 — beta cycle

Iterate `2.0.0a2 … 2.0.0b1 … 2.0.0rc1` as testers report issues. Each round:
tag → PyPI publish (automatic) → feedstock version bump PR (still targeting
`mesas_dev`). Keep a CHANGELOG entry per beta.

## Step 5 — general availability

1. Merge `develop` → `main`; tag `v2.0.0`.
2. PyPI: publish final (workflow handles it).
3. conda-forge: feedstock PR setting `version: 2.0.0` and removing
   `channel_targets` (reverts to `main` label). The regular bot will handle
   subsequent PyPI releases automatically once it sees them.
4. Read the Docs: mark the release version as `stable`.
5. GitHub release notes; if Zenodo integration is enabled, the release mints a
   DOI — keep `CITATION.cff` in sync.

## Step 6 — post-release

- README/installation docs: promote `pip install mesas` and
  `conda install -c conda-forge mesas`; remove beta instructions.
- Announce; deprecate the Fortran-era branches (`master`, `dev*`, `fixfortran`).
- Remaining Phase 7 feature work continues on `develop` per
  `IMPROVEMENT_PLAN.md`.
