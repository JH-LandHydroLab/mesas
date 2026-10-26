.. _results:

==================
Extracting results
==================

After calling ``model.run()``, results are available in two ways:

1. **Outflux concentrations** appear as new columns in ``model.data_df``
   with names like ``"<solute> --> <flux>"``.

2. **Age-ranked state variables** are accessible through getter methods
   or the :class:`~mesas.sas.model.ModelResult` object at
   ``model.result``.

Saving concentration timeseries
================================

.. code-block:: python

    model.run()
    model.data_df.to_csv("results.csv")

Getter methods
==============

The following methods return age-ranked arrays. Each accepts optional
``timestep``, ``agestep``, or ``inputtime`` keyword arguments to
slice the result.

``model.get_sT()``
    Age-ranked storage density. Shape ``(max_age, n_output_steps)``.

``model.get_pQ(flux)``
    Age-ranked transit time distribution for the named flux.

``model.get_mT(sol)``
    Age-ranked solute mass density for the named solute.

``model.get_CT(sol)``
    Age-ranked concentration ``mT / sT`` (NaN where ``sT = 0``).

``model.get_mQ(flux, sol)``
    Age-ranked solute mass flux for a given flux and solute.

``model.get_mR(sol)``
    Age-ranked reaction mass for a given solute.

``model.get_water_balance()``
    Water conservation residual. Should be near machine precision when
    ``record_state=True``.

``model.get_solute_balance(sol)``
    Solute conservation residual for a given solute.

``model.get_ST()``
    Cumulative storage ``ST = cumsum(sT) * dt``.

The ModelResult object
======================

``model.result`` returns a :class:`~mesas.sas.model.ModelResult` that
supports both dict-style and attribute-style access:

.. code-block:: python

    # These are equivalent:
    sT = model.result["sT"]
    sT = model.result.sT

    # Snake_case names for balance arrays:
    wb = model.result.water_balance
    sb = model.result.solute_balance

Available keys: ``sT``, ``pQ``, ``water_balance``, ``dsTdSj``, plus
``C_Q``, ``mT``, ``mQ``, ``mR``, ``solute_balance``, ``dmTdSj`` and
``dCdSj`` when at least one solute is configured.

.. note::

    The camelCase names ``WaterBalance`` and ``SoluteBalance`` are
    deprecated. They still work, but dict-style access (e.g.
    ``model.result["WaterBalance"]``) emits a ``DeprecationWarning``.
    Use ``water_balance`` and ``solute_balance`` instead.

Array ordering
==============

Getter methods return arrays with age as the first axis and timestep
as the second:

- ``sT``: ``(max_age, n_output_steps)``
- ``pQ``: ``(max_age, n_output_steps, n_fluxes)``
- ``mT``: ``(max_age, n_output_steps, n_solutes)``
- ``C_Q``: ``(n_timesteps, n_fluxes, n_solutes)``

The order of fluxes and solutes matches ``model.fluxorder`` and
``model.solorder``.

Managing memory for long runs
=============================

With ``record_state=True`` the recorded arrays grow as the *square* of the
timeseries length — a 20-year daily run records several GB. Four options
keep long runs tractable; they compose freely:

**Record fewer timesteps.** Either pass ``record_every=k`` to record every
k-th step, or pass the name of a boolean column in ``data_df`` to record
exactly the flagged steps::

    model = Model(df, config="config.json", record_every=30)     # every 30th step
    model = Model(df, config="config.json", record_state="obs")  # steps where df["obs"] is True

**Record fewer arrays.** ``record_arrays`` selects which state arrays to
allocate and fill; the others are never allocated. ``mQ`` is usually the
largest::

    model = Model(df, config="config.json", record_state=True,
                  record_arrays={"sT", "pQ"})

**Record in float32.** ``record_dtype="float32"`` halves memory. The
computation itself is always float64, and the recorded state stays accurate
to about seven significant digits — ample for plotting and analysis. Use the
default float64 when you want the ``water_balance``/``solute_balance``
diagnostics to close to machine precision.

**Record to disk.** ``record_to="some/directory"`` stores the recorded
arrays as memory-mapped ``.npy`` files instead of RAM, so memory stops being
the limiting factor entirely. Results are accessed exactly as usual (the
arrays in ``model.result`` are read-only memory-maps), and the directory is
self-describing: it can be reloaded later — even in a different session,
without re-running the model — with::

    result = Model.load_state("some/directory")
    result["sT"]           # read-only memmap, loads lazily from disk
    result["run_metadata"] # dict: dt, fluxes, solutes, shapes, versions...

Predicted outflow concentrations (``C_Q`` and the ``"<solute> --> <flux>"``
columns in ``data_df``) are always computed in full float64 precision and
are unaffected by any of these options.
