.. _options:

===================
Optional parameters
===================

Model options control the numerical solver, timestep size, and what
state is recorded. They can be set in three ways:

1. As keyword arguments when creating the model::

    model = Model(data_df, sas_specs=..., dt=0.01, n_substeps=5)

2. In the ``"options"`` section of a JSON config file:

.. code-block:: json

    {
        "options": {
            "dt": 0.01,
            "n_substeps": 5
        }
    }

3. By assigning a dict after creation::

    model.options = {"n_substeps": 10}

Only the specified options are changed; others retain their current
values.

The :class:`~mesas.sas.model.ModelOptions` dataclass can also be used
directly::

    from mesas import ModelOptions
    opts = ModelOptions(dt=0.01, n_substeps=5, num_scheme=2)

Available options
=================

``dt`` : float, default = 1.0
    Timestep size, in appropriate units. The product ``dt * flux`` must
    equal the volume of fluid transferred in one timestep.

``verbose`` : bool, default = False
    Print solver progress information.

``debug`` : bool, default = False
    Print detailed debug output from the solver.

``warning`` : bool, default = True
    Enable/disable solver warnings about numerical issues.

``num_scheme`` : int, default = 4
    Numerical integration scheme:

    - ``1`` — Forward Euler (1st order)
    - ``2`` — Midpoint method (2nd order)
    - ``4`` — Runge-Kutta 4 (4th order, recommended)

``n_substeps`` : int, default = 1
    Number of sub-timesteps per full timestep. Increasing this improves
    numerical accuracy at the cost of longer run times. Sub-step results
    are aggregated back to the full timestep.

``max_age`` : int, default = len(data_df)
    Maximum water age to track (in timesteps). Set to a value smaller
    than the timeseries length to reduce computation time. Water older
    than ``max_age`` is assigned concentration ``C_old``.

``sT_init`` : 1-D numpy array, default = zeros
    Initial age-ranked storage distribution (density form). Useful for
    spin-up. If provided, ``max_age`` is set to ``len(sT_init)``.

``influx`` : str, default = ``"J"``
    Name of the column in ``data_df`` containing the inflow rate.

``record_state`` : bool or str, default = False
    Controls which timesteps are recorded in the output arrays:

    - ``False`` — record only the final timestep (saves memory)
    - ``True`` — record every timestep (required for water/solute
      balance checks)
    - A string — name of a boolean column in ``data_df`` indicating
      which timesteps to record

``jacobian`` : bool, default = False
    Not implemented in the v2 solver: setting ``True`` raises
    ``NotImplementedError``. Use numerical jacobians instead (e.g.
    ``mesas.me.recursive_split.fit_model`` with
    ``jacobian_mode="numerical"``).

``record_every`` : int or None, default = None
    Record the state at the end of every k-th timestep — a memory-saving
    alternative to ``record_state=True``.

``record_arrays`` : "all" or iterable of str, default = "all"
    Which state arrays to record (subset of ``{"sT", "pQ", "mT", "mQ",
    "mR", "water_balance", "solute_balance"}``). Unselected arrays are
    never allocated.

``record_dtype`` : {"float64", "float32"}, default = "float64"
    Storage dtype for the recorded arrays; ``"float32"`` halves memory.
    Computation is always float64.

``record_to`` : str or None, default = None
    Directory in which to store the recorded arrays as disk-backed
    ``.npy`` memory maps instead of RAM. Reloadable later with
    :meth:`Model.load_state`. See :ref:`results`.

``validate_inputs`` : bool, default = True
    Check input data and SAS parameters for NaN, negative fluxes, and
    non-positive distribution parameters before running, raising a
    descriptive error. Set ``False`` to skip.

``ST_smallest_segment`` : float, default = 0.01
    Minimum allowed piecewise segment length along the :math:`S_T` axis.

``ST_largest_segment`` : float, default = inf
    Maximum allowed piecewise segment length along the :math:`S_T` axis.
