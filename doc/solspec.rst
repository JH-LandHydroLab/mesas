.. _solspec:

============================
Specifying solute parameters
============================

Solute parameters are also given using a nested set of python dictionaries or a JSON entry in a ``config.json`` file. These are usually set when the model object is created. See below for how to change solute parameters after model creation.

Keys of the highest level of the dictionary are assumed to be the ``data_df`` column names for the input concentrations of each solute. The value associated with each key should be a (possibly empty) dictionary of parameters. For example:

.. code-block:: json

    {
    
    "...": "...",
    
    "solute_parameters": {
        "C1 [per mil]": {},
        "C2 [stone/bushel]": {}
        }
    }

The ``"...": "..."`` in the example above should not be included in a real file. There are used here to show where additional information :ref:`sasspec` and setting :ref:`options` may be included.

The equivalent python dictionary is:

.. code-block:: python

    my_solute_parameters = {
        "C1 [per mil]": {},
        "C2 [stone/bushel]": {}
        }

In this case `mesas.py` would expect to find columns ``"C1 [per mil]"`` and ``"C2 [stone/bushel]"`` in the :ref:`timeseries input data <inputs>` that contain the input concentrations of two solutes. Since an empty dict ``{}`` is passed in each case, default parameters (representing a conservative ideal tracer initially absent from the system) will be used.

The parameter dictionary may specify any of the following default ``key:value`` pairs. If a ``key:value`` pair does not appear in the dictionary, the default value will be used.

``mT_init`` (array-like, default = [0.0, 0.0, ...])
  Initial age-ranked mass in the system. This is useful if the system is initialized by some sort of spin-up. Each entry is age-ranked mass in an age interval of duration :math:`\Delta t`. If ``mT_init`` is specified, ``sT_init`` must also be specified in :ref:`options`, and be of the same length. The element-wise ratio ``mT_init/sT_init`` gives the age-ranked concentration ``CS`` of the water in storage at time zero. Note that if ``sT_init`` is specified but ``mT_init`` is not, the concentrations associated with each non-zero value of ``sT_init`` will be zero.

``C_old`` (float, default = 0.0)
  Old water concentration. This will be the concentration of all water released of unknown age. If ``sT_init`` is not specified, this will be all water in storage at ``t=0``. If ``sT_init`` is specified, it will be all water older than the last non-zero entry in ``sT_init``.

``k1`` (float or string, default = 0.0)
  First-order reaction rate constant. The rate of production/consumption of solute :math:`s` is modeled as:

.. math:: \dot{m}_R^s(T,t)=k_1^s(t)(C_{eq}^s(t)s_T(t,T) - m_T(t,T))

..

  where :math:`m_R^s(T,t)` is the rate of change of mass of solute :math:`s` in storage. Note that the reaction rate has units of [1/time], and should be in the same time units as the fluxes. The reaction rate can be made time-varying by associating ``k1`` with a string referring to a column of reaction rates in ``data_df``.

``C_eq`` (float or string, default = 0.0)
  Equilibrium concentration. See above for role in modeling first order reactions. Note the equilibrium concentration can be made time-varying by associating ``C_eq`` with a string referring to a column of equilibrium concebntrations in ``data_df``.

``alpha`` (dict, default = 1.0)
  A dict giving partitioning coefficients for each outflow. The rate of export :math:`\dot{m}_q^s(T,t)` of solute :math:`s` through outflow :math:`q` is:

.. math:: \dot{m}_q^s(T,t)=\alpha_q^s(t)\frac{m_T^s(T,t)}{s_T(t,T)}p_q(T,t)Q_q(t)

..

  Thus if :math:`\alpha_q^s=1` the outflow concentration of water of a given age will equal that in storage. If :math:`\alpha_q^s=0`, the solute will not be exported with outflow :math:`q`. Values of :math:`0<\alpha_q^s<1` will result in partial exclusion of the solute from the outflow, and :math:`\alpha_q^s>1` will result in preferential removal via outflow outflow :math:`q`.

  The keys in the ``alpha`` dict must match keys in top level keys of ``sas_specs``. Each key may be associated with a number or a string referring to a column of partitioning coefficients in ``data_df``.
  {"Q": 1., ...}   # Partitioning coefficient for flux "Q"

``observations`` (dict, default = ``{}``)
  This dict provides the name of columns in `data_df` that contain observations that may be used to calibrate/validate the model"s predictions of outflow concentrations. Keys are outflow fluxes named in top level keys of ``sas_specs``, e.g. ``"observations":{"Q": "obs C in Q", ...}``.

--------------------
Modifying parameters
--------------------

There are two equivalent ways to modify the parameters of an existing model.

Assigning a dict
  The model property ``<my_model>.solute_parameters`` can be assigned a dict of valid key-value pairs. Note that this rebuilds the solute parameters from scratch: only the solutes named in the new dict will be retained, and any parameters not given in the dict are reset to their default values.

  To remove all solute parameters (so no solute transport will be modelled) set ``<my_model>.solute_parameters=None``. Default parameters can then be set by assigning an empty dict to each solute ``<my_model>.solute_parameters={"C1":{}, ...}``

Using the ``<my_model>.set_solute_parameters()`` function
  Individual properties of an existing solute can be updated using this convenience function, leaving its other parameters unchanged. Pass the solute name and a dict of the parameters to change, like this:

.. code-block:: python

    <my_model>.set_solute_parameters("C1", {"C_old": 22.5})

This would set the ``C_old`` property associated with solute ``"C1"`` to ``22.5``.


