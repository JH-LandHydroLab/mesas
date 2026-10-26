
==========
Quickstart
==========

This tutorial walks through creating, running, and visualising a simple SAS
model from scratch. By the end you will have a working model of tracer
transport through a well-mixed reservoir.

1. Create input data
====================

A mesas model needs a pandas DataFrame containing at least:

- an **influx** column (default name ``"J"``),
- one or more **outflux** columns, and
- optionally, one or more **solute concentration** columns.

.. code-block:: python

    import numpy as np
    import pandas as pd

    N = 200          # timesteps
    dt = 0.01        # timestep size
    Q_0 = 1.0        # steady flow rate
    S_0 = 5.0        # storage volume

    data_df = pd.DataFrame(index=range(N))
    data_df["J"] = Q_0                         # influx
    data_df["Q"] = Q_0                         # outflux (steady state)
    data_df["S_0"] = S_0                       # storage (used in SAS spec)
    data_df["C"] = 0.0                         # tracer concentration
    data_df.loc[10:20, "C"] = 10.0             # pulse of tracer

2. Define the SAS function
==========================

The SAS function determines *how water of different ages is selected for
discharge*. The simplest case is a **uniform distribution** over the total
storage — equivalent to a well-mixed reservoir.

.. code-block:: python

    sas_specs = {
        "Q": {                           # flux name (must match a column)
            "Q_SAS": {                   # component name (arbitrary label)
                "ST": [0, "S_0"]         # piecewise-linear CDF from 0 to S_0
            }
        }
    }

The ``"ST"`` key gives the breakpoints of a piecewise-linear CDF. Here we
use ``"S_0"`` (a string) to reference the time-varying column in the
DataFrame — the SAS function endpoint tracks the storage volume at each
timestep.

3. Define solute parameters
===========================

Tell the model which DataFrame column holds the input concentration:

.. code-block:: python

    solute_parameters = {
        "C": {                           # must match a column name
            "C_old": 0.0                 # concentration of pre-initial water
        }
    }

An empty dict ``{}`` uses all defaults (``C_old=0``). See
:ref:`solspec` for the full list of solute parameters.

4. Create and run the model
===========================

.. code-block:: python

    from mesas.sas.model import Model

    model = Model(
        data_df,
        sas_specs=sas_specs,
        solute_parameters=solute_parameters,
        dt=dt,
        verbose=False,
    )
    model.run()

5. View the results
===================

Predicted outflux concentrations appear as new columns in
``model.data_df`` with the naming pattern
``"<solute> --> <flux>"``:

.. code-block:: python

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(model.data_df.index * dt, model.data_df["C"], label="Input C")
    ax.plot(model.data_df.index * dt, model.data_df["C --> Q"], label="Output C")
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.legend()
    plt.show()

You should see the tracer pulse smeared out by the well-mixed reservoir.

6. Access internal state variables
==================================

After running, the model stores age-ranked arrays accessible via
getter methods or the :class:`~mesas.sas.model.ModelResult` object:

.. code-block:: python

    sT = model.get_sT()             # age-ranked storage density
    pQ = model.get_pQ("Q")          # age-ranked transit time distribution
    wb = model.get_water_balance()   # should be near machine precision

    print(f"Water balance max residual: {abs(wb).max():.2e}")

Using a JSON config file
=========================

Instead of passing dicts directly, you can store the configuration in a
JSON file:

.. code-block:: json

    {
        "sas_specs": {
            "Q": {
                "Q_SAS": {
                    "ST": [0, "S_0"]
                }
            }
        },
        "solute_parameters": {
            "C": {"C_old": 0.0}
        },
        "options": {
            "dt": 0.01,
            "verbose": false
        }
    }

Then create the model with:

.. code-block:: python

    model = Model(data_df, config="config.json")

Next steps
==========

- :ref:`sasspec` — learn about gamma, beta, and piecewise SAS functions
- :ref:`solspec` — configure solute reactions and evapoconcentration
- :ref:`options` — adjust numerical scheme, substeps, and recording
- :ref:`results` — extract age-ranked storage, transit times, and mass
