.. _inputs:

=============================
Timeseries Inputs and Outputs
=============================

A single `Pandas <https://pandas.pydata.org/>`_ dataframe ``data_df`` is used to store all the input timeseries needed by the model. These data can be provided to the model as a path to a ``.csv`` file that will be automatically imported and stored as a dataframe.

The dataframe or ``.csv`` should be constructed before creating the model object, and then included in the model initialization. The following are equivalent:

.. code-block:: python

    import pandas as pd
    from mesas import Model

    # read input timeseries from a .csv file into a dataframe
    my_dataframe = pd.read_csv('my_input_timeseries.csv', ...)

    # create model
    my_model = Model(data_df=my_dataframe, ...)

and 

.. code-block:: python

    from mesas import Model

    # create model
    my_model = Model(data_df="path/to/my_data.csv", ...)

Model outputs are appended to the copy of the dataframe stored in ``my_model``. To access the model version call the property directly:

.. code-block:: python

    t = my_model.data_df.index
    y = my_model.data_df['some column name']
    plt.plot(t,y)

This version can also be modified in-situ, though the model must be re-run to generate corresponding results:

.. code-block:: python

    # Generate results with current inputs
    my_model.run()
    # Modify an input
    my_model.data_df['input variable'] = new_version_of_input_variable
    # Re-run model to update results
    my_model.run()

-----------------------
Input and output fluxes
-----------------------

To run the model the timeseries dataframe must contain one input flux timeseries, and at least one output flux. These can be specified in any units, but should be consistent with one another, and the flux values multiplied by the value in ``my_model.options['dt']`` (see :ref:`options`) should equal the total input and output mass of fluid (i.e. water) in each timestep. The timesteps as assumed to be all of equal interval, and equal length, and should contain no ``NaN`` values.

A simple steady flow model can be constructed by creating timeseries with constant values:

.. code-block:: python

    import pandas as pd
    import numpy as np

    N = 1000 # number of timesteps to run
    J = 2.5 # inflow rate
    Q = J # steady flow so input=output

    my_dataframe = pd.DataFrame(index = np.arange(N))

    my_dataframe['J'] = J
    my_dataframe['Q'] = Q

    ...

    my_model = Model(data_df=my_dataframe, ...)

    my_model.run()

The name of the column that contains the inputs is ``'J'`` by default, but can be modified by changing the ``'influx'`` option (see :ref:`options`). The name of the column that contains each flux is specified in the ``sas_specs`` input dictionary (see :ref:`sasspec`)

----------------------
Other input timeseries
----------------------

The timeseries dataframe also stores timeseries used in the specification of SAS functions (see :ref:`sasspec`) and solutes (see :ref:`solspec`). The column names specified in the ``sas_specs`` and ``solute_parameters`` inputs must exactly match a column in the ``data_df`` dataframe.

Here is a minimal example with steady inflow, time-variable discharge according to a linear storage-discharge relationship, uniform sampling, and a pulse of tracer input at a timestep some short time after the storage begins to fill. Note that the total storage ``S`` is stored in a column of the dataframe named ``'S'``, which is used in the specification of the uniform SAS function in ``my_sas_spec``. Similarly, the concentration timeseries is stored in a column of the dataframe named ``'Cin'``, which corresponds to a top-level key in ``my_solute_parameters``.

.. testcode:: ['a']

    import pandas as pd
    import numpy as np
    from mesas.sas.model import Model

    N = 1000 # number of timesteps to run
    t = np.arange(N)

    J = 2.5 # inflow rate
    k = 1/20 # hydraulic response rate
    Q = J * (1 - np.exp(-k * t))
    S = Q / k
    S[0] = S[1]/1000

    Cin = np.zeros(N)
    Cin[10] = 100./J

    my_dataframe = pd.DataFrame(index = t, data={'J':J, 'Q':Q, 'S':S, 'Cin':Cin})

    my_sas_specs = {
        'Q':{
            'a uniform distribution over total storage':{
                'ST': [0, 'S']
                }
            }
        }

    my_solute_parameters = {
        'Cin': {}
        }

    config = {
        'sas_specs': my_sas_specs,
        'solute_parameters': my_solute_parameters
        }

    my_model = Model(data_df=my_dataframe, config=config)

    my_model.run()

.. testcode:: ['a']
   :hide:

   print('Cin --> Q' in my_model.data_df.columns)
   print(not np.any(np.isnan(my_model.data_df['Cin --> Q'])))
   print((my_model.data_df['Cin --> Q'].sum()>50.) &(my_model.data_df['Cin --> Q'].sum()<=100.))

.. testoutput:: ['a']
   :hide:

   True
   True
   True

-----------------
Output timeseries
-----------------

If a timeseries of solute input concentrations is provided and its name appears as a top-level key in the ``solute_parameters`` dictionary, timeseries of output concentrations will be generated for each output flux specified in the ``sas_specs``.

The predicted outflow concentration timeseries will appear as a new column in the dataframe with the name ``'<solute column name> --> <flux column name>'``. For example, the outflow concentrations in the simple model given above will appear in the column ``'Cin --> Q'``.

To save the model outputs as a csv use::

    my_model.data_df.to_csv('outputs.csv')