.. _config:

=====================
Configuring the model
=====================

To run `mesas.py` the user must provide some information about how the model is to be configured. First, at least one SAS function must be specified. In addition you may wish to provide information about solutes to be transported through the system, and/or modify the default model options.

The recommended way of providing configuration data is in a text file formatted using the `JSON standard <https://www.json.org/json-en.html>`_. Typically this file is called something like ``config.json``. The contents look very much like a python ``dict``, with some differences. A minimal example of a configuration file is a text file containing the following text:

.. code-block:: json

    {
    "sas_specs":{
        "Flux out":{
            "SAS fun":{
                "ST": [0, 100]
                }
            }
        }
    }


This specifies a single uniform SAS function between 0 and 100 associated with the flux named ``"Flux out"``.

`mesas.py` looks for three top-level entries in the file:

* ``"sas_specs"``: spcification of the SAS functions (see :ref:`sasspec`)
* ``"solute_parameters"``: information about solutes to be transported through the system (see :ref:`solspec`)
* ``"options"``: optional arguments for customizing the model run (see :ref:`options`)

For more information click the links above. Here is an example of the contents of a ``config.json`` file that uses all three of these top-level fields.

.. code-block:: json

    {
    "sas_specs":{
        "Flux out":{
            "SAS fun":{
                "ST": [0, 100]
                }
            }
        },

    "solute_parameters":{
        "solute A":{
            "C_old": 0.5
            }
        },

    "options":{
        "dt": 3600
        }
    }

