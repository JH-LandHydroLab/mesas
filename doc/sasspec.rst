.. _sasspec:

========================
Specifying SAS functions
========================

SAS functions are specified using a nested dictionary-like structure, stored as either python code or in a ``config.json`` file (see :ref:`config`). A simple JSON specification looks like this:

.. code-block:: json

    {
    "sas_specs":{
        "Flux out":{
            "SAS fun":{
                "ST": [0, 100]
                }
            }
        },
    
    "...": "..."
    
    }

The ``"...": "..."`` in the example above should not be included in a real file. There are used here to show where additional information :ref:`solspec` and setting :ref:`options` may be included.

The equivalent python dictionary looks like this:


.. code-block:: python

    my_sas_spec = {
        "Flux out":{
            "SAS fun":{
                "ST": [0, 100]
                }
            }
        }

The levels of the ``sas_specs`` nested dictionary / JSON entry are as follows:

Level 1: One ``key:value`` pair per outflow flux (e.g. discharge, ET, groudwater discharge, etc.)
  Each key is a string that corresponds to a column in the input timeseries dataset ``data_df`` that contains the corresponding outflow flux rates. To specify multiple outflows, just add additional ``key:value`` pairs to this level of the dictionary.

Level 2: One ``key:value`` pair per component SAS function to be combined in a time-variable weighted sum
  If there is more than one key in a dict at this level, the model will assume that the dictionary values are specifications for SAS functions, and the keys correspond to columns in ``data_df`` with the time-varying weight for that component. This can be a useful way to specify time-varying SAS functions, but the individual SAS functions can themselves be time-varying.

Level 3: ``key:value`` pairs giving properties of the SAS function
  Where properties are given as strings rather than numbers it will be assumed (with a few exceptions) that the associated values are time-varying and are given by the corresponding column in ``data_df``.

There are four ways to specify a SAS function in the level 3 dict:

 - As an exact gamma or beta distribution
 - Using any distribution from ``scipy.stats`` (which will be converted into a piecewise linear approximation in the model)
 - As a piecewise linear CDF
 - As a B-Spline PDF (not yet implemented)

----------------------------------
Using a gamma or beta distribution
----------------------------------

The gamma and beta distributions are often used to model SAS functions due to their flexibility. An important difference between them is that the beta distribution only has non-zero probability over a finite interval, while the gamma distribution is defined for infinitely large values. The beta distribution is useful when the total storage volume can be reliably inferred from tracer data. The gamma may be preferable where a large portion of the storage volume turns over very slowly, and so its volume is difficult to constrain from the tracer data.

The PDF of a beta distribution is given by:

.. math:: f(x)=x^{\alpha-1}(1-x)^{\beta-1}N(\alpha,\beta)

..

where :math:`\alpha` and :math:`\beta` are parameters and :math:`N(\alpha,\beta)` normalizes the distribution to have unit area. The gamma distribution is given by:

.. math:: f(x)=x^{\alpha-1}e^{-x}N(\alpha)

..

which has only one shape parameter, :math:`\alpha`, and again :math:`N(\alpha,\beta)` normalizes the distribution. These distributions can be used flexibly by converting the input values of :math:`S_T` into a normalized form :math:`x`:

.. math:: x(T,t)=\frac{S_T(T,t) - S_\mathrm{loc}(t)}{S_\mathrm{scale}(t)}

..

where 

 - :math:`S_\mathrm{loc}(t)` or ``"loc"`` : the location parameter, which shifts the distribution to the right for values >0 and to the left for values <0
 - :math:`S_\mathrm{scale}(t)` or ``"scale"`` : the scale parameter

Note that both ``"loc"`` and ``"scale"`` must be given explicitly in the ``"args"`` dict when using the built-in gamma and beta distributions -- omitting either will result in an error.

The desired distribution is specified using the key ``"func"``, and the associated parameters using the keyword ``"args"``, as illustrated in the example below. All parameters can be made time-varying by setting them to a string corresponding to a column in ``data_df``.

+++++++
Example
+++++++

Here is an examples of a SAS specification for two fluxes, ``"Discharge"`` and ``"ET"``, whose SAS function will be modeled as a gamma and beta distribution respectively. The scale parameter of the discharge gamma distribution is set to ``"S0"`` indicating that its values can be found in that column of the ``data_df`` dataframe.

.. code-block:: json

    {
    "sas_specs":{
        "Discharge": {
            "Discharge SAS fun": {
                "func": "gamma", 
                "args": {
                    "a": 0.62,
                    "scale": "S0",
                    "loc": 0.0
                    }
                }
            },
        "ET": {
            "ET SAS fun": {
                "func": "beta",
                "args": {
                    "a": 2.31,
                    "b": 0.627,
                    "scale": 1402,
                    "loc": 248
                    }
                }
            }
        },

    "...": "..."

    }

In this case the model will look for columns in ``data_df`` called ``"Discharge"`` and ``"ET"``, and assume the values in these columns are timeseries of outflows from the control volume. Note that the values in these columns must be in the same units.

The ``"Discharge"`` flux has a single component SAS function named "Discharge SAS fun". Since there is only one component SAS function for the "Discharge" flux there does not need to be a column in the dataframe called "Discharge SAS fun". We specify the SAS function as a gamma distribution with the key:value pair ``"scipy.stats": gamma``. The distribution properties are set in the dictionary labeled ``"args"``. The gamma distribution with shape parameter ``"a"`` which is here set to ``0.62``.

The "ET" flux has a SAS function named ``"ET SAS fun"``.  This is specified to be a beta distribution, which has two shape parameters: ``"a"`` and ``"b"``.  As before, these are set in the ``"args"`` dictionary, along with the scale and shape parameters.


--------------------------------------------------
Using parameterized distributions from scipy.stats
--------------------------------------------------

``Scipy.stats`` provides a `large library <https://docs.scipy.org/doc/scipy/reference/stats.html>`_ of probability distributions that can be used to specify a SAS function. Note that only continuous distributions with non-negative support are valid SAS functions (though the support need not be finite).

The continuous distribution is converted into a piecewise linear approximation, which is then passed into the core number-crunching part of the code. This is done because evaluating the native ``scipy.stats`` functions was found to be too computationally expensive.

To use them, the distributions are specified using the same format as above, but with the additional key ``"use": "scipy.stats"``. The Level 3 dictionary in this case should therefore have four key:value pairs

 - ``"func"`` : <a string giving the name of a ``scipy.stats`` distribution>
 - ``"use"`` : ``"scipy.stats"``
 - ``"args"`` : <a dict of parameters to be passed to the distribution
 - ``"nsegment"`` : <an integer giving the number of segments to use in the piecewise linear approximation (optional, default is 25)>

The dict associated with ``"args"`` specifies parameters for the associated distribution. These can be given as a number, or as a string that refers to a column in ``data_df``.

Each function in ``scipy.stats`` requires at least two parameters:

 - ``"loc"`` : the location parameter, which shifts the distribution to the right for values >0 and to the left for values <0 (default is 0)
 - ``"scale"`` : the scale parameter (default is 1)

These two parameters are used to convert the input values of :math:`S_T` into a normalized form :math:`x`:

.. math:: x(T,t)=\frac{S_T(T,t) - S_\mathrm{loc}(t)}{S_\mathrm{scale}(t)}

..

Additional parameters are needed for a subset of functions (see the ``scipy.stats`` `documentation <https://docs.scipy.org/doc/scipy/reference/stats.html>`_). For example, the gamma distribution requires a shape parameter ``"a"``, and the beta distribution requires two parameters ``"a"`` and ``"b"``.

+++++++
Example
+++++++

Here is an examples of a SAS specification for two fluxes, ``"Discharge"`` and ``"ET"``, whose SAS function will be modeled as a gamma and beta distribution respectively. These will be converted into piecewise linear approximations with 50 segments.

.. code-block:: json

    {
    "sas_specs": {
        "Discharge": {
            "Discharge SAS fun": {
                "func": "gamma",
                "use": "scipy.stats",
                "args": {
                    "a": 0.62,
                    "scale": 5724.0,
                    "loc": 0.0
                    },
                "nsegment": 50
                }
            },
        "ET": {
            "ET SAS fun": {
                "func": "beta",
                "use": "scipy.stats",
                "args": {
                    "a": 2.31,
                    "b": 0.627,
                    "scale": 1402,
                    "loc": 248
                    },
                "nsegment": 50
                }
            }
        },

    "...": "..."

    }


-------------------------
As a piecewise linear CDF
-------------------------

A SAS function can be specified by supplying the breakpoints of a piecewise linear cumulative distribution (i.e. a piecewise constant PDF).

At minimum, the values of :math:`S_T` (corresponding to breakpoints in the piecewise linear approximation) must be supplied. These are given by the ``"ST"`` key, which must be associated with a list of strictly-increasing non-negative values. Non-increasing or negative values in this list will result in an error. The first value does not need to be zero. The values can be given as a fixed number, or as a string referring to a column in ``data_df``.

Values of the associated cumulative probability can optionally be supplied with the key ``"P"``, which must be associated with a list of non-decreasing numbers between 0 and 1 of the same length as the list in ``"ST"``. The first entry must be ```0`` and the last must be ``1``. Again, the values can be given as a fixed number, or as a string referring to a column in ``data_df``. If ``"P"`` is not supplied it will be assumed that each increment of ``"ST"`` represents an equal increment of probability.

+++++++
Example
+++++++

Here is an example, where storage is given in units of millimeters:

.. code-block:: json

    {
    "sas_specs": {
        "Discharge": {
            "Discharge SAS fun": {
                "ST": [0, 553, "Total Storage"],
                "P" : [ 0, 0.8, 1.0]
                }
            },
        "ET": {
            "ET SAS fun": {
                "ST": [50, 250, 800]
                }
            }
        },

    "...": "..."

    }

This specifies that for ``"Discharge"`` 80% of the discharge should be uniformly selected from the youngest 553 mm, and the remaining 20% from between 553 mm and the (presumably time-varying) value given in ``data_df["Total Storage"]``.

For "ET",  only the "ST" values are provided, so mesas.py will assume the "P" values are uniformly spaced from 0 to 1. Here no ET will be drawn from the youngest 50 mm of storage, 50% will be drawn from between 50 and 250 mm, and 50% will be drawn from between 250 mm and 800 mm.
