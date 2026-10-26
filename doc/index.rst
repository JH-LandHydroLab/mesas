.. MESAS documentation master file, created by sphinx-quickstart on Thu Sep 12 19:57:56 2019. You can adapt this file completely to your liking, but it should at least contain the root `toctree` directive.

=====
MESAS
=====
Multiscale Estimation of StorAge Selection functions


What are StorAge Selection functions?
-------------------------------------

StorAge Selection is a theoretical framework for modeling transport through control volumes. It is appropriate if you are interested a system that can be treated as a single control volume (or a collection of such), and wish to make minimal assumptions about the internal organization of the transport. SAS assumes that the material leaving a system is some combination of the material that entered at earlier times. This can be useful for constructing simple models of very complicated flow systems, and for inferring the emergent transport properties of a system from tracer data.

SAS is a very general framework -- at heart, it is composed of two parts: a way of keeping track of changes in the age structure of the 'population' of fluid parcels in a system, and a rule (or rules) that determines how that age structure changes. By using tracers to investigate what rule best mimics the behaviour of a given system, we can learn important information about that system -- most directly, something about the volume of tracer-transporting fluid that is actively 'turning over'. SAS is a generalization of commonly-used transit time theory to the case where fluxes vary in time, and there may be more than one outflow (e.g. discharge and ET).

For more information see the free HydroLearn course: `Tracers and transit times in time-variable hydrologic systems: A gentle introduction to the StorAge Selection (SAS) approach <https://edx.hydrolearn.org/courses/course-v1:JHU+570.412+Sp2020>`_

What is mesas.py?
-----------------

The package will eventually include a number of component tools for building SAS models and using them to analyze tracer data.

Currently the main repository codebase includes three submodules:

mesas.sas
     The main submodule for building and running SAS models
mesas.utils
     Tools for manipulating SAS model input data and results
mesas.utils.vis
     Visualization of SAS model results

I hope to expand these in the future, and welcome contributors through the `GitHub repository <https://github.com/JH-LandHydroLab/mesas>`_

Support, bugs, suggestions, etc.
--------------------------------
If you find a bug please open an issue on GitHub here: https://github.com/JH-LandHydroLab/mesas/issues

You can also use the issue tracer to make suggestions to improve the code.

If you need further help using the code or interpreting the results, please get in touch: charman1 at jhu dot edu


Table of contents
-----------------
.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   concepts

.. toctree::
   :maxdepth: 2
   :caption: User guide

   inputs
   config
   sasspec
   solspec
   options
   results

.. toctree::
   :maxdepth: 2
   :caption: Reference

   functions.rst
   specs.rst
   model.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
