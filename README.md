# mesas.py

StorAge Selection is a theoretical framework for modeling transport through control volumes. It is appropriate if you are interested a system that can be treated as a single control volume (or a collection of such), and wish to make minimal assumptions about the internal organization of the transport. SAS assumes that the material leaving a system is some combination of the material that entered at earlier times. This can be useful for constructing simple models of very complicated flow systems, and for inferring the emergent transport properties of a system from tracer data.

For more information see the free HydroLearn course: [Tracers and transit times in time-variable hydrologic systems: A gentle introduction to the StorAge Selection (SAS) approach](https://edx.hydrolearn.org/courses/course-v1:JHU+570.412+Sp2020)

## Installation

### Stable release (v1.0, Fortran)

The current stable release on conda-forge is the older Fortran-based v1.0:

    conda install -c conda-forge mesas

This installs any additional dependencies at the same time.

### v2.0 (pure Python, in beta)

Version 2.0 is a pure Python reimplementation whose numerical solver is
JIT-compiled with [Numba](https://numba.pydata.org/) -- no Fortran compiler
needed. It is currently in beta. Install the latest development version from
source:

    git clone https://github.com/JH-LandHydroLab/mesas.git
    cd mesas
    pip install -e .

Further instructions can be found here: https://mesas.readthedocs.io/en/latest/installation.html

## Documentation

Documentation for the code is available here: https://mesas.readthedocs.io/en/latest/

## Citation

If you use mesas.py in your research, please cite:

Harman, C. J. and Xu Fei, E.: mesas.py v1.0: a flexible Python package for modeling solute transport and transit times using StorAge Selection functions, Geosci. Model Dev., 17, 477-495, https://doi.org/10.5194/gmd-17-477-2024, 2024.

[![DOI](https://zenodo.org/badge/183813641.svg)](https://zenodo.org/badge/latestdoi/183813641)
