
============
Installation
============

From PyPI or conda-forge
========================

The simplest way to install mesas is with pip::

    pip install mesas

Or using Conda::

    conda install -c conda-forge mesas

Either method will install all required dependencies (NumPy, SciPy, pandas,
Numba, matplotlib). No Fortran compiler is needed: as of v2, mesas is pure
Python, with the numerical solver JIT-compiled by Numba.

.. note::

   The legacy Fortran-based v1.x remains available by pinning the version,
   e.g. ``conda install -c conda-forge mesas=1.20240418``.

From source
===========

To install the latest development version from GitHub::

    git clone https://github.com/JH-LandHydroLab/mesas.git
    cd mesas
    pip install -e .

.. note::

   No Fortran compiler is needed. The numerical solver uses
   `Numba <https://numba.pydata.org/>`_ for JIT compilation of
   pure Python code.

Requirements
============

- Python >= 3.10
- NumPy >= 1.22
- SciPy
- pandas
- Numba
- matplotlib
