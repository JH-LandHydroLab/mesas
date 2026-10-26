.. _concepts:

========
Concepts
========

This page introduces the StorAge Selection (SAS) framework at a level
appropriate for users of the package. For the full mathematical
treatment, see `Harman (2015) <https://doi.org/10.1002/2014WR015707>`_
and `Benettin et al. (2022) <https://doi.org/10.5194/gmd-15-3881-2022>`_
and `Harman and Xu Fei (2024) <https://gmd.copernicus.org/articles/17/477/2024/>`_.

What is StorAge Selection?
==========================

StorAge Selection (SAS) is a framework for modeling transport through a
control volume — a catchment, a soil column, a lake, or any system
where fluid enters, is stored, and eventually leaves.

The key idea is simple: **at each moment in time, the water leaving a
system is a mixture of water that entered at different times in the
past.** The SAS function describes *which ages of water are
preferentially selected for discharge*.

Age-ranked storage
==================

The foundation of SAS is the concept of **age-ranked storage**. At any
time :math:`t`, the water in storage can be sorted by how long it has been
in the system. We define:

.. math::

    S_T(T, t) = \int_0^T s_T(\tau, t)\, d\tau

where :math:`s_T(\tau, t)` is the density of water of age :math:`\tau`
in storage at time :math:`t`, and :math:`S_T(T, t)` is the cumulative
storage up to age :math:`T`. The total storage is
:math:`S(t) = S_T(\infty, t)`.

The SAS function
================

The **SAS function** :math:`\Omega_Q(S_T, t)` is a cumulative
distribution function that maps age-ranked storage to cumulative
probability:

.. math::

    \Omega_Q(S_T, t) = P(\text{discharged water has age-ranked storage} \leq S_T)

When :math:`\Omega_Q` is a straight line from 0 to :math:`S(t)`, all
ages of water are equally likely to be discharged — this is the
**well-mixed** case. When :math:`\Omega_Q` rises steeply at small
:math:`S_T`, young water is preferentially discharged. When it rises
steeply near :math:`S(t)`, old water is preferred.

The backward transit time distribution for outflow :math:`Q` is:

.. math::

    p_Q(T, t) = \frac{\partial \Omega_Q}{\partial S_T} \cdot s_T(T, t)

Types of SAS functions
======================

MESAS supports several types of SAS functions:

**Piecewise-linear**
    Defined by breakpoints in :math:`(S_T, P)` space. The simplest is a
    uniform distribution: ``"ST": [0, S_max]``. Adding more breakpoints
    creates more complex shapes.

**Gamma distribution**
    A flexible one-parameter family. The shape parameter :math:`a`
    controls whether young water (:math:`a < 1`) or intermediate-aged
    water (:math:`a > 1`) is preferentially discharged.
    Specified with ``"func": "gamma"``.

**Beta distribution**
    Defined over a finite interval, useful when total storage volume is
    known. Has two shape parameters :math:`a` and :math:`b`.
    Specified with ``"func": "beta"``.

**Kumaraswamy distribution**
    Similar to the beta but with a closed-form CDF, making it
    computationally efficient.
    Specified with ``"func": "kumaraswamy"``.

All distribution-based SAS functions accept ``loc`` and ``scale``
parameters that shift and stretch the distribution along the
:math:`S_T` axis.

Solute transport
================

Once the water fluxes are solved, solute transport follows
directly. The concentration of water of a given age in storage
determines the concentration of the outflow:

.. math::

    C_Q(t) = \int_0^\infty \frac{m_T(T,t)}{s_T(T,t)} \, p_Q(T,t) \, dT

where :math:`m_T(T,t)` is the age-ranked solute mass in storage.

MESAS also supports:

- **First-order reactions** via ``k1`` and ``C_eq`` parameters
- **Evapoconcentration** via the ``alpha`` parameter
- **Multiple solutes** tracked simultaneously

Time-varying parameters
========================

Any SAS function or solute parameter can be made time-varying by
setting its value to a string that names a column in the input
DataFrame. For example, setting ``"scale": "S_0"`` in a gamma
specification causes the scale parameter to be read from the ``"S_0"``
column at each timestep.

Further reading
===============

- `Harman (2015) <https://doi.org/10.1002/2014WR015707>`_ — Original SAS theory paper
- `Harman and Xu Fei (2024) <https://gmd.copernicus.org/articles/17/477/2024/>`_ — Unified framework for age and transit time distributions
- `Benettin et al. (2022) <https://doi.org/10.5194/gmd-15-3881-2022>`_ — MESAS model description paper (GMD)
- `HydroLearn course <https://edx.hydrolearn.org/courses/course-v1:JHU+570.412+Sp2020>`_ — Free online course on SAS theory
