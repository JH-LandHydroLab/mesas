"""

================
Module functions
================

This module defines classes representing SAS functions.

Currently there are two classes : :class:`Piecewise` and :class:`Continuous`. The first allows the SAS function to be
specified as a set of breakpoints in a piecewise linear form of the cumulative distribution. The second allows any
distribution specified as a scipy.stats ``rv_continuous`` class to be used as a SAS function. However, it will be
converted into a piecewise form when the model is run.

Calling an instance of either class supplied with values of ST (as an array, list, or number) evaluates
the CDF, and returns corresponding cumulative probabilities. The :func:`Piecewise.inv` method takes cumulative probabilities and
returns values of ST.

"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from scipy.stats import rv_continuous


class _SASFunctionBase:
    """Abstract base class for StorAge Selection (SAS) functions.

    SAS functions map cumulative age-ranked storage (ST) to cumulative
    probability (P), defining how water of different ages is selected
    for discharge. Subclasses must implement ``__call__`` (the CDF) and
    ``inv`` (the inverse CDF).

    Attributes
    ----------
    ST : np.ndarray
        Breakpoint values of cumulative age-ranked storage.
    P : np.ndarray
        Cumulative probabilities corresponding to each ST breakpoint.
    """

    _use: str | None = None

    def __init__(self) -> None:
        pass

    def _make_interpolators(self) -> None:
        """Create linear interpolators for the CDF and inverse CDF.

        Uses ``scipy.interpolate.interp1d`` on the stored ST and P arrays.
        """
        self.interp1d_inv = interp1d(
            self.P,
            self.ST,
            fill_value=(self.ST_min, self.ST_max),
            kind="linear",
            copy=False,
            bounds_error=False,
            assume_sorted=True,
        )
        self.interp1d = interp1d(
            self.ST, self.P, fill_value=(0.0, 1.0), kind="linear", copy=False, bounds_error=False, assume_sorted=True
        )

    def __getitem__(self, i: Any) -> _SASFunctionBase:
        return self

    def __setitem__(self, i: Any) -> None:
        raise TypeError("You're not allowed to modify the SAS function for a specific index")

    # Next we define a number of properties. These act like attributes, but special functions
    # are called when the attributes are queried or assigned to

    # @property
    # def has_params(self):
    #    return self._has_params

    @property
    def argsS(self) -> np.ndarray | None:
        """np.ndarray or None : Storage-axis parameters of the SAS function."""
        return None

    @argsS.setter
    def argsS(self, new_args: np.ndarray) -> None:
        raise NotImplementedError("Method for setting args directly has not been defined")

    @property
    def argsP(self) -> np.ndarray | None:
        """np.ndarray or None : Probability-axis parameters of the SAS function."""
        return None

    @argsP.setter
    def argsP(self, new_args: np.ndarray) -> None:
        raise NotImplementedError("Method for setting args directly has not been defined")

    @property
    def ST(self) -> np.ndarray:
        """np.ndarray : Cumulative age-ranked storage breakpoints."""
        return self._ST

    @ST.setter
    def ST(self, new_ST: np.ndarray) -> None:
        raise NotImplementedError("Method for setting ST directly has not been defined")

    @property
    def P(self) -> np.ndarray:
        """np.ndarray : Cumulative probabilities at each ST breakpoint."""
        return self._P

    @P.setter
    def P(self, new_P: np.ndarray) -> None:
        raise NotImplementedError("Method for setting P directly has not been defined")

    @property
    def parameter_list(self) -> np.ndarray:
        """np.ndarray : Segment-length parameterisation of the SAS function."""
        return self._parameter_list

    @parameter_list.setter
    def parameter_list(self, new_parameter_list: np.ndarray) -> None:
        raise NotImplementedError("Method for setting parameters directly has not been defined")

    def subdivided_copy(self, *args: Any, **kwargs: Any) -> Piecewise:
        """Return a copy with one segment subdivided.

        Raises
        ------
        TypeError
            Always; must be overridden by subclasses that support subdivision.
        """
        raise TypeError("Cannot subdivide this type of function")

    def _convert_segment_list_to_ST(self, seglst: np.ndarray) -> np.ndarray:
        """Convert a segment-length array to cumulative ST breakpoints.

        Parameters
        ----------
        seglst : np.ndarray
            Array whose first element is ST_min and subsequent elements are
            segment lengths.

        Returns
        -------
        np.ndarray
            Cumulative sum giving ST breakpoint positions.
        """
        return np.cumsum(seglst)

    def _convert_ST_to_segment_list(self, ST: np.ndarray) -> np.ndarray:
        """Convert cumulative ST breakpoints to a segment-length array.

        Parameters
        ----------
        ST : np.ndarray
            Monotonically increasing ST breakpoints.

        Returns
        -------
        np.ndarray
            Array whose first element is ST[0] and subsequent elements are
            the differences between consecutive ST values.
        """
        return np.r_[ST[0], np.diff(ST)]

    def inv(self, P: np.ndarray) -> np.ndarray:
        """Evaluate the inverse CDF (quantile function) of the SAS function.

        Parameters
        ----------
        P : np.ndarray
            Cumulative probabilities in [0, 1].

        Returns
        -------
        np.ndarray
            Corresponding cumulative age-ranked storage values.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Inverse CDF method not been defined")

    def __call__(self, ST: np.ndarray) -> np.ndarray:
        """Evaluate the CDF of the SAS function.

        Parameters
        ----------
        ST : np.ndarray
            Cumulative age-ranked storage values.

        Returns
        -------
        np.ndarray
            Corresponding cumulative probabilities.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("CDF method not been defined")

    def __repr__(self) -> str:
        """Return a repr of the SAS function"""
        raise NotImplementedError("__repr__ not been defined")

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> list[Line2D]:
        """Plot the SAS function CDF.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axes to plot on. If ``None``, a new figure and axes are created.
        **kwargs
            Additional keyword arguments forwarded to ``ax.plot``.

        Returns
        -------
        list of matplotlib.lines.Line2D
            The plotted line(s).
        """
        if ax is None:
            fig, ax = plt.subplots()
        return ax.plot(self.ST, self.P, **kwargs)

    def get_jacobian(
        self,
        dCdSj: np.ndarray,
        index: np.ndarray | None = None,
        mode: str = "segment",
        logtransform: bool = True,
    ) -> np.ndarray:
        """Compute a limited Jacobian of model predictions.

        Parameters
        ----------
        dCdSj : np.ndarray
            Jacobian of concentrations with respect to ST breakpoints,
            produced by the Fortran solver.
        index : np.ndarray or None, optional
            Timestep indices to include. Defaults to all timesteps.
        mode : str, optional
            ``'segment'`` (default) returns derivatives with respect to
            segment lengths; ``'endpoint'`` with respect to ST values.
        logtransform : bool, optional
            If True and ``mode='segment'``, return derivatives with respect
            to log-transformed segment lengths. Default is True.

        Returns
        -------
        np.ndarray
            The Jacobian matrix.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("get_jacobian not been defined")


class Piecewise(_SASFunctionBase):
    """Piecewise-linear SAS function defined by ST breakpoints.

    The CDF is represented as a piecewise-linear curve through a set of
    ``(ST, P)`` pairs.  Linear interpolation is used to evaluate the CDF
    and its inverse.

    You can define a piecewise sas function in several different ways. The default is a single segment from 0 to 1:

        >>>sas_fun = Piecewise()
        >>>sas_fun([0.1, 0.3, 0.9])
        array([0.1, 0.3, 0.9])

    The range of the single segment can be modified by providing ``ST_min`` and ``ST_max`` keywords

        >>>sas_fun = Piecewise(ST_min=100, ST_max=110)
        >>>sas_fun([99, 101, 103, 109, 111])
        array([0. , 0.1, 0.3, 0.9, 1. ])

    The (ST,P) pairs defining the CDF are given by the ``ST`` and ``P`` attributes

        >>>sas_fun.ST
        array([100.        , 103.00391164, 104.17022005,
        110.        ])
        >>>sas_fun.P
        array([0.        , 0.33333333, 0.66666667, 1.        ])

    The segments defining the function are stored in the ``segment_list``
    attribute as a numpy array:

        >>>seglst = sas_fun.segment_list
        array([100.        ,   3.00391164,   1.1663084 ,   5.82977995])

    The first item in the array is ``ST_min``, and each following value is the interval
    along the ST axis to the end of each segment.

    Once created, the values of ``segment_list`` or ``ST`` can be modified and the other will update too.
    Values of ``P`` can also be changed.

    Parameters
    ----------
    ST : array_like or None, optional
        Explicit ST breakpoints. Must be length > 1 with ``ST[0] >= 0``.
    P : array_like or None, optional
        Cumulative probabilities at each breakpoint. Must start at 0, end
        at 1, and be non-decreasing.  Length must match ``ST``.
    nsegment : int, optional
        Number of segments when ``ST`` is not provided. Default is 1.
    ST_max : float, optional
        Upper bound of the SAS function support. Default is 1.0.
    ST_min : float, optional
        Lower bound of the SAS function support. Default is 0.0.
    auto : str, optional
        How to auto-generate breakpoints: ``'uniform'`` (default) for
        equally spaced, ``'random'`` for random recursive splitting.
    """

    def __init__(
        self,
        ST: np.ndarray | list[float] | None = None,
        P: np.ndarray | list[float] | None = None,
        nsegment: int = 1,
        ST_max: float = 1.0,
        ST_min: float = 0.0,
        auto: str = "uniform",
    ) -> None:
        self.ST_min = float(ST_min)
        self.ST_max = float(ST_max)
        # self._has_params = False

        # Note that the variables _ST, _P and _parameter_list are assigned to below
        # instead of their corresponding properties. This is to avoid triggering the _make_interpolators function
        # until the end

        if ST is not None:
            # Use the given ST values.
            # Validation must match the ST property setter: the solver's
            # piecewise evaluation silently drops CDF jumps at repeated
            # breakpoints, and interp1d(assume_sorted=True) returns garbage
            # for decreasing values.
            if len(ST) <= 1:
                raise ValueError(f"ST must have at least 2 values, got {len(ST)}")
            if ST[0] < 0:
                raise ValueError(f"ST[0] must be >= 0, got {ST[0]}")
            if not np.all(np.diff(ST) > 0):
                raise ValueError(f"ST values must be strictly increasing. Got ST = {np.asarray(ST)}")
            self.nsegment = len(ST) - 1
            self._ST = np.array(ST, dtype=float)
            self._parameter_list = self._convert_ST_to_segment_list(self._ST)

        else:
            self.nsegment = nsegment

            if auto == "uniform":
                self._ST = np.linspace(self.ST_min, self.ST_max, self.nsegment + 1)
                self._parameter_list = self._convert_ST_to_segment_list(self._ST)
            elif auto == "random":
                # Generate a random function
                # Note this should probably be encapsulated in a function
                self._ST = np.zeros(self.nsegment + 1)
                ST_scaled = np.zeros(self.nsegment + 1)
                ST_scaled[-1] = 1.0
                for i in range(self.nsegment - 1):
                    ST_scaled[self.nsegment - i - 1] = np.random.uniform(0, ST_scaled[self.nsegment - i], 1)

                self._ST[:] = self.ST_min + (self.ST_max - self.ST_min) * ST_scaled
                self._parameter_list = self._convert_ST_to_segment_list(self._ST)

        # Make a list of probabilities P
        if P is not None:
            # Use the supplied values
            if P[0] != 0:
                raise ValueError(f"P must start at 0, got P[0]={P[0]}")
            if P[-1] != 1:
                raise ValueError(f"P must end at 1, got P[-1]={P[-1]}")
            if not np.all(np.diff(P) >= 0):
                raise ValueError("P must be non-decreasing")
            if len(P) != len(self._ST):
                raise ValueError(f"P length ({len(P)}) must match ST length ({len(self._ST)})")
            self._P = np.r_[P]

        else:
            # Make P equally-spaced
            self._P = np.linspace(0, 1, self.nsegment + 1, endpoint=True)

        # self._has_params = True
        # call this to make the interpolation functions
        self._make_interpolators()

    @property
    def argsS(self) -> np.ndarray:
        """np.ndarray : ST breakpoints (alias for ``ST``)."""
        return self._ST

    @property
    def argsP(self) -> np.ndarray:
        """np.ndarray : Probability breakpoints (alias for ``P``)."""
        return self._P

    @_SASFunctionBase.ST.setter
    def ST(self, new_ST: np.ndarray) -> None:
        if len(new_ST) <= 1:
            raise ValueError(f"ST must have at least 2 values, got {len(new_ST)}")
        if new_ST[0] < 0:
            raise ValueError(f"ST[0] must be >= 0, got {new_ST[0]}")
        if not np.all(np.diff(new_ST) > 0):
            raise ValueError(
                f"ST values must be strictly increasing. Got ST = {new_ST}. "
                "If ST values are not distinct, try changing "
                "'ST_largest_segment' and/or 'ST_smallest_segment'."
            )
        self._ST = new_ST
        self.ST_min = self._ST[0]
        self.ST_max = self._ST[-1]
        self._parameter_list = self._convert_ST_to_segment_list(self._ST)
        self._make_interpolators()

    @_SASFunctionBase.P.setter
    def P(self, new_P: np.ndarray) -> None:
        if new_P[0] != 0:
            raise ValueError(f"P must start at 0, got P[0]={new_P[0]}")
        if new_P[-1] != 1:
            raise ValueError(f"P must end at 1, got P[-1]={new_P[-1]}")
        if len(new_P) != len(self._ST):
            raise ValueError(f"P length ({len(new_P)}) must match ST length ({len(self._ST)})")
        self._P = new_P
        self._make_interpolators()

    @_SASFunctionBase.parameter_list.setter
    def parameter_list(self, new_parameter_list: np.ndarray) -> None:
        self._parameter_list = new_parameter_list
        self.ST = self._convert_segment_list_to_ST(self._parameter_list)

    def subdivided_copy(self, segment: int, split_frac: float = 0.5) -> Piecewise:
        """Return a new Piecewise instance with one segment split in two.

        Parameters
        ----------
        segment : int
            Zero-based index of the segment to split.
        split_frac : float, optional
            Fraction along the segment where the split occurs. Default 0.5.

        Returns
        -------
        Piecewise
            A new SAS function with ``nsegment + 1`` segments.
        """
        if segment >= self.nsegment:
            raise ValueError(f"segment index {segment} out of range (nsegment={self.nsegment})")
        P1 = self.P[segment]
        P2 = self.P[segment + 1]
        ST1 = self.ST[segment]
        ST2 = self.ST[segment + 1]
        P_new = P1 + split_frac * (P2 - P1)
        ST_new = ST1 + split_frac * (ST2 - ST1)
        P = np.insert(self.P, segment + 1, P_new)
        ST = np.insert(self.ST, segment + 1, ST_new)
        return Piecewise(ST=ST, P=P)

    def inv(self, P: np.ndarray) -> np.ndarray:
        """Evaluate the inverse CDF of the SAS function.

        Parameters
        ----------
        P : array_like
            Cumulative probabilities (scalar, list, or array).

        Returns
        -------
        np.ndarray
            Corresponding ST values, same shape as input.
        """
        P_arr = np.array(P)
        P_ravel = P_arr.ravel()
        return self.interp1d_inv(P_ravel).reshape(P_arr.shape)

    def __call__(self, ST: np.ndarray) -> np.ndarray:
        """Evaluate the CDF of the SAS function.

        Parameters
        ----------
        ST : array_like
            Age-ranked storage values (scalar, list, or array).

        Returns
        -------
        np.ndarray
            Corresponding cumulative probabilities, same shape as input.
        """
        ST_arr = np.array(ST)
        ST_ravel = ST_arr.ravel()
        return self.interp1d(ST_ravel).reshape(ST_arr.shape)

    def __repr__(self) -> str:
        """Return a repr of the SAS function"""
        if not hasattr(self, "nsegment"):
            return "Not initialized"
        result = "        ST: "
        for i in range(self.nsegment + 1):
            result += "{ST:<10.4}  ".format(ST=self.ST[i])
        result += "\n"
        result += "        P : "
        for i in range(self.nsegment + 1):
            result += "{P:<10.4}  ".format(P=self.P[i])
        result += "\n"
        return result

    def get_jacobian(
        self,
        dCdSj: np.ndarray,
        index: np.ndarray | None = None,
        mode: str = "segment",
        logtransform: bool = True,
    ) -> np.ndarray:
        """Compute a limited Jacobian of predicted concentrations.

        If :math:`\\vec{y} = f(\\vec{x})` the Jacobian is a matrix where each i, j entry is

        .. math:

            J_{i,j}=\\frac{\\partial y_i}{\\partial x_j}

        ..

        Here, the :math:`y_i` values are the predicted concentrations at each timestep,
        and the :math:`x_j` are either the lengths of each piecewise segment along with ST_min (default),
        or the values of ST at the endpoints of each segment (with `mode='endpoint'`).

        The Jacobian is limited as it represents the sensitivity of the model predictions of output
        concentrations at each timestep to variations in the sas function at that timestep, and neglects
        the cumulative effects that changing the sas function would have on the state variables (ST and MS).

        Parameters
        ----------
        dCdSj : np.ndarray
            Jacobian of concentrations w.r.t. ST breakpoints from the
            Fortran solver, shape ``(n_timesteps, nsegment + 1)``.
        index : np.ndarray or None, optional
            Timestep indices to include. Defaults to all timesteps.
        mode : str, optional
            ``'endpoint'`` returns derivatives w.r.t. ST values;
            ``'segment'`` (default) w.r.t. segment lengths.
        logtransform : bool, optional
            If True and ``mode='segment'``, return derivatives w.r.t.
            log-transformed segment lengths. Default is True.

        Returns
        -------
        np.ndarray
            The Jacobian matrix.
        """

        if index is None:
            index = np.arange(dCdSj.shape[0])

        J_S = dCdSj[index, :]

        if mode == "endpoint":
            return J_S

        # To get the derivative with respect to the segment length, we add up the derivative w.r.t. the
        # endpoints that would be displaced by varying that segment
        endpoint_to_segment = np.triu(np.ones(self.nsegment + 1), k=0)
        J_seg = np.dot(endpoint_to_segment, J_S.T).T

        if logtransform:
            J_seglog = J_seg * self._parameter_list
            return J_seglog
        else:
            return J_seg


class kumaraswamy_gen(rv_continuous):
    """Kumaraswamy distribution as a scipy ``rv_continuous`` subclass.

    The Kumaraswamy distribution is similar to the Beta distribution but
    has a closed-form CDF:

    .. math::

        F(x; a, b) = 1 - (1 - x^a)^b, \\quad x \\in [0, 1]

    Parameters
    ----------
    a : float
        First shape parameter (> 0).
    b : float
        Second shape parameter (> 0).
    """

    def _cdf(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return 1 - (1 - x**a) ** b

    def _pdf(self, x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * b * x ** (a - 1) * (1 - x**a) ** (b - 1)


kumaraswamy = kumaraswamy_gen(a=0.0, b=1.0, name="kumaraswamy")


class Continuous(_SASFunctionBase):
    """SAS function backed by a continuous probability distribution.

    Wraps a ``scipy.stats`` continuous distribution (or one of the built-in
    distributions: gamma, beta, kumaraswamy) as a SAS function.  When used
    with ``use='scipy.stats'``, the continuous CDF is also discretised into
    a piecewise-linear lookup table for use by the Fortran solver.

    Parameters
    ----------
    use : str
        Either ``'builtin'`` for optimised built-in distributions or
        ``'scipy.stats'`` for arbitrary ``rv_continuous`` distributions.
    func : str or rv_continuous
        Distribution name (``'gamma'``, ``'beta'``, ``'kumaraswamy'``) when
        ``use='builtin'``, or a ``scipy.stats.rv_continuous`` instance /
        name string when ``use='scipy.stats'``.
    func_kwargs : dict[str, Any]
        Keyword arguments passed to the distribution to create a frozen
        instance (e.g. ``{'a': 2, 'loc': 0, 'scale': 100}``).
    P : array_like or None, optional
        Probability breakpoints for the piecewise lookup table.  Only used
        when ``use='scipy.stats'``.
    nsegment : int, optional
        Number of segments in the lookup table when ``P`` is not given.
        Default is 25.
    ST_max : float, optional
        Upper storage bound for the lookup table. Default is ``float('inf')``.

    Attributes
    ----------
    func : scipy.stats.rv_frozen
        The frozen distribution instance.
    """

    _builtinfuncdict: dict[str, int] = {"gamma": 1, "beta": 2, "kumaraswamy": 3}

    def __init__(
        self,
        use: str,
        func: str | rv_continuous,
        func_kwargs: dict[str, Any],
        P: np.ndarray | list[float] | None = None,
        nsegment: int = 25,
        ST_max: float = 1.797693134862315e308,
    ) -> None:
        if use == "builtin" and func in self._builtinfuncdict.keys():
            try:
                self._func = getattr(scipy.stats, func)
            except AttributeError:
                if func == "kumaraswamy":
                    self._func = kumaraswamy
                else:
                    self._func = None
            self._builtinfunctype = self._builtinfuncdict[func]
        elif use == "scipy.stats" and isinstance(func, rv_continuous):
            self._func = func
            self._builtinfunctype = None
        elif use == "scipy.stats" and isinstance(func, str):
            self._func = getattr(scipy.stats, func)
            self._builtinfunctype = None
        else:
            raise Exception("'use' keyword must be either 'builtin' or 'scipy.stats'")
        self._use = use

        self._frozen_func = self._func(**func_kwargs)
        if self._use == "builtin":
            if func == "gamma":
                self._argsS = []
                self._argsS += [func_kwargs["loc"], func_kwargs["scale"]]
                self._argsS += [func_kwargs["a"]]
            if func == "beta":
                self._argsS = []
                self._argsS += [func_kwargs["loc"], func_kwargs["scale"]]
                self._argsS += [func_kwargs["a"], func_kwargs["b"]]
            if func == "kumaraswamy":
                self._argsS = []
                self._argsS += [func_kwargs["loc"], func_kwargs["scale"]]
                self._argsS += [func_kwargs["a"], func_kwargs["b"]]
            self._argsP = np.ones_like(self._argsS) * np.nan
        elif self._use == "scipy.stats":
            # generate a piecewise approximation
            self.ST_max = float(ST_max)
            # Make a list of probabilities P
            if P is not None:
                # Use the supplied values
                self.nsegment = len(P) - 1
                self.P = np.r_[P]
            else:
                # Make P equally-spaced
                self.nsegment = nsegment
                self.P = np.linspace(0, 1, self.nsegment + 1, endpoint=True)
            self._argsS = self._ST
            self._argsP = self._P

    @property
    def argsS(self) -> list[float] | np.ndarray:
        """list or np.ndarray : Storage-axis distribution parameters."""
        return self._argsS

    @property
    def argsP(self) -> np.ndarray:
        """np.ndarray : Probability-axis distribution parameters."""
        return self._argsP

    @property
    def func(self) -> scipy.stats.rv_frozen:
        """scipy.stats.rv_frozen : The frozen distribution instance."""
        return self._frozen_func

    @func.setter
    def func(self, new_func: Any) -> None:
        raise ValueError("Function type cannot be changed")

    @_SASFunctionBase.ST.setter
    def ST(self, new_ST: np.ndarray) -> None:
        if len(new_ST) <= 1:
            raise ValueError(f"ST must have at least 2 values, got {len(new_ST)}")
        if new_ST[0] < 0:
            raise ValueError(f"ST[0] must be >= 0, got {new_ST[0]}")
        if not np.all(np.diff(new_ST) > 0):
            raise ValueError(f"ST values must be strictly increasing. Got ST = {new_ST}")
        if new_ST[-1] > self.ST_max:
            self._ST = new_ST
            self.ST_max = self._ST[-1]
        else:
            self._ST = np.r_[new_ST, self.ST_max]
        self._P = self.func.cdf(self._ST)

    @_SASFunctionBase.P.setter
    def P(self, new_P: np.ndarray) -> None:
        if new_P[0] != 0:
            raise ValueError(f"P must start at 0, got P[0]={new_P[0]}")
        if new_P[-1] != 1:
            raise ValueError(f"P must end at 1, got P[-1]={new_P[-1]}")
        if not np.all(np.diff(new_P) >= 0):
            raise ValueError("P must be non-decreasing")
        self._P = new_P
        # if self._has_params:
        self._ST = self.func.ppf(self._P)

    def inv(self, P: np.ndarray) -> np.ndarray:
        """Evaluate the inverse CDF of the SAS function.

        Parameters
        ----------
        P : array_like
            Cumulative probabilities.

        Returns
        -------
        np.ndarray
            Corresponding ST values.
        """
        return self.func.ppf(P)

    def __call__(self, ST: np.ndarray) -> np.ndarray:
        """Evaluate the CDF of the SAS function.

        Parameters
        ----------
        ST : array_like
            Cumulative age-ranked storage values.

        Returns
        -------
        np.ndarray
            Corresponding cumulative probabilities.
        """
        return self.func.cdf(ST)

    def __repr__(self) -> str:
        """Return a repr of the SAS function"""
        result = f"       {self._func.name} distribution\n"
        result += f"        parameters: {self._frozen_func.args}\n"
        result += "        Lookup table version:\n"
        result += "          ST: "
        for i in range(self.nsegment + 1):
            result += "{ST:<10.4}  ".format(ST=self.ST[i])
        result += "\n"
        result += "          P : "
        for i in range(self.nsegment + 1):
            result += "{P:<10.4}  ".format(P=self.P[i])
        result += "\n"
        return result

    def get_jacobian(
        self,
        dCdSj: np.ndarray,
        index: np.ndarray | None = None,
        mode: str = "segment",
        logtransform: bool = True,
    ) -> np.ndarray:
        """Compute a limited Jacobian of predicted concentrations.

        Parameters
        ----------
        dCdSj : np.ndarray
            Jacobian from the Fortran solver.
        index : np.ndarray or None, optional
            Timestep indices. Defaults to all.
        mode : str, optional
            ``'segment'`` or ``'endpoint'``.
        logtransform : bool, optional
            Log-transform segment derivatives. Default True.

        Returns
        -------
        np.ndarray
            The Jacobian matrix.

        Raises
        ------
        NotImplementedError
            Always raised; not yet implemented for continuous functions.
        """
        raise NotImplementedError("get_jacobian not been defined")
