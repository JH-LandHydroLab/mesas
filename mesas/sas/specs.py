from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.interpolate import interp1d

from mesas.sas.functions import Continuous, Piecewise


class SAS_Spec:
    """Blended SAS (StorAge Selection) function specification.

    Represents a time-varying SAS function composed of one or more weighted
    ``Component`` objects. Each component defines a piecewise-linear or
    continuous CDF over cumulative storage (ST). Components are blended at
    each timestep using their weight time series to produce a single CDF
    and inverse CDF.

    Parameters
    ----------
    spec : dict[str, dict]
        Nested dictionary keyed by component label. Each value is a
        component specification dictionary (passed to ``Component``).
    data_df : pd.DataFrame
        Input time-series data containing any columns referenced by
        component specifications (e.g. weight columns, parameter columns).

    Attributes
    ----------
    components : OrderedDict[str, Component]
        Ordered mapping of component labels to ``Component`` instances.
    N : int
        Number of timesteps (set after ``make_spec_ts`` is called).
    ST : np.ndarray | None
        Padded (N, max_breakpoints) matrix of ST breakpoints, or ``None``
        before ``make_spec_ts`` is called.
    P : np.ndarray | None
        Corresponding padded probability matrix.
    """

    def __init__(self, spec: dict[str, dict], data_df: pd.DataFrame) -> None:
        self.components: OrderedDict[str, Component] = OrderedDict()
        is_only_component = len(spec) == 1
        for label, component_spec in spec.items():
            self.components[label] = Component(label, component_spec, data_df, is_only_component)
        self._componentorder = list(self.components.keys())
        self._comp2learn_componentorder = self._componentorder
        self._data_df = data_df

    def make_spec_ts(self, data_df: pd.DataFrame | None = None) -> None:
        """Build per-timestep CDF and inverse-CDF interpolators.

        Blends all component SAS functions at each timestep using their
        respective weights and constructs piecewise-linear interpolation
        objects for the forward and inverse CDF.

        Parameters
        ----------
        data_df : pd.DataFrame or None, optional
            Override data frame. If ``None``, uses the data frame passed at
            construction.
        """
        if data_df is None:
            data_df = self._data_df

        # Trigger the method to create interpolators
        self.N = len(data_df)
        self._interp1d_inv: list[interp1d] = list(range(self.N))
        self._interp1d: list[interp1d] = list(range(self.N))
        self.ST_lists: list[np.ndarray] = list(range(self.N))
        self.P_lists: list[np.ndarray] = list(range(self.N))
        self.ST = None
        self.P = None
        for i in range(self.N):
            self.ST_lists[i] = np.sort(
                np.unique(np.concatenate([component.sas_fun[i].ST for label, component in self.components.items()]))
            )
            self.P_lists[i] = np.zeros_like(self.ST_lists[i])
            for label, component in self.components.items():
                self.P_lists[i] += component.sas_fun[i](self.ST_lists[i]) * component.weights.values[i]
            ST_min = self.ST_lists[i][0]
            ST_max = self.ST_lists[i][-1]
            self._interp1d_inv[i] = interp1d(
                self.P_lists[i],
                self.ST_lists[i],
                fill_value=(ST_min, ST_max),
                kind="linear",
                copy=False,
                bounds_error=False,
                assume_sorted=True,
            )
            self._interp1d[i] = interp1d(
                self.ST_lists[i],
                self.P_lists[i],
                fill_value=(0.0, 1.0),
                kind="linear",
                copy=False,
                bounds_error=False,
                assume_sorted=True,
            )
        # create matricies of ST, P padded with extra values on the left
        maxj = np.max([len(ST) for ST in self.ST_lists])
        self.ST = np.zeros((maxj, self.N))
        self.P = np.zeros_like(self.ST)
        for i in range(self.N):
            Nj = len(self.ST_lists[i])
            self.ST[maxj - Nj : maxj, i] = self.ST_lists[i]
            self.P[maxj - Nj : maxj, i] = self.P_lists[i]
            if Nj < maxj:
                self.ST[: maxj - Nj, i] = -1 - np.arange(maxj - Nj)
                self.P[: maxj - Nj, i] = np.zeros(maxj - Nj)
        self.ST = self.ST.T
        self.P = self.P.T

    def __call__(self, ST: np.ndarray | float, i: int) -> np.ndarray:
        """Evaluate the blended CDF at a given timestep.

        Parameters
        ----------
        ST : np.ndarray or float
            Age-ranked cumulative storage values.
        i : int
            Zero-based timestep index.

        Returns
        -------
        np.ndarray
            Cumulative probabilities corresponding to *ST*.
        """
        return self._interp1d[i](ST)

    def inv(self, P: np.ndarray | float, i: int) -> np.ndarray:
        """Evaluate the inverse CDF (quantile function) at a given timestep.

        Parameters
        ----------
        P : np.ndarray or float
            Cumulative probabilities in [0, 1].
        i : int
            Zero-based timestep index.

        Returns
        -------
        np.ndarray
            Cumulative storage (ST) values corresponding to *P*.
        """
        return self._interp1d_inv[i](P)

    def __repr__(self) -> str:
        """Return a string representation of all components."""
        result = ""
        for label, component in self.components.items():
            result += component.__repr__()
        return result

    def get_parameter_list(self) -> np.ndarray:
        """Return concatenated parameter arrays from all learnable components.

        Returns
        -------
        np.ndarray
            1-D array of all SAS function parameters, ordered by component.
        """
        return np.concatenate(
            [self.components[label].sas_fun[0].parameter_list for label in self._comp2learn_componentorder]
        )

    def update_from_parameter_list(self, parameter_list: np.ndarray) -> None:
        """Update component parameters and rebuild interpolators.

        Distributes entries of *parameter_list* across each learnable
        component's SAS function, then calls ``make_spec_ts`` to refresh
        the per-timestep CDF interpolators.

        Parameters
        ----------
        parameter_list : np.ndarray
            1-D array with the same length as ``get_parameter_list()``.
        """
        starti = 0
        for label in self._comp2learn_componentorder:
            component = self.components[label]
            nparams = len(component.sas_fun[0].parameter_list)
            new_params = parameter_list[starti : starti + nparams]
            # Propagate to every per-timestep SAS function: the solver reads
            # all timesteps via Component.argsS, so updating only element 0
            # would leave the calibrated parameters applied at t=0 alone.
            for fun in component._sas_funs:
                fun.parameter_list = np.array(new_params)
            starti += nparams
        self.make_spec_ts()

    def subdivided_copy(self, label: str, segment: int, **kwargs: Any) -> SAS_Spec:
        """Return a copy with one component's piecewise segment split in two.

        Parameters
        ----------
        label : str
            Which component's SAS function to subdivide.
        segment : int
            Which segment to subdivide (numbered from 0 for the youngest).
        **kwargs
            Forwarded to the SAS function's ``subdivided_copy`` method
            (e.g. ``split_frac``).

        Returns
        -------
        SAS_Spec
            A new spec with the subdivided component; interpolators rebuilt.
        """
        new_spec = deepcopy(self)
        component = new_spec.components[label]
        component.sas_fun = [fun.subdivided_copy(segment, **kwargs) for fun in component._sas_funs]
        new_spec.make_spec_ts()
        return new_spec

    def plot(self, ax: Axes | None = None, **kwargs: Any) -> dict[str, Any]:
        """Plot the SAS function for each component.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axes to draw on. A new figure is created if ``None``.
        **kwargs
            Additional keyword arguments forwarded to each component's
            ``plot`` method.

        Returns
        -------
        dict[str, Any]
            Mapping of component labels to the line objects returned by
            each component's ``plot`` call.
        """
        if ax is None:
            fig, ax = plt.subplots()
        componentlines = dict(
            [(label, component.plot(ax=ax, **kwargs)) for label, component in self.components.items()]
        )
        ax.set_xlabel("$S_T$")
        ax.set_ylabel(r"$\Omega(S_T)$")
        ax.legend(frameon=False)
        return componentlines

    def get_jacobian(self, *args: Any, index: np.ndarray | None = None, **kwargs: Any) -> np.ndarray:
        """Compute the Jacobian of the blended SAS function w.r.t. parameters.

        Calls ``get_jacobian`` on each learnable component's SAS function,
        weights the result, and concatenates column-wise.

        Parameters
        ----------
        *args
            Positional arguments forwarded to each component's
            ``get_jacobian``.
        index : np.ndarray or None, optional
            Timestep indices at which to evaluate. Defaults to all timesteps.
        **kwargs
            Additional keyword arguments forwarded to each component's
            ``get_jacobian``.

        Returns
        -------
        np.ndarray
            Jacobian matrix with shape ``(len(index), total_params)``.
        """
        if index is None:
            index = np.arange(self.N)
        cat_me = [
            self.components[label].sas_fun[0].get_jacobian(*args, index=index, **kwargs).T
            * self.components[label].weights.values[index]
            for label in self._comp2learn_componentorder
        ]
        return np.concatenate(cat_me, axis=0).T


class Component:
    """A single SAS function component with an associated weight time series.

    A ``Component`` pairs a per-timestep SAS function (either
    ``Piecewise`` or ``Continuous``) with a weight series so that multiple
    components can be blended by ``SAS_Spec``.

    Parameters
    ----------
    label : str
        Identifier for this component, used as a key in ``SAS_Spec``
        and as a column name in *data_df* for the weight series (when
        there are multiple components).
    spec : dict
        Specification dictionary. Must contain either ``"ST"`` (and
        optionally ``"P"``) keys for piecewise-linear SAS functions, or
        ``"args"`` with ``"func"``/``"scipy.stats"`` keys for continuous
        distributions. Values may be scalars, lists, or column-name
        strings referencing *data_df*.
    data_df : pd.DataFrame
        Input time-series data frame.
    is_only_component : bool
        If ``True``, weights default to 1.0 for every timestep instead
        of being read from *data_df*.

    Attributes
    ----------
    label : str
        Component identifier.
    weights : pd.Series
        Per-timestep blending weights.
    type : int
        Internal SAS function type code (``-1`` for scipy/piecewise).
    N : int
        Number of timesteps.
    """

    def __init__(
        self,
        label: str,
        spec: dict[str, Any],
        data_df: pd.DataFrame,
        is_only_component: bool,
    ) -> None:
        self.label = label
        self._spec = spec.copy()
        self._data_df = data_df
        self.N = len(data_df)
        self.weights = pd.Series(index=data_df.index, data=1.0) if is_only_component else data_df[label]
        ST = self.expand(spec.pop("ST")) if "ST" in spec else _NoneList()
        P = self.expand(spec.pop("P")) if "P" in spec else _NoneList()
        if "args" in spec:
            if not isinstance(spec["args"], dict):
                raise TypeError(f"Component '{label}': 'args' must be a dict, got {type(spec['args']).__name__}")
            argdict = OrderedDict()
            for arg, value in spec.pop("args").items():
                argdict[arg] = self.expand(value)
            self._args = [dict(zip(argdict, values)) for values in zip(*argdict.values())]
            if "scipy.stats" in spec:
                use = "scipy.stats"
                func = spec.pop("scipy.stats")
            else:
                use = spec.pop("use", "builtin")
                func = spec.pop("func")
            self._sas_funs = [
                Continuous(use, func, func_kwargs=self._args[i], P=P[i, :], **spec) for i in range(self.N)
            ]
            if use == "scipy.stats":
                self.type = -1
            elif use == "builtin":
                self.type = self._sas_funs[0]._builtinfunctype
            else:
                raise ValueError(f"Component '{label}': 'use' must be 'builtin' or 'scipy.stats', got '{use}'")
        elif ST is not None:
            self.type = -1
            self._sas_funs = [Piecewise(ST=ST[i, :], P=P[i, :], **spec) for i in range(self.N)]
        else:
            raise Exception(f"Insufficient information to specify SAS function for {label}")

    def expand(self, value: Any) -> np.ndarray:
        """Expand a specification value into a per-timestep array.

        Scalars are broadcast to all timesteps, strings are looked up as
        columns in the data frame, and iterables are recursively expanded
        and stacked.

        Parameters
        ----------
        value : float, str, or Iterable
            A scalar constant, a column name in the data frame, or a
            sequence of such values (one per breakpoint).

        Returns
        -------
        np.ndarray
            Array of shape ``(N,)`` for scalar/string inputs or
            ``(N, len(value))`` for iterable inputs, where *N* is the
            number of timesteps.
        """
        if isinstance(value, Iterable) and not isinstance(value, str):
            return np.concatenate([[self.expand(x)] for x in value], axis=0).T
        else:
            if isinstance(value, str):
                return self._data_df[value].values
            else:
                return float(value) * np.ones(self.N)

    @property
    def ST(self) -> np.ndarray:
        """Stacked ST breakpoints from all timesteps, shape ``(n_breakpoints, N)``."""
        return np.concatenate([[sas_fun.ST] for sas_fun in self._sas_funs]).T

    @property
    def P(self) -> np.ndarray:
        """Stacked P breakpoints from all timesteps, shape ``(n_breakpoints, N)``."""
        return np.concatenate([[sas_fun.P] for sas_fun in self._sas_funs]).T

    @property
    def argsS(self) -> np.ndarray:
        """Stacked argsS arrays from all timesteps."""
        return np.concatenate([[sas_fun.argsS] for sas_fun in self._sas_funs]).T

    @property
    def argsP(self) -> np.ndarray:
        """Stacked argsP arrays from all timesteps."""
        return np.concatenate([[sas_fun.argsP] for sas_fun in self._sas_funs]).T

    @property
    def sas_fun(self) -> list[Continuous | Piecewise]:
        """List of per-timestep SAS function objects."""
        return [sas_fun for sas_fun in self._sas_funs]

    @sas_fun.setter
    def sas_fun(self, new_sas_fun: Continuous | Piecewise | list) -> None:
        if isinstance(new_sas_fun, (list, tuple)):
            if len(new_sas_fun) != self.N:
                raise ValueError(
                    f"Component '{self.label}': expected {self.N} SAS functions (one per timestep), "
                    f"got {len(new_sas_fun)}"
                )
            self._sas_funs = list(new_sas_fun)
        else:
            # A single function applies at every timestep. Each timestep gets
            # its own copy because parameter updates mutate them individually.
            self._sas_funs = [deepcopy(new_sas_fun) for _ in range(self.N)]

    def __getitem__(self, i: int) -> Continuous | Piecewise:
        """Return the SAS function for timestep *i*."""
        return self._sas_funs[i]

    def __repr__(self) -> str:
        """Return a string representation of the component specification."""
        result = ""
        result += self._spec.__repr__()
        return result

    def plot(self, *args: Any, i: int = 0, **kwargs: Any) -> Any:
        """Plot the component's SAS function at one timestep.

        Parameters
        ----------
        *args
            Positional arguments forwarded to the SAS function's ``plot``.
        i : int, optional
            Timestep index of the SAS function to plot. Default 0.
        **kwargs
            Keyword arguments forwarded to the SAS function's ``plot``.
            The component ``label`` is automatically included.

        Returns
        -------
        Any
            Return value from the underlying SAS function's ``plot`` method.
        """
        return self._sas_funs[i].plot(*args, label=self.label, **kwargs)


class _NoneList:
    """Sentinel iterable that returns ``None`` for any index access.

    Used internally as a default when ST or P breakpoints are not
    provided in a component specification.
    """

    def __init__(self) -> None:
        pass

    def __getitem__(self, *args: Any, **kwargs: Any) -> None:
        return None
