"""StorAge Selection (SAS) transport model.

This module implements the :class:`Model` class, the main entry point for
building and running SAS models.  A Model wraps:

- a timeseries DataFrame of influx / outflux data,
- SAS function specifications (one per outflux), and
- optional solute transport parameters.

Call :meth:`Model.run` to invoke the solver; results are then
available through accessor methods (``get_sT``, ``get_pQ``, etc.) or
through the :class:`ModelResult` object returned by :attr:`Model.result`.
"""

from __future__ import annotations

import copy
import json
import os
import warnings
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import pandas as pd

from mesas.sas.specs import SAS_Spec

from ._solve_numba import RECORDABLE_ARRAYS, solve

dtype = np.float64


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelOptions:
    """Options controlling model execution.

    Parameters
    ----------
    dt : float
        Timestep size. Default 1.
    verbose : bool
        Print solver progress. Default False.
    num_scheme : int
        Runge-Kutta scheme: 1 (Euler), 2 (RK2), or 4 (RK4). Default 4.
    debug : bool
        Enable debug output. Default False.
    warning : bool
        Print solver warnings. Default True.
    jacobian : bool
        Compute Jacobian arrays. Default False.
    n_substeps : int
        Number of sub-timesteps per full step. Default 1.
    max_age : int or None
        Maximum water age (timesteps). Defaults to timeseries length.
    sT_init : np.ndarray or None
        Initial age-ranked storage. Defaults to zeros of length *max_age*.
    influx : str
        Name of the influx column in the data DataFrame. Default ``"J"``.
    ST_smallest_segment : float
        Minimum piecewise segment length. Default 0.01.
    ST_largest_segment : float
        Maximum piecewise segment length. Default inf.
    record_state : bool or str
        If ``True``, record every timestep; if ``False``, record only the
        last; if a string, treat it as a boolean column name in data_df.
        Default False.
    record_every : int or None
        Record the state at the end of every k-th timestep (a memory-saving
        alternative to ``record_state=True``; the recorded arrays shrink
        proportionally). Cannot be combined with a ``record_state`` column
        name. Default None.
    record_arrays : "all" or iterable of str
        Which state arrays to record: subset of {"sT", "pQ", "mT", "mQ",
        "mR", "water_balance", "solute_balance"}. Arrays not selected are
        never allocated (``mQ`` is typically the largest). Balance arrays
        require their ingredients: ``water_balance`` needs ``sT`` and
        ``pQ``; ``solute_balance`` needs ``mT``, ``mQ`` and ``mR``.
        Default "all".
    record_dtype : {"float64", "float32"}
        Storage dtype for the recorded arrays (halves memory when
        "float32"). Computation is always float64; balance diagnostics
        recorded as float32 are correspondingly less precise. Default
        "float64".
    record_to : str or None
        Directory in which to store the recorded arrays as disk-backed
        ``.npy`` memory-maps instead of RAM (created if needed). Removes
        RAM as the constraint for long runs; results in
        :attr:`Model.result` become read-only memmap arrays, and the
        directory is self-describing and reloadable via
        :meth:`Model.load_state`. Default None (RAM).
    validate_inputs : bool
        If ``True`` (default), :meth:`Model.run` first checks the input
        data for NaN values and negative fluxes and raises a descriptive
        error. Set to ``False`` to skip the check.
    """

    dt: float = 1.0
    verbose: bool = False
    num_scheme: int = 4
    debug: bool = False
    warning: bool = True
    jacobian: bool = False
    n_substeps: int = 1
    max_age: int | None = None
    sT_init: np.ndarray | None = field(default=None, repr=False)
    influx: str = "J"
    ST_smallest_segment: float = 1.0 / 100
    ST_largest_segment: float = np.inf
    record_state: bool | str = False
    record_every: int | None = None
    record_arrays: Any = "all"
    record_dtype: str = "float64"
    record_to: str | None = None
    validate_inputs: bool = True

    # Names of all valid option keys (for backward-compat dict validation)
    _VALID_KEYS: frozenset[str] = field(
        default=frozenset(
            {
                "dt",
                "verbose",
                "num_scheme",
                "debug",
                "warning",
                "jacobian",
                "n_substeps",
                "max_age",
                "sT_init",
                "influx",
                "ST_smallest_segment",
                "ST_largest_segment",
                "record_state",
                "record_every",
                "record_arrays",
                "record_dtype",
                "record_to",
                "validate_inputs",
            }
        ),
        init=False,
        repr=False,
    )

    @classmethod
    def from_dict(cls, d: dict) -> ModelOptions:
        """Create from a dict, raising ``KeyError`` on unknown keys."""
        valid = {f.name for f in fields(cls) if f.init}
        unknown = set(d) - valid
        if unknown:
            raise KeyError(f"Invalid options: {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in valid})

    def update(self, d: dict) -> None:
        """Update fields from a dict, raising on unknown keys."""
        valid = {f.name for f in fields(self.__class__) if f.init}
        unknown = set(d) - valid
        if unknown:
            raise KeyError(f"Invalid options: {sorted(unknown)}")
        for k, v in d.items():
            if k in valid:
                setattr(self, k, v)

    def to_dict(self) -> dict:
        """Return a plain dict of all option values."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}


@dataclass
class SoluteSpec:
    """Parameters for a single solute species.

    Parameters
    ----------
    C_old : float
        Concentration of water older than the initial condition. Default 0.
    mT_init : float or str
        Initial age-ranked solute mass. A float is broadcast; a string
        is looked up as a column in data_df. Default 0.
    k1 : float or str
        First-order reaction rate. Default 0.
    C_eq : float or str
        Equilibrium concentration. Default 0.
    alpha : dict[str, float or str]
        Per-flux evapoconcentration factors. Defaults to 1.0 for each flux.
    observations : dict or str
        Observation column mapping for calibration. Default empty dict.
    """

    C_old: float = 0.0
    mT_init: float | str = 0.0
    k1: float | str = 0.0
    C_eq: float | str = 0.0
    alpha: dict[str, float | str] = field(default_factory=dict)
    observations: dict | str = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict, flux_names: list[str]) -> SoluteSpec:
        """Create from a legacy dict, filling in default alpha per flux."""
        default_alpha = {flux: 1.0 for flux in flux_names}
        alpha = d.get("alpha", default_alpha)
        return cls(
            C_old=d.get("C_old", 0.0),
            mT_init=d.get("mT_init", 0.0),
            k1=d.get("k1", 0.0),
            C_eq=d.get("C_eq", 0.0),
            alpha=alpha,
            observations=d.get("observations", {}),
        )

    def to_dict(self) -> dict:
        """Return a plain dict."""
        return {
            "C_old": self.C_old,
            "mT_init": self.mT_init,
            "k1": self.k1,
            "C_eq": self.C_eq,
            "alpha": self.alpha,
            "observations": self.observations,
        }


class ModelResult:
    """Container for SAS model results with attribute-style access.

    Provides both dict-style (``result["sT"]``) and attribute-style
    (``result.sT``) access to result arrays. Also provides snake_case
    aliases for the camelCase result names.

    Parameters
    ----------
    data : dict
        Raw result dict from the solver.
    """

    # Map snake_case names -> original camelCase keys
    _ALIASES: dict[str, str] = {
        "water_balance": "WaterBalance",
        "solute_balance": "SoluteBalance",
    }
    # Reverse map for deprecation warnings
    _DEPRECATED: dict[str, str] = {
        "WaterBalance": "water_balance",
        "SoluteBalance": "solute_balance",
    }

    def __init__(self, data: dict) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        # Guard against infinite recursion during unpickling/copying:
        # protocol lookups (__setstate__, __deepcopy__, __reduce_ex__, ...)
        # arrive before _data exists in __dict__, and referencing self._data
        # here would re-enter __getattr__.
        if name.startswith("_") or "_data" not in self.__dict__:
            raise AttributeError(f"No result named '{name}'")
        # Check snake_case aliases first
        if name in self._ALIASES:
            return self._data[self._ALIASES[name]]
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"No result named '{name}'")

    def __getitem__(self, key: str) -> Any:
        # Support deprecated camelCase keys with warning
        if key in self._DEPRECATED:
            new_name = self._DEPRECATED[key]
            warnings.warn(
                f"Result key '{key}' is deprecated, use '{new_name}' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if key not in self._data and (key in RECORDABLE_ARRAYS or key in self._DEPRECATED):
            raise KeyError(
                f"Result '{key}' was not recorded in this run. Include it in the record_arrays option to record it."
            )
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def __repr__(self) -> str:
        shapes = {k: v.shape if hasattr(v, "shape") else type(v).__name__ for k, v in self._data.items()}
        return f"ModelResult({shapes})"


def _processinputs(input: dict | str) -> dict:
    """Load model input from a dict or a JSON file path.

    Parameters
    ----------
    input : dict or str
        Either a dict (returned as-is) or a path to a ``.json`` file.

    Returns
    -------
    dict
        The parsed input dictionary.
    """
    if isinstance(input, dict):
        return input
    elif isinstance(input, str):
        if os.path.isfile(input):
            with open(input) as json_file:
                data = json.load(json_file)
            return data
        else:
            raise FileNotFoundError(f"I could not find the input file: {input}")
    else:
        raise TypeError("input must be either a dict, or a path to a .json file")


class Model:
    """StorAge Selection (SAS) transport model.

    Construct with timeseries data, SAS function specifications, and
    (optionally) solute parameters, then call :meth:`run` to execute the
    Fortran solver.

    Parameters
    ----------
    data_df : pd.DataFrame or str
        Timeseries data, or a path to a CSV file.  Must contain at least
        the influx column (default ``"J"``) and one outflux column per
        entry in *sas_specs*.
    config : dict or str, optional
        Configuration dict or path to a JSON config file.  May contain
        ``"options"``, ``"sas_specs"``, and ``"solute_parameters"`` keys.
    sas_specs : dict or str, optional
        SAS specification dict (one key per outflux) or JSON path.
    solute_parameters : dict or str, optional
        Solute transport parameters or JSON path.
    **kwargs
        Any model option (``dt``, ``verbose``, ``num_scheme``, etc.)
        passed as keyword arguments overrides config-file values.
    """

    def __init__(
        self,
        data_df: pd.DataFrame | str,
        config: dict | str | None = None,
        sas_specs: dict | str | None = None,
        solute_parameters: dict | str | None = None,
        **kwargs: Any,
    ) -> None:
        # defaults
        self._result = None
        self.jacobian = {}
        # load the timeseries data
        self.data_df = data_df
        # check for a configuration file
        if config:
            config = _processinputs(config)
        # process options — build a ModelOptions dataclass
        self._options_obj = ModelOptions()
        components_to_learn = kwargs.pop("components_to_learn", None)
        if config and "options" in config:
            # ModelOptions.update raises KeyError on unknown option names
            self._options_obj.update(config["options"])
        # Any remaining kwargs must be valid options — reject typos loudly
        # rather than silently discarding them (e.g. Model(df, veborse=True))
        unknown_kwargs = set(kwargs) - ModelOptions._VALID_KEYS
        if unknown_kwargs:
            raise TypeError(
                f"Unknown keyword argument(s) for Model: {sorted(unknown_kwargs)}. "
                f"Valid options are: {sorted(ModelOptions._VALID_KEYS)}"
            )
        if kwargs:
            self._options_obj.update(kwargs)
        self._apply_options()
        # get the SAS specification
        if config and "sas_specs" in config:
            sas_specs = config["sas_specs"]
        elif sas_specs:
            sas_specs = _processinputs(sas_specs)
        else:
            raise ValueError("No SAS specification found!")
        self.sas_specs = sas_specs  # setter parses raw dicts into SAS_Spec objects
        self._numflux = len(self.sas_specs)
        self._fluxorder = list(self.sas_specs.keys())
        # get solute transport parameters
        if config and "solute_parameters" in config:
            solute_parameters = config["solute_parameters"]
        elif solute_parameters:
            solute_parameters = _processinputs(solute_parameters)
        # defaults for solute transport
        self._default_parameters = {
            "mT_init": 0.0,
            "C_old": 0.0,
            "k1": 0.0,
            "C_eq": 0.0,
            "alpha": {flux: 1.0 for flux in self._fluxorder},
            "observations": {},
        }
        self.solute_parameters = solute_parameters
        self.components_to_learn = components_to_learn

    def _apply_options(self) -> None:
        """Resolve max_age, sT_init, and index_ts from current options."""
        opts = self._options_obj
        if opts.sT_init is not None:
            opts.sT_init = np.asarray(opts.sT_init, dtype=float)
            if opts.max_age is not None and opts.max_age != len(opts.sT_init):
                raise ValueError(
                    f"max_age ({opts.max_age}) conflicts with len(sT_init) ({len(opts.sT_init)}). "
                    "When sT_init is given, max_age is taken from its length; "
                    "either omit max_age or make them consistent."
                )
            opts.max_age = len(opts.sT_init)
        if opts.max_age is None:
            opts.max_age = self._timeseries_length
        if opts.max_age > self._timeseries_length:
            raise ValueError(f"max_age ({opts.max_age}) cannot exceed timeseries_length ({self._timeseries_length})")
        if opts.sT_init is None:
            opts.sT_init = np.zeros(opts.max_age)
        self._max_age = opts.max_age
        if opts.record_state is False:
            self._index_ts = np.array([self._timeseries_length - 1])
        elif opts.record_state is True:
            self._index_ts = np.arange(self._timeseries_length)
        elif isinstance(opts.record_state, str):
            if opts.record_state not in self.data_df.columns:
                raise ValueError(
                    f"record_state column '{opts.record_state}' not found in data_df. "
                    f"Available columns: {list(self.data_df.columns)}"
                )
            self._index_ts = np.where(self.data_df[opts.record_state])[0]
        else:
            raise TypeError(
                f"record_state must be True, False, or the name of a boolean column in data_df, "
                f"got {opts.record_state!r}"
            )
        if opts.record_every is not None:
            if isinstance(opts.record_state, str):
                raise ValueError(
                    "record_every cannot be combined with a record_state column name; "
                    "use one or the other to choose which timesteps to record"
                )
            if not isinstance(opts.record_every, (int, np.integer)) or opts.record_every < 1:
                raise ValueError(f"record_every must be a positive integer, got {opts.record_every!r}")
            # Record the state at the end of every k-th timestep
            self._index_ts = np.arange(opts.record_every - 1, self._timeseries_length, opts.record_every)
        if opts.record_dtype not in ("float64", "float32"):
            raise ValueError(f"record_dtype must be 'float64' or 'float32', got {opts.record_dtype!r}")
        if opts.record_arrays == "all":
            record_set = set(RECORDABLE_ARRAYS)
        else:
            if isinstance(opts.record_arrays, str):
                raise TypeError(
                    f"record_arrays must be 'all' or an iterable of array names, got {opts.record_arrays!r}"
                )
            record_set = set(opts.record_arrays)
            unknown = record_set - RECORDABLE_ARRAYS
            if unknown:
                raise ValueError(
                    f"Unknown record_arrays entries: {sorted(unknown)}. Valid names: {sorted(RECORDABLE_ARRAYS)}"
                )
            if "water_balance" in record_set and not {"sT", "pQ"} <= record_set:
                raise ValueError(
                    "record_arrays: 'water_balance' requires 'sT' and 'pQ' to also be recorded "
                    "(it is computed from them)"
                )
            if "solute_balance" in record_set and not {"mT", "mQ", "mR"} <= record_set:
                raise ValueError(
                    "record_arrays: 'solute_balance' requires 'mT', 'mQ' and 'mR' to also be recorded "
                    "(it is computed from them)"
                )
        self._record_set = frozenset(record_set)

    def __repr__(self):
        """Creates a repr for the model"""
        result = ""
        for flux, sas_spec in self.sas_specs.items():
            result += f"flux = {flux}\n"
            result += sas_spec.__repr__()
        return result

    def parse_sas_specs(self, sas_specs: dict) -> dict[str, SAS_Spec]:
        """Validate and convert raw SAS spec dicts into :class:`SAS_Spec` objects.

        Returns a new dict; the caller's dict is not modified.
        """
        parsed: dict[str, SAS_Spec] = {}
        for flux, spec_in in sas_specs.items():
            if flux not in self.data_df.columns:
                raise ValueError(f"Flux '{flux}' not found in data_df columns. Available: {list(self.data_df.columns)}")
            if isinstance(spec_in, SAS_Spec):
                parsed[flux] = spec_in
            else:
                if not isinstance(spec_in, dict):
                    raise TypeError(
                        f"SAS spec for flux '{flux}' must be a dict or SAS_Spec, got {type(spec_in).__name__}"
                    )
                for label, component_spec in spec_in.items():
                    if not isinstance(component_spec, dict):
                        raise TypeError(
                            f"Component '{label}' of flux '{flux}' must be a dict, got {type(component_spec).__name__}"
                        )
                parsed[flux] = SAS_Spec(deepcopy(spec_in), self.data_df)
        return parsed

    def copy_without_results(self) -> Model:
        """Return a deep copy of this model with results cleared."""
        return Model(
            copy.deepcopy(self._data_df),
            sas_specs=copy.deepcopy(self._sas_specs),
            solute_parameters=copy.deepcopy(self._solute_parameters),
            components_to_learn=copy.deepcopy(
                self._components_to_learn if hasattr(self, "_components_to_learn") else None
            ),
            **copy.deepcopy(self._options_obj.to_dict()),
        )

    def subdivided_copy(self, flux, label, segment):
        """
        Creates a copy of the model with one segment of a sas function subdivided in two

        :param flux: name of the flux
        :param label: name of the component
        :param segment: segment (numbered from 0 for the youngest)
        :return: a new Model instance with one piecewise segment subdivided
        """

        # make a copy
        new_model = self.copy_without_results()

        # subdivide the component
        new_model.sas_specs[flux] = new_model.sas_specs[flux].subdivided_copy(label, segment)

        return new_model

    @property
    def fluxorder(self):
        return self._fluxorder

    @fluxorder.setter
    def fluxorder(self, new_fluxorder):
        raise AttributeError("fluxorder property is read-only")

    @property
    def solorder(self):
        return self._solorder

    @solorder.setter
    def solorder(self, new_solorder):
        raise AttributeError("solorder property is read-only")

    @property
    def result(self) -> ModelResult:
        """Results of running the SAS model. See :ref:`results`.

        Returns a :class:`ModelResult` supporting both dict-style
        (``result["sT"]``) and attribute-style (``result.sT``) access.
        """
        if self._result is None:
            raise AttributeError("Results are only available after calling .run()")
        return self._result

    @result.setter
    def result(self, result: Any) -> None:
        raise AttributeError("Model results are read-only. Use .run() method to generate results.")

    @property
    def data_df(self):
        """Pandas dataframe holding the model inputs. See :ref:`inputs`"""
        return self._data_df

    @data_df.setter
    def data_df(self, new_data_df):
        if isinstance(new_data_df, pd.core.frame.DataFrame):
            self._data_df = new_data_df.copy()
        elif isinstance(new_data_df, str):
            self._data_df = pd.read_csv(new_data_df)
        else:
            raise TypeError("data_df must be either a pandas dataframe, or a path to a .csv file")
        if len(self._data_df) == 0:
            raise ValueError("data_df is empty — the model needs at least one timestep of input data")
        self._timeseries_length = len(self._data_df)

    @property
    def options(self) -> dict:
        """Options for running the model (returns a dict for backward compat).

        Assign a dict to update options. Use ``model.options_obj`` to access
        the :class:`ModelOptions` dataclass directly.
        """
        return self._options_obj.to_dict()

    @options.setter
    def options(self, new_options: dict | ModelOptions) -> None:
        if isinstance(new_options, ModelOptions):
            self._options_obj = new_options
        else:
            new_options = _processinputs(new_options)
            self._options_obj.update(new_options)
        self._apply_options()

    @property
    def sas_specs(self):
        """Specification of the SAS functions. See :ref:`sasspec`"""
        return self._sas_specs

    @sas_specs.setter
    def sas_specs(self, new_sas_specs):
        # Parse raw spec dicts into SAS_Spec objects (already-parsed specs
        # pass through unchanged), so that assigning a raw dict after
        # construction behaves the same as passing one to the constructor.
        self._sas_specs = self.parse_sas_specs(_processinputs(new_sas_specs))
        self._numflux = len(self._sas_specs)
        self._fluxorder = list(self._sas_specs.keys())

    def set_sas_spec(self, flux: str, sas_spec: SAS_Spec) -> None:
        """Replace the SAS specification for a given flux."""
        self._sas_specs[flux] = sas_spec
        self._sas_specs[flux].make_spec_ts(self.data_df)

    def set_component(self, flux: str, component: Any) -> None:
        """Replace a single component within a flux's SAS specification."""
        label = component.label
        self._sas_specs[flux].components[label] = component
        self._sas_specs[flux].make_spec_ts()

    def set_sas_fun(self, flux: str, label: str, sas_fun: Any) -> None:
        """Replace the SAS function of a named component.

        Parameters
        ----------
        flux : str
            Name of the flux whose SAS spec contains the component.
        label : str
            Label of the component to modify.
        sas_fun : Piecewise, Continuous, or list thereof
            A single SAS function (applied at every timestep) or a list
            with one function per timestep.
        """
        self._sas_specs[flux].components[label].sas_fun = sas_fun
        self._sas_specs[flux].make_spec_ts()

    def get_component_labels(self) -> dict[str, list[str]]:
        """Return ``{flux: [label, ...]}`` for all SAS components."""
        component_labels = {}
        for flux in self._fluxorder:
            component_labels[flux] = list(self._sas_specs[flux].components.keys())
        return component_labels

    @property
    def components_to_learn(self):
        """
        A dictionary of lists giving the component labels that you want to train with MESAS

        The components in this dictionary can be set by assigning a dictionary. For example, this would
        include only the 'min' and 'max' components of the sas spec associated with the 'Q' flux.

            >>> model.components_to_learn = {'Q':['min', 'max']}

        The parameters associated with these components can be obtained as a 1-D array with:

            >>> seglist = model.get_parameter_list()

        and modified using:

            >>> model.update_from_parameter_list(seglst)

        :return:
        """
        return self._components_to_learn

    @components_to_learn.setter
    def components_to_learn(self, label_dict):
        if label_dict is not None:
            self._components_to_learn = OrderedDict(
                (flux, [label for label in self._sas_specs[flux]._componentorder if label in label_dict[flux]])
                for flux in self._fluxorder
                if flux in label_dict.keys()
            )
            self._comp2learn_fluxorder = list(self._components_to_learn.keys())
            for flux in self._comp2learn_fluxorder:
                self._sas_specs[flux]._comp2learn_componentorder = self._components_to_learn[flux]

    def get_parameter_list(self):
        """
        Returns a concatenated list of segments from self.components_to_learn
        """
        return np.concatenate([self.sas_specs[flux].get_parameter_list() for flux in self._comp2learn_fluxorder])

    def update_from_parameter_list(self, parameter_list):
        """
        Modifies the components in self.components_to_learn from a concatenated list of segments
        """
        starti = 0
        for flux in self._comp2learn_fluxorder:
            nparams = len(self.sas_specs[flux].get_parameter_list())
            self.sas_specs[flux].update_from_parameter_list(parameter_list[starti : starti + nparams])
            starti += nparams

    @property
    def solute_parameters(self):
        """Parameters describing solute behavior. See :ref:`solspec`"""
        return self._solute_parameters

    @solute_parameters.setter
    def solute_parameters(self, new_solute_parameters):
        # set parameters for solute transport
        # provide defaults if absent
        if new_solute_parameters is not None:
            new_solute_parameters = _processinputs(new_solute_parameters)
            self._solute_parameters = {}
            for sol, params in new_solute_parameters.items():
                if sol not in self.data_df.columns:
                    raise ValueError(
                        f"Solute '{sol}' has no matching input concentration column in data_df. "
                        f"Available columns: {list(self.data_df.columns)}"
                    )
                self._solute_parameters[sol] = deepcopy(self._default_parameters)
                self.set_solute_parameters(sol, params)
            self._numsol = len(self._solute_parameters)
            self._solorder = list(self._solute_parameters.keys())
        else:
            self._numsol = 0
            self._solute_parameters = {}
            self._solorder = []

    def set_solute_parameters(self, sol: str, params: dict) -> None:
        invalid_parameters = [paramkey for paramkey in params.keys() if paramkey not in self._default_parameters.keys()]
        if any(invalid_parameters):
            raise KeyError("invalid parameters for {}: {}".format(sol, invalid_parameters))
        self._solute_parameters[sol].update(params)

    def _create_solute_inputs(self):
        numsol = max(self._numsol, 1)
        C_J = np.zeros((self._timeseries_length, numsol), dtype=dtype)
        mT_init = np.zeros((self._max_age, numsol), dtype=dtype)
        C_old = np.zeros(numsol, dtype=dtype)
        k1 = np.zeros((self._timeseries_length, numsol), dtype=dtype)
        C_eq = np.zeros((self._timeseries_length, numsol), dtype=dtype)
        alpha = np.ones((self._timeseries_length, self._numflux, numsol), dtype=dtype)
        if self.solute_parameters is not None:

            def _get_array(param, N, name, sol):
                if isinstance(param, str):
                    if param in self.data_df:
                        return self.data_df[param].values
                    raise ValueError(
                        f"Column '{param}' given for parameter '{name}' of solute '{sol}' "
                        f"was not found in data_df. Available columns: {list(self.data_df.columns)}"
                    )
                return param * np.ones(N)

            for isol, sol in enumerate(self._solorder):
                C_J[:, isol] = self.data_df[sol].values
                C_old[isol] = self.solute_parameters[sol]["C_old"]
                mT_init[:, isol] = _get_array(self.solute_parameters[sol]["mT_init"], self._max_age, "mT_init", sol)
                k1[:, isol] = _get_array(self.solute_parameters[sol]["k1"], self._timeseries_length, "k1", sol)
                C_eq[:, isol] = _get_array(self.solute_parameters[sol]["C_eq"], self._timeseries_length, "C_eq", sol)
                for iflux, flux in enumerate(self._fluxorder):
                    alpha[:, iflux, isol] = _get_array(
                        self.solute_parameters[sol]["alpha"][flux], self._timeseries_length, f"alpha[{flux}]", sol
                    )
        return C_J, mT_init, C_old, alpha, k1, C_eq

    def _create_sas_lookup(self):
        """Build flattened SAS lookup tables for the Fortran solver.

        The Fortran solver expects all SAS function parameters packed into
        flat arrays.  This method iterates over fluxes and their components
        to produce:

        - ``SAS_args``: breakpoint ST values, shape ``(nC_total, max_nargs)``
        - ``P_list``: breakpoint P values, same shape
        - ``weights``: component blending weights, shape ``(timeseries_length, nC_total)``
        - ``component_type``: int type code per component (0=piecewise, 1+=continuous)
        - ``nC_list``: number of components per flux
        - ``nargs_list``: number of breakpoints per component
        """
        nC_list = []  # number of components for each flux
        component_list = []  # flat list of all Component objects across all fluxes
        for flux in self._fluxorder:
            nC_list.append(len(self._sas_specs[flux].components))
            for component in self._sas_specs[flux]._componentorder:
                component_list.append(self.sas_specs[flux].components[component])
        nargs_list = [len(component.argsS) for component in component_list]
        component_type = [component.type for component in component_list]
        nC_total = np.sum(nC_list)
        nargs_total = np.sum(nargs_list)
        # Pack breakpoint arrays: each row is one component's ST or P values
        SAS_args = np.column_stack([[component.argsS] for component in component_list]).T
        P_list = np.column_stack([[component.argsP] for component in component_list]).T
        # Blending weights: shape (timeseries_length, nC_total)
        weights = np.column_stack([component.weights for component in component_list])
        return SAS_args, P_list, weights, component_type, nC_list, nC_total, nargs_list, nargs_total

    def validate(self) -> None:
        """Check the input data for common problems and raise if found.

        Checks performed:

        - the influx column and every outflux column exist and contain no
          NaN values and no negative values
        - every solute input concentration column contains no NaN values

        NaN in any of these columns propagates silently through the solver
        and produces NaN (or partially NaN) outputs; negative fluxes are
        physically meaningless but run without complaint.

        Raises
        ------
        ValueError
            With the offending column name, the number of bad values, and
            the first offending index. Set the model option
            ``validate_inputs=False`` to skip this check during ``run()``.
        """
        opts = self._options_obj

        def _check(col, allow_negative):
            if col not in self.data_df.columns:
                raise ValueError(f"Column '{col}' not found in data_df. Available: {list(self.data_df.columns)}")
            values = self.data_df[col].values
            if not np.issubdtype(np.asarray(values).dtype, np.number):
                raise ValueError(f"Column '{col}' is not numeric (dtype {np.asarray(values).dtype})")
            isnan = np.isnan(values)
            if isnan.any():
                idx = np.where(isnan)[0]
                raise ValueError(
                    f"Column '{col}' contains {len(idx)} NaN value(s) (first at index {idx[0]}). "
                    "Fill or remove them before running, or set validate_inputs=False to skip this check."
                )
            if not allow_negative:
                isneg = values < 0
                if isneg.any():
                    idx = np.where(isneg)[0]
                    raise ValueError(
                        f"Column '{col}' contains {int(isneg.sum())} negative value(s) "
                        f"(first at index {idx[0]}: {values[idx[0]]}). Fluxes must be non-negative. "
                        "Set validate_inputs=False to skip this check."
                    )

        _check(opts.influx, allow_negative=False)
        for flux in self._fluxorder:
            _check(flux, allow_negative=False)
        for sol in self._solorder:
            # concentrations may legitimately be negative (e.g. isotope delta values)
            _check(sol, allow_negative=True)

    def run(self) -> None:
        """Execute the SAS model with current parameters.

        Calls the solver (``solve``) and stores results in
        ``self._result``.  Predicted outflux concentrations are also
        written back into ``self.data_df`` as columns named
        ``"<solute> --> <flux>"``.

        Results are accessible via :attr:`result` and the ``get_*`` methods.
        """
        # --- 0. Validate inputs (opt out with validate_inputs=False) ---
        if self._options_obj.validate_inputs:
            self.validate()
        # --- 1. Extract water flux timeseries ---
        opts = self._options_obj
        J = self.data_df[opts.influx].values  # influx [T]
        Q = self.data_df[self._fluxorder].values  # outfluxes [T, numflux]
        sT_init = opts.sT_init
        timeseries_length = self._timeseries_length
        numflux = self._numflux

        # --- 2. Build SAS lookup tables (flattened for Fortran) ---
        SAS_args, P_list, weights, component_type, nC_list, nC_total, nargs_list, nargs_total = (
            self._create_sas_lookup()
        )
        SAS_args = np.ascontiguousarray(SAS_args)
        P_list = np.ascontiguousarray(P_list)
        weights = np.ascontiguousarray(weights)

        # --- 3. Build solute input arrays ---
        C_J, mT_init, C_old, alpha, k1, C_eq = self._create_solute_inputs()
        numsol = max(self._numsol, 1)

        # --- 4. Unpack scalar options ---
        dt = opts.dt
        verbose = opts.verbose
        debug = opts.debug
        warning = opts.warning
        jacobian = opts.jacobian
        if jacobian:
            raise NotImplementedError(
                "options['jacobian']=True is not implemented in the Numba solver: "
                "the returned sensitivities would be all zeros. Use numerical "
                "jacobians instead (e.g. mesas.me.fit_model with jacobian_mode='numerical')."
            )
        n_substeps = opts.n_substeps
        max_age = opts.max_age
        num_scheme = opts.num_scheme

        # Which timesteps to record (depends on record_state option)
        index_ts = self._index_ts

        # Which arrays to record (solute arrays are dropped when no solutes)
        record_set = set(self._record_set)
        if self._numsol == 0:
            record_set -= {"mT", "mQ", "mR", "solute_balance"}
        record_dir = os.fspath(opts.record_to) if opts.record_to is not None else None

        # Truncate initial conditions to max_age
        sT_init = sT_init[:max_age]
        mT_init = mT_init[:max_age, :]

        # --- 5. Call the solver ---
        # The solver expects ~30 positional arguments in a specific order.
        # SAS_args/P_list are transposed (numargs_total, timeseries_length).
        fresult = solve(
            J,  # influx timeseries
            Q,  # outflux timeseries [T, numflux]
            np.ascontiguousarray(SAS_args.T),  # SAS breakpoint ST values
            np.ascontiguousarray(P_list.T),  # SAS breakpoint P values
            weights,  # component blending weights
            sT_init,  # initial age-ranked storage
            dt,  # timestep size
            verbose,
            debug,
            warning,
            jacobian,  # flags
            mT_init,  # initial age-ranked solute mass
            C_J,  # influx concentrations
            alpha,  # evapoconcentration factors
            k1,  # first-order reaction rates
            C_eq,  # equilibrium concentrations
            C_old,  # concentration of pre-initial water
            n_substeps,  # sub-timesteps per full step
            component_type,  # int type code per component
            nC_list,  # num components per flux
            nargs_list,  # num breakpoints per component
            index_ts,  # which timesteps to record
            num_scheme,  # RK scheme (1=Euler, 2=RK2, 4=RK4)
            numflux,
            numsol,
            max_age,
            timeseries_length,
            len(index_ts),
            nC_total,
            nargs_total,
            record_to=record_dir,
            record_arrays=frozenset(record_set),
            record_dtype=opts.record_dtype,
        )
        sT, pQ, WaterBalance, mT, mQ, mR, C_Q, dsTdSj, dmTdSj, dCdSj, SoluteBalance = fresult

        # --- 6. Store results ---
        # Solver arrays are already age-first: (max_age, n_recorded_steps, ...).
        # Only arrays selected via record_arrays are present; the Jacobian
        # entries are singleton placeholders (not computed by the Numba solver).
        result_data = {}
        if self._numsol > 0:
            result_data["C_Q"] = C_Q
            # Write predicted concentrations back into the DataFrame
            for isol, sol in enumerate(self._solorder):
                for iflux, flux in enumerate(self._fluxorder):
                    colname = sol + " --> " + flux
                    self._data_df[colname] = C_Q[:, iflux, isol]
        if "sT" in record_set:
            result_data["sT"] = sT
        if "pQ" in record_set:
            result_data["pQ"] = pQ
        if "water_balance" in record_set:
            # snake_case canonical name + deprecated camelCase alias
            result_data["water_balance"] = WaterBalance
            result_data["WaterBalance"] = WaterBalance
        result_data["dsTdSj"] = dsTdSj
        if self._numsol > 0:
            if "mT" in record_set:
                result_data["mT"] = mT
            if "mQ" in record_set:
                result_data["mQ"] = mQ
            if "mR" in record_set:
                result_data["mR"] = mR
            if "solute_balance" in record_set:
                result_data["solute_balance"] = SoluteBalance
                result_data["SoluteBalance"] = SoluteBalance
            result_data["dmTdSj"] = dmTdSj
            result_data["dCdSj"] = dCdSj
        self._result = ModelResult(result_data)

        # --- 7. Persist run metadata alongside disk-backed state ---
        if record_dir is not None:
            self._write_run_metadata(record_dir, record_set)

    def _write_run_metadata(self, record_dir: str, record_set: set) -> None:
        """Make a ``record_to`` directory self-describing and reloadable."""
        from datetime import datetime, timezone

        opts = self._options_obj
        np.save(os.path.join(record_dir, "index_ts.npy"), self._index_ts)
        if self._numsol > 0:
            np.save(os.path.join(record_dir, "C_Q.npy"), np.asarray(self._result["C_Q"]))
        try:
            from importlib.metadata import version as _pkg_version

            mesas_version = _pkg_version("mesas")
        except Exception:
            mesas_version = "unknown"
        meta = {
            "format": "mesas-run-v1",
            "mesas_version": mesas_version,
            "created": datetime.now(timezone.utc).isoformat(),
            "dt": opts.dt,
            "n_substeps": opts.n_substeps,
            "num_scheme": opts.num_scheme,
            "max_age": int(self._max_age),
            "timeseries_length": int(self._timeseries_length),
            "num_recorded_steps": int(len(self._index_ts)),
            "fluxorder": list(self._fluxorder),
            "solorder": list(self._solorder),
            "record_dtype": opts.record_dtype,
            "record_arrays": sorted(record_set),
            "arrays": {name: list(self._result[name].shape) for name in sorted(record_set)},
        }
        with open(os.path.join(record_dir, "mesas_run.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load_state(record_dir: str) -> ModelResult:
        """Reload recorded state from a ``record_to`` directory.

        The arrays are opened as read-only memory-maps, so this is cheap
        even for very large runs. Returns a :class:`ModelResult` with the
        recorded arrays plus ``C_Q``, ``index_ts``, and ``run_metadata``
        (the contents of ``mesas_run.json``).
        """
        record_dir = os.fspath(record_dir)
        meta_path = os.path.join(record_dir, "mesas_run.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No mesas_run.json found in {record_dir!r} — was this directory "
                "created by running a Model with record_to set?"
            )
        with open(meta_path) as f:
            meta = json.load(f)
        data: dict[str, Any] = {"run_metadata": meta}
        for name in meta.get("record_arrays", []):
            path = os.path.join(record_dir, name + ".npy")
            if os.path.exists(path):
                data[name] = np.load(path, mmap_mode="r")
        # deprecated camelCase aliases
        if "water_balance" in data:
            data["WaterBalance"] = data["water_balance"]
        if "solute_balance" in data:
            data["SoluteBalance"] = data["solute_balance"]
        for extra in ("C_Q", "index_ts"):
            path = os.path.join(record_dir, extra + ".npy")
            if os.path.exists(path):
                data[extra] = np.load(path, mmap_mode="r")
        return ModelResult(data)

    def get_jacobian(self, mode="segment", logtransform=True):
        """Not available: the Numba solver does not compute Jacobian arrays."""
        raise NotImplementedError(
            "Model.get_jacobian requires the solver's Jacobian arrays, which the "
            "Numba solver does not compute. Use numerical jacobians instead "
            "(e.g. mesas.me.fit_model with jacobian_mode='numerical')."
        )

    def get_residuals(self):
        residuals = None
        for isol, sol in enumerate(self._solorder):
            if "observations" in self.solute_parameters[sol]:
                for iflux, flux in enumerate(self._comp2learn_fluxorder):
                    if flux in self.solute_parameters[sol]["observations"]:
                        obs = self.solute_parameters[sol]["observations"][flux]
                        C_obs = self.data_df[obs]
                        iflux = list(self._fluxorder).index(flux)
                        isol = list(self._solorder).index(sol)
                        this_residual = self.result["C_Q"][:, iflux, isol] - C_obs.values
                        if residuals is None:
                            residuals = this_residual
                        else:
                            residuals = np.concatenate((residuals, this_residual), axis=0)
                        self.data_df[f"residual {flux}, {sol}, {obs}"] = this_residual
        return residuals

    def get_obs_index(self):
        index = None
        for isol, sol in enumerate(self._solorder):
            if "observations" in self.solute_parameters[sol]:
                for iflux, flux in enumerate(self._comp2learn_fluxorder):
                    if flux in self.solute_parameters[sol]["observations"]:
                        obs = self.solute_parameters[sol]["observations"][flux]
                        C_obs = self.data_df[obs]
                        this_index = ~np.isnan(C_obs.values)
                        if index is None:
                            index = this_index
                        else:
                            index = np.concatenate((index, this_index), axis=0)
        return np.nonzero(index)[0]

    def _get_result(
        self, X: np.ndarray, timestep: int | None = None, agestep: int | None = None, inputtime: int | None = None
    ) -> np.ndarray:
        n_given = sum(x is not None for x in (timestep, agestep, inputtime))
        if n_given > 1:
            raise ValueError("Only one of timestep, agestep, or inputtime may be given")
        if timestep is not None:
            return X[:, timestep]
        if agestep is not None:
            return X[agestep, :]
        if inputtime is not None:
            return np.diagonal(X, offset=inputtime)
        return X

    def get_water_balance(self, **kwargs) -> np.ndarray:
        """Water conservation residual array, shape ``(max_age, n_output_steps)``.

        Should be near machine precision when ``record_state=True``.
        """
        X = self.result["water_balance"]
        return self._get_result(X, **kwargs)

    def get_WaterBalance(self, **kwargs) -> np.ndarray:
        """Deprecated: use :meth:`get_water_balance` instead."""
        warnings.warn(
            "get_WaterBalance() is deprecated, use get_water_balance()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_water_balance(**kwargs)

    def get_sT(self, **kwargs) -> np.ndarray:
        """Age-ranked storage density ``sT``, shape ``(max_age, n_output_steps)``."""
        X = self.result["sT"]
        return self._get_result(X, **kwargs)

    def get_pQ(self, flux: str, **kwargs) -> np.ndarray:
        """Age-ranked outflux probability ``pQ`` for a given flux."""
        iflux = list(self._fluxorder).index(flux)
        X = self.result["pQ"][:, :, iflux]
        return self._get_result(X, **kwargs)

    def get_mT(self, sol: str, **kwargs) -> np.ndarray:
        """Age-ranked solute mass density ``mT`` for a given solute."""
        isol = list(self._solorder).index(sol)
        X = self.result["mT"][:, :, isol]
        return self._get_result(X, **kwargs)

    def get_CT(self, sol: str, **kwargs) -> np.ndarray:
        """Age-ranked solute concentration ``CT = mT / sT``."""
        isol = list(self._solorder).index(sol)
        sT = self._get_result(self.result["sT"], **kwargs)
        mT = self._get_result(self.result["mT"][:, :, isol], **kwargs)
        return np.where(sT > 0, mT / sT, np.nan)

    def get_mR(self, sol: str, **kwargs) -> np.ndarray:
        """Age-ranked reaction mass ``mR`` for a given solute."""
        isol = list(self._solorder).index(sol)
        X = self.result["mR"][:, :, isol]
        return self._get_result(X, **kwargs)

    def get_solute_balance(self, sol: str, **kwargs) -> np.ndarray:
        """Solute conservation residual for a given solute.

        Should be near machine precision when ``record_state=True``.
        """
        isol = list(self._solorder).index(sol)
        X = self.result["solute_balance"][:, :, isol]
        return self._get_result(X, **kwargs)

    def get_SoluteBalance(self, sol: str, **kwargs) -> np.ndarray:
        """Deprecated: use :meth:`get_solute_balance` instead."""
        warnings.warn(
            "get_SoluteBalance() is deprecated, use get_solute_balance()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_solute_balance(sol, **kwargs)

    def get_mQ(self, flux: str, sol: str, **kwargs) -> np.ndarray:
        """Age-ranked solute outflux ``mQ`` for a given flux and solute."""
        iflux = list(self._fluxorder).index(flux)
        isol = list(self._solorder).index(sol)
        X = self.result["mQ"][:, :, iflux, isol]
        return self._get_result(X, **kwargs)

    def get_ST(self, **kwargs) -> np.ndarray:
        """Cumulative storage ``ST = cumsum(sT) * dt``."""
        sT = self.get_sT()
        ST = np.cumsum(sT, axis=0) * self._options_obj.dt
        return self._get_result(ST, **kwargs)
