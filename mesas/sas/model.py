"""

    Module Models
    =============

    Text here
"""
import copy
from collections import OrderedDict
from copy import deepcopy
import pandas as pd
import json

import numpy as np
from .solve import solvesas as solve

dtype = np.float64

from mesas.sas.specs import SAS_Spec
from collections import OrderedDict

import os

def _processinputs(input):
    if isinstance(input, dict):
        return input
    elif isinstance(input, str):
        if os.path.isfile(input):
            with open(input) as json_file:
                data = json.load(json_file) 
            return data
        else:
            raise FileNotFoundError(f'I could not find the input file: {input}')
    else:
        raise TypeError('input must be either a dict, or a path to a .json file')



class Model:
    '''
    Class for building and running StorAge Selection models

    An instance of this class must be constructed, populated with
    parameters (held in the dicts :ref:`sas_specs <sasspec>` and optionally :ref:`solute_parameters <solspec>`),
    associated with a dataset (`data_df`) and run using the `.run()` method.
    '''

    def __init__(self, data_df, config=None, sas_specs=None, solute_parameters=None, **kwargs):
        # defaults
        self._result = None
        self.jacobian = {}
        # load the timeseries data
        self.data_df = data_df
        # check for a configuration file
        if config:
            config = _processinputs(config)
        # process any options
        self._default_options = {
            'dt': 1,
            'verbose': False,
            'num_scheme': 4,
            'debug': False,
            'warning': True,
            'jacobian': False,
            'n_substeps': 1,
            'max_age': None,
            'sT_init': None,
            'influx': 'J',
            'ST_smallest_segment': 1./100,
            'ST_largest_segment': np.inf,
            'record_state': False
        }
        self._options = self._default_options
        components_to_learn = kwargs.get('components_to_learn')
        if config and 'options' in config.keys():
            self.options = config['options']
        self.options = kwargs
        # get the SAS specification
        if config and 'sas_specs' in config.keys():
            sas_specs = config['sas_specs']
        elif sas_specs:
            sas_specs = _processinputs(sas_specs)
        else:
            raise ValueError("No SAS specification found!")
        self.sas_specs = self.parse_sas_specs(sas_specs)
        self._numflux = len(self.sas_specs)
        self._fluxorder = list(self.sas_specs.keys())
        # get solute transport parameters
        if config and 'solute_parameters' in config.keys():
            solute_parameters = config['solute_parameters']
        elif solute_parameters:
            solute_parameters = _processinputs(solute_parameters)
        # defaults for solute transport
        self._default_parameters = {
            'mT_init': 0.,
            'C_old': 0.,
            'k1': 0.,
            'C_eq': 0.,
            'alpha': dict((flux, 1.) for flux in self._fluxorder),
            'observations': {}
        }
        self.solute_parameters = solute_parameters
        self.components_to_learn = components_to_learn

    def __repr__(self):
        """Creates a repr for the model"""
        repr = ''
        for flux, sas_spec in self.sas_specs.items():
            repr += f'flux = {flux}\n'
            repr += sas_spec.__repr__()
        return repr

    def parse_sas_specs(self, sas_specs):
        for flux, spec_in in sas_specs.items():
            spec = deepcopy(spec_in)
            assert flux in self.data_df.columns
            if isinstance(spec, SAS_Spec):
                sas_specs[flux] = spec
            else:
                assert isinstance(spec, dict)
                assert all([isinstance(component_spec, dict) for component_spec in spec.values()])
                sas_specs[flux] = SAS_Spec(spec, self.data_df)
        return sas_specs

    def copy_without_results(self):
        #returns a new instance of the model, but clears any results
        return Model(
            copy.deepcopy(self._data_df),
            config={
                "sas_specs": copy.deepcopy(self._sas_specs),
                "solute_parameters": copy.deepcopy(self._solute_parameters),
                "options": copy.deepcopy(self._options),
            },
            # components_to_learn=copy.deepcopy(self._components_to_learn),
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
    def result(self):
        """Results of running the sas model with the current parameters. See :ref:`results`
        """
        if self._result is None:
            raise AttributeError(
                "results are only defined once the model is run. Use .run() method to generate results ")
        else:
            return self._result

    @result.setter
    def result(self, result):
        raise AttributeError("Model results are read-only. Use .run() method to generate results ")

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
            raise TypeError('data_df must be either a pandas dataframe, or a path to a .csv file')
        self._timeseries_length = len(self._data_df)


    @property
    def options(self):
        """Options for running the model. See :ref:`options`
        """
        return self._options

    @options.setter
    def options(self, new_options):
        new_options = _processinputs(new_options)
        invalid_options = [optkey for optkey in new_options.keys()
                           if optkey not in self._default_options.keys()]
        if any(invalid_options):
            raise KeyError("Invalid options: {}".format(invalid_options))
        self._options.update(new_options)
        if self._options['max_age'] is None:
            self._options['max_age'] = self._timeseries_length
        assert self._options['max_age'] <= self._timeseries_length
        if self._options['sT_init'] is None:
            self._options['sT_init'] = np.zeros(self._options['max_age'])
        else:
            self._options['max_age'] = len(self._options['sT_init'])
        self._max_age = self._options['max_age']
        if self._options['record_state']==False:
            self._index_ts = np.array([self._timeseries_length-1])
        elif self._options['record_state']==True:
            self._index_ts = np.arange(self._timeseries_length)
        else:
            self._index_ts = np.where(self.data_df[self._options['record_state']])[0]

    @property
    def sas_specs(self):
        """Specification of the SAS functions. See :ref:`sasspec`"""
        return self._sas_specs

    @sas_specs.setter
    def sas_specs(self, new_sas_specs):
        self._sas_specs = _processinputs(new_sas_specs)
        self._numflux = len(self._sas_specs)
        self._fluxorder = list(self._sas_specs.keys())

    def set_sas_spec(self, flux, sas_spec):
        self._sas_specs[flux] = sas_spec
        self._sas_specs[flux].make_spec_ts(self.data_df)

    def set_component(self, flux, component):
        label = component.label
        self._sas_specs[flux].components[label] = component
        self._sas_specs[flux].make_spec_ts()

    def set_sas_fun(self, flux, label, sas_fun):
        self._sas_specs[flux].components[label].sas_fun = sas_fun
        self._sas_specs[flux].make_spec_ts()

    def get_component_labels(self):
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
            self._components_to_learn = OrderedDict((flux,
                                                     [label for label in self._sas_specs[flux]._componentorder if
                                                      label in label_dict[flux]]
                                                     ) for flux in self._fluxorder if flux in label_dict.keys())
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
            self.sas_specs[flux].update_from_parameter_list(parameter_list[starti: starti + nparams])
            starti += nparams

    @property
    def solute_parameters(self):
        """Parameters describing solute behavior. See :ref:`solspec`


        """
        return self._solute_parameters

    @solute_parameters.setter
    def solute_parameters(self, new_solute_parameters):
        # set parameters for solute transport
        # provide defaults if absent
        if new_solute_parameters is not None:
            new_solute_parameters = _processinputs(new_solute_parameters)
            self._solute_parameters = {}
            for sol, params in new_solute_parameters.items():
                self._solute_parameters[sol] = deepcopy(self._default_parameters)
                self.set_solute_parameters(sol, params)
            self._numsol = len(self._solute_parameters)
            self._solorder = list(self._solute_parameters.keys())
        else:
            self._numsol = 0
            self._solute_parameters = {}
            self._solorder = []

    def set_solute_parameters(self, sol, params):
        invalid_parameters = [paramkey for paramkey in params.keys()
                              if paramkey not in self._default_parameters.keys()]
        if any(invalid_parameters):
            raise KeyError("invalid parameters for {}: {}".format(sol, invalid_parameters))
        self._solute_parameters[sol].update(params)

    def _create_solute_inputs(self):
        numsol = max(self._numsol,1)
        C_J = np.zeros((self._timeseries_length, numsol), dtype=dtype)
        mT_init = np.zeros((self._max_age, numsol), dtype=dtype)
        C_old = np.zeros(numsol, dtype=dtype)
        k1 = np.zeros((self._timeseries_length, numsol), dtype=dtype)
        C_eq = np.zeros((self._timeseries_length, numsol), dtype=dtype)
        alpha = np.ones((self._timeseries_length, self._numflux, numsol), dtype=dtype)
        if self.solute_parameters is not None:
            def _get_array(param, N):
                if isinstance(param, str) and param in self.data_df:
                    return self.data_df[param].values
                elif isinstance(param, np.ndarray):
                    return param
                else:
                    return param * np.ones(N)

            for isol, sol in enumerate(self._solorder):
                C_J[:, isol] = self.data_df[sol].values
                C_old[isol] = self.solute_parameters[sol]['C_old']
                mT_init[:, isol] = _get_array(self.solute_parameters[sol]['mT_init'], self._max_age)
                k1[:, isol] = _get_array(self.solute_parameters[sol]['k1'], self._timeseries_length)
                C_eq[:, isol] = _get_array(self.solute_parameters[sol]['C_eq'],
                                           self._timeseries_length)
                for iflux, flux in enumerate(self._fluxorder):
                    alpha[:, iflux, isol] = _get_array(
                        self.solute_parameters[sol]['alpha'][flux], self._timeseries_length)
        return C_J, mT_init, C_old, alpha, k1, C_eq

    def _create_sas_lookup(self):
        nC_list = []
        nargs_list = []
        component_list = []
        component_type = []
        for flux in self._fluxorder:
            nC_list.append(len(self._sas_specs[flux].components))
            for component in self._sas_specs[flux]._componentorder:
                component_list.append(self.sas_specs[flux].components[component])
        nargs_list = [len(component.argsS) for component in component_list]
        component_type = [component.type for component in component_list]
        nC_total = np.sum(nC_list)
        nargs_total = np.sum(nargs_list)
        SAS_args = np.column_stack([[component.argsS] for component in component_list]).T
        P_list = np.column_stack([[component.argsP] for component in component_list]).T
        weights = np.column_stack([component.weights for component in component_list])
        return SAS_args, P_list, weights, component_type, nC_list, nC_total, nargs_list, nargs_total

    def run(self):
        """
        Call this method to run the model with current SAS specification, options, solute parameters, and timeseries dataframe. Results can then be accessed through the `.result` property of the model object
        :return: None
        """
        # water fluxes
        J = self.data_df[self.options['influx']].values
        Q = self.data_df[self._fluxorder].values
        sT_init = self.options['sT_init']
        timeseries_length = self._timeseries_length
        numflux = self._numflux
        #
        # SAS lookup table
        SAS_args, P_list, weights, component_type, nC_list, nC_total, nargs_list, nargs_total = self._create_sas_lookup()
        SAS_args = np.asfortranarray(SAS_args)
        P_list = np.asfortranarray(P_list)
        weights = np.asfortranarray(weights)
        #
        # Solutes
        C_J, mT_init, C_old, alpha, k1, C_eq = self._create_solute_inputs()
        numsol = max(self._numsol,1)
        #
        # options
        dt = self.options['dt']
        verbose = self.options['verbose']
        debug = self.options['debug']
        warning = self.options['warning']
        jacobian = self.options['jacobian']
        n_substeps = self.options['n_substeps']
        max_age = self.options['max_age']
        num_scheme = self.options['num_scheme']

        index_ts = self._index_ts

        # Enforce max_age
        sT_init = sT_init[:max_age]
        mT_init = mT_init[:max_age, :]

        # call the Fortran code
        fresult = solve(
            J, Q, np.asfortranarray(SAS_args.T), np.asfortranarray(P_list.T), np.asfortranarray(weights),
            sT_init, dt, verbose, debug, warning, jacobian,
            mT_init, np.asfortranarray(C_J), np.asfortranarray(alpha), np.asfortranarray(k1),
            np.asfortranarray(C_eq), C_old,
            n_substeps, component_type, nC_list, nargs_list, index_ts, num_scheme, numflux, numsol, max_age, timeseries_length, len(index_ts), nC_total, nargs_total)
        sT, pQ, WaterBalance, mT, mQ, mR, C_Q, dsTdSj, dmTdSj, dCdSj, SoluteBalance = fresult

        if self._numsol > 0:
            self._result = {'C_Q': C_Q}
            for isol, sol in enumerate(self._solorder):
                for iflux, flux in enumerate(self._fluxorder):
                    colname = sol + ' --> ' + flux
                    self._data_df[colname] = C_Q[:, iflux, isol]
        else:
            self._result = {}
        self._result.update({'sT': np.moveaxis(sT, -1, 0), 'pQ': np.moveaxis(pQ, -1, 0),
                             'WaterBalance': np.moveaxis(WaterBalance, -1, 0), 'dsTdSj':np.moveaxis(dsTdSj, -1, 0)})
        if self._numsol > 0:
            self._result.update({'mT': np.moveaxis(mT, -1, 0), 'mQ': np.moveaxis(mQ, -1, 0), 'mR': np.moveaxis(mR, -1, 0),
            'SoluteBalance': np.moveaxis(SoluteBalance, -1, 0),
                                 'dmTdSj':np.moveaxis(dmTdSj, -1, 0), 'dCdSj':dCdSj})

    def get_jacobian(self, mode='segment', logtransform=True):
        J = None
        self.jacobian = {}
        for isol, sol in enumerate(self._solorder):
            if 'observations' in self.solute_parameters[sol]:
                self.jacobian[sol] = {}
                iP = 0
                for isolflux, solflux in enumerate(self._fluxorder):
                    if solflux in self.solute_parameters[sol]['observations']:
                        J_seg = None
                        for iflux, flux in enumerate(self._comp2learn_fluxorder):
                            for label in self._components_to_learn[flux]:
                                nP = len(self.sas_specs[flux].components[label].sas_fun.P)
                                J_S = np.squeeze(self.result['dCdSj'][:, iP:iP+nP, isolflux, isol])
                                if mode == 'endpoint':
                                    pass
                                elif mode == 'segment':
                                    # To get the derivative with respect to the segment length, we add up the derivative w.r.t. the
                                    # endpoints that would be displaced by varying that segment
                                    A = np.triu(np.ones(nP), k=0)
                                    J_S = np.dot(A, J_S.T).T
                                    if logtransform:
                                        J_S = J_S * self.sas_specs[flux].components[label].sas_fun._parameter_list
                                if J_seg is None:
                                    J_seg = J_S
                                else:
                                    J_seg = np.c_[J_seg, J_S]
                            PQ = np.sum(self.result['pQ'][:, :, iflux], axis=0)
                            J_old = 1 - PQ
                            J_old_sol = np.zeros((self._timeseries_length, self._numsol))
                            J_old_sol[:, list(self._solorder).index(sol)] = J_old.T
                            J_sol = np.c_[J_seg, J_old_sol]
                        if J is None:
                            J = J_sol
                        else:
                            J = np.concatenate((J, J_sol), axis=0)
                        self.jacobian[sol][flux] = {}
                        self.jacobian[sol][flux]['seg'] = J_seg
                        self.jacobian[sol][flux]['C_old'] = J_old
                    iP +=nP
        return J

    def get_residuals(self):
        ri = None
        for isol, sol in enumerate(self._solorder):
            if 'observations' in self.solute_parameters[sol]:
                for iflux, flux in enumerate(self._comp2learn_fluxorder):
                    if flux in self.solute_parameters[sol]['observations']:
                        obs = self.solute_parameters[sol]['observations'][flux]
                        C_obs = self.data_df[obs]
                        iflux = list(self._fluxorder).index(flux)
                        isol = list(self._solorder).index(sol)
                        this_ri = self.result['C_Q'][:, iflux, isol] - C_obs.values
                        if ri is None:
                            ri = this_ri
                        else:
                            ri = np.concatenate((ri, this_ri), axis=0)
                        self.data_df[f'residual {flux}, {sol}, {obs}'] = this_ri
        return ri

    def get_obs_index(self):
        index = None
        for isol, sol in enumerate(self._solorder):
            if 'observations' in self.solute_parameters[sol]:
                for iflux, flux in enumerate(self._comp2learn_fluxorder):
                    if flux in self.solute_parameters[sol]['observations']:
                        obs = self.solute_parameters[sol]['observations'][flux]
                        C_obs = self.data_df[obs]
                        this_index = ~np.isnan(C_obs.values)
                        if index is None:
                            index = this_index
                        else:
                            index = np.concatenate((index, this_index), axis=0)
        return np.nonzero(index)[0]

    def _get_result(self, X, timestep=None, agestep=None, inputtime=None):
        if not timestep is None:
            # Only one can be given
            assert agestep is None
            assert inputtime is None
            return X[:,timestep]
        if not agestep is None:
            # Only one can be given
            assert timestep is None
            assert inputtime is None
            return X[agestep, :]
        if not inputtime is None:
            # Only one can be given
            assert agestep is None
            assert timestep is None
            return np.diagonal(X, offset=inputtime)
        return X

    def get_WaterBalance(self, **kwargs):
        X = self.result['WaterBalance']
        return self._get_result(X, **kwargs)

    def get_sT(self, **kwargs):
        X = self.result['sT']
        return self._get_result(X, **kwargs)

    def get_pQ(self, flux, **kwargs):
        iflux = list(self._fluxorder).index(flux)
        X = self.result['pQ'][:,:,iflux]
        return self._get_result(X, **kwargs)

    def get_mT(self, sol, **kwargs):
        isol = list(self._solorder).index(sol)
        X = self.result['mT'][:,:,isol]
        return self._get_result(X, **kwargs)

    def get_CT(self, sol, **kwargs):
        isol = list(self._solorder).index(sol)
        sT = self._get_result(self.result['sT'], **kwargs)
        mT = self._get_result(self.result['mT'][:,:,isol], **kwargs)
        return np.where(sT>0, mT / sT, np.NaN)

    def get_mR(self, sol, **kwargs):
        isol = list(self._solorder).index(sol)
        X = self.result['mR'][:,:,isol]
        return self._get_result(X, **kwargs)

    def get_SoluteBalance(self, sol, **kwargs):
        isol = list(self._solorder).index(sol)
        X = self.result['SoluteBalance'][:,:,isol]
        return self._get_result(X, **kwargs)

    def get_mQ(self, flux, sol, **kwargs):
        iflux = list(self._fluxorder).index(flux)
        isol = list(self._solorder).index(sol)
        X = self.result['mQ'][:,:,iflux, isol]
        return self._get_result(X, **kwargs)

    def get_ST(self, **kwargs):
        sT = self.get_sT()
        ST = np.cumsum(sT, axis=0) * self.options['dt']
        return self._get_result(ST, **kwargs)
