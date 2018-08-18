#!/usr/bin/env python
"""
Loglikelihood functions for use with PyPolyChord (PolyChord's python wrapper).

PolyChord v1.14 requires likelihoods to be callables with parameter and return
types:

Parameters
----------
theta: float or 1d numpy array

Returns
-------
logl: float
    Loglikelihood.
phi: list of length nderived
    Any derived parameters.

We use classes with the loglikelihood defined in the __call__ property, as
this provides convenient way of storing other information such as
hyperparameter values. These objects can be used in the same way as functions
due to python's "duck typing" (alternatively you can define likelihoods
using functions).
"""
import os
import warnings
import numpy as np
import scipy.integrate
import bsr.basis_functions as bf
import bsr.neural_networks as nn


class FittingLikelihood(object):

    """Loglikelihood for fitting a sum of basis functions to the data."""

    def __init__(self, data, function, nfunc, **kwargs):
        """
        Set up likelihood object's hyperparameter values.
        """
        self.adaptive = kwargs.pop('adaptive', False)
        self.global_bias = kwargs.pop('global_bias', False)
        self.use_hyper = kwargs.pop('use_hyper', function.__name__[:2] == 'nn')
        if kwargs:
            raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
        self.data = data
        assert data['x_error_sigma'] is None or data['x2'] is None, (
            'Not yet set up to deal with x errors in 2d')
        self.function = function
        self.nfunc = nfunc
        if self.function.__name__[:2] == 'nn':
            assert not self.global_bias
            assert isinstance(self.nfunc, list)
            assert len(nfunc) >= 2, nfunc
            if data['x2'] is None:
                assert nfunc[0] == 1
            else:
                assert nfunc[0] == 2
            self.ndim = nn.nn_num_params(self.nfunc)
        elif self.function.__name__ == 'adfam_gg_ta_1d':
            assert self.adaptive
            assert not self.global_bias
            self.ndim = self.nfunc * len(bf.get_bf_param_names(bf.gg_1d))
            self.ndim += 1  # For adaptive family parameter
        else:
            self.ndim = self.nfunc * len(bf.get_bf_param_names(function))
        if self.global_bias:
            self.ndim += 1
        if self.adaptive:
            self.ndim += 1
        if self.use_hyper:
            self.ndim += 1

    def fit(self, theta, x1, x2=None):
        """
        Fit the data using the model and parameters theta.
        """
        if self.function.__name__[:2] == 'nn':
            if self.use_hyper:
                theta = theta[:-1]  # remove hyperparameter
            if self.adaptive:
                theta = nn.adaptive_theta(theta, self.nfunc)
            if isinstance(x1, (int, float)):
                # This is just a single example (M=1), so we will output a
                # scalar
                if x2 is None:
                    x = x1
                else:
                    x = np.asarray([x1, x2])
                return self.function(x, theta, self.nfunc)
            else:
                # We need to reshape to array with shape (ndim, M) where M is
                # number of data points
                if x2 is None:
                    assert x1.ndim == 1, x1.shape
                    x = x1.reshape((1, x1.shape[0]))
                    return self.function(x, theta, self.nfunc)
                else:
                    assert x1.shape == x2.shape
                    x = np.zeros((2, x1.size))
                    x[0, :] = x1.flatten(order='C')
                    x[1, :] = x2.flatten(order='C')
                    out = self.function(x, theta, self.nfunc)
                    return out.reshape(x1.shape, order='C')
        elif self.function.__name__ == 'adfam_gg_ta_1d':
            return self.function(
                x1, theta, self.nfunc, x2=x2,
                global_bias=self.global_bias, adaptive=self.adaptive)
        else:
            return bf.sum_basis_funcs(
                self.function, theta, self.nfunc, x1, x2=x2,
                global_bias=self.global_bias, adaptive=self.adaptive)

    def get_param_names(self):
        """Get list of parameter names as str."""
        if self.function.__name__[:2] == 'nn':
            return nn.get_nn_param_names(
                self.nfunc, use_hyper=self.use_hyper)
        else:
            if self.function.__name__ == 'adfam_gg_ta_1d':
                bf_params = bf.get_bf_param_names(bf.gg_1d)
            else:
                bf_params = bf.get_bf_param_names(self.function)
            param_names = []
            for param in bf_params:
                for i in range(self.nfunc):
                    param_names.append('{0}_{1}'.format(param, i + 1))
            if self.global_bias:
                param_names = ['a_0'] + param_names
            if self.adaptive:
                param_names = ['N'] + param_names
            if self.function.__name__ == 'adfam_gg_ta_1d':
                param_names = ['T'] + param_names
            assert len(param_names) == self.ndim
            return param_names

    def get_param_latex_names(self):
        """Get list of parameter names as str."""
        if self.function.__name__[:2] == 'nn':
            return nn.get_nn_param_latex_names(
                self.nfunc, use_hyper=self.use_hyper)
        elif self.function.__name__ == 'adfam_gg_ta_1d':
            return self.get_param_names()
        else:
            bf_params = bf.get_param_latex_names(
                bf.get_bf_param_names(self.function))
            param_names = []
            for param in bf_params:
                assert param[-1] == '$'
                if param[-2] == '}':
                    for i in range(self.nfunc):
                        param_names.append(
                            param[:-2] + ',' + str(i + 1) + '}$')
                else:
                    for i in range(self.nfunc):
                        param_names.append('{0}_{1}$'.format(
                            param[:-1], i + 1))
            if self.global_bias:
                param_names = ['$a_0$'] + param_names
            if self.adaptive:
                param_names = ['$N$'] + param_names
            assert len(param_names) == self.ndim
            return param_names

    def get_file_root(self, nlive, num_repeats, dynamic_goal=None):
        """Get a standard string for save names."""
        if self.adaptive:
            method = 'adaptive'
        else:
            method = 'vanilla'
        root_name = self.data['data_name'] + '_' + method
        root_name += '_' + self.function.__name__
        if self.function.__name__[:2] == 'nn':
            assert isinstance(self.nfunc, list)
            root_name += '_'
            root_name += (str(self.nfunc).replace('[', '').replace(']', '')
                          .replace(',', '').replace(' ', '_'))
        else:
            root_name += '_' + str(self.nfunc) + 'funcs'
        root_name += '_' + str(nlive) + 'nlive'
        root_name += '_' + str(num_repeats) + 'reps'
        root_name += '_dg' + str(dynamic_goal)
        return root_name.replace('.', '_')

    def fit_fgivenx(self, x1, theta):
        """Wrapper for correct arg order for fgivenx."""
        return self.fit(theta, x1)

    def fit_mean(self, theta, x1, x2=None, **kwargs):
        """Get the mean fit for each row of a 2d array of thetas. Optionally
        you can also get the weighted mean."""
        w_rel = kwargs.pop('w_rel', None)
        evens = kwargs.pop('evens', True)
        assert theta.ndim == 2
        if w_rel is not None:
            assert theta.shape[0] == w_rel.shape[0], (
                '{} != {}'.format(theta.shape[0], w_rel.shape[0]))
            w_rel /= w_rel.max()
            w_rel_sum = np.sum(w_rel)
            if evens:
                state = np.random.get_state()
                np.random.seed(0)
                w_rel = (np.random.random(w_rel.shape) < w_rel).astype(int)
                state = np.random.set_state(state)
            inds = np.nonzero(w_rel)[0]
            if inds.shape[0] == 0:
                warnings.warn(
                    ('fit_mean: No points with nonzero weight! {} points '
                     'were input with sum(w_rel)={}'.format(
                         theta.shape[0], w_rel_sum)), UserWarning)
                return np.zeros(x1.shape)
            ys = np.apply_along_axis(self.fit, 1, theta[inds, :], x1, x2)
            ys *= w_rel[inds][:, np.newaxis, np.newaxis]
            return np.sum(ys, axis=0) / np.sum(w_rel[inds])
        else:
            ys = np.apply_along_axis(self.fit, 1, theta, x1, x2)
            return np.mean(ys, axis=0)

    def write_cpp_config(self, file_root, base_dir='chains'):
        """Writes a config file for the C++ version of this likelihood,
        including the data and model to fit. This is saved to the path:

        base_dir/file_root.cfg

        Parameters
        ----------
        file_root: str
        base_dir: str, optional
        """
        filepath = os.path.join(base_dir, file_root + '.cfg')
        with open(filepath, 'w') as cfg_file:
            cfg_file.write('nfunc={}\n'.format(self.nfunc))
            cfg_file.write('fit_func={}\n'.format(self.function.__name__))
            cfg_file.write('y_error_sigma={}\n'.format(
                self.data['y_error_sigma']))
            if self.data['x_error_sigma'] is None:
                cfg_file.write('x_error_sigma=0\n')
            else:
                cfg_file.write('x_error_sigma={}\n'.format(
                    self.data['x_error_sigma']))
            cfg_file.write('x1={}\n'.format(cpp_format_array(self.data['x1'])))
            cfg_file.write('y={}\n'.format(cpp_format_array(self.data['y'])))
            if self.data['x2'] is not None:
                cfg_file.write('x2={}\n'.format(cpp_format_array(
                    self.data['x2'])))
            cfg_file.write('adaptive={}\n'.format(self.adaptive))


    def integrand(self, X, Y, xj, yj):
        """Helper function for integrating."""
        expo = ((xj - X) / self.data['x_error_sigma']) ** 2
        expo += ((yj - Y) / self.data['y_error_sigma']) ** 2
        return np.exp(-0.5 * expo)

    def __call__(self, theta):
        """
        Calculate loglikelihood(theta), as well as any derived parameters.

        Parameters
        ----------
        theta: float or 1d numpy array

        Returns
        -------
        logl: float
            Loglikelihood
        phi: list of length nderived
            This likelihood does not use any derived parameters and therefore
            just returns an empty list.
        """
        # check there are the expected number of params
        assert theta.shape == (self.ndim,), (
            'theta.shape={0} != (self.ndim,)=({1},)'.format(
                theta.shape, self.ndim))
        if self.data['x_error_sigma'] is None:
            # No x errors so no need to integrate
            ytheta = self.fit(theta, self.data['x1'], self.data['x2'])
            logl = np.sum(log_gaussian_given_r(
                abs(ytheta - self.data['y']), self.data['y_error_sigma']))
        else:
            # x errors require integration
            # First calculate normalisation constants for all points'
            # contributions in one go
            logl = -np.log(2 * np.pi * self.data['y_error_sigma']
                           * self.data['x_error_sigma']
                           * (self.data['x1max'] - self.data['x1min']))
            logl *= self.data['y'].size
            # Now integrate:
            # number of points needs to be high (~1000) or polychord gives the
            # 'nondeterministic likelihood' warning as likelihoods evaluate
            # inconsistently due to rounding errors on the integration
            nsamp = 1001
            X = np.linspace(self.data['x1min'], self.data['x1max'], nsamp)
            dx = (self.data['x1max'] - self.data['x1min']) / (nsamp - 1)
            Y = self.fit(theta, X)
            for i, y_ind in enumerate(self.data['y']):
                contribution = scipy.integrate.simps(
                    self.integrand(X, Y, self.data['x1'][i], y_ind), dx=dx)
                if contribution == 0:
                    logl = -np.inf
                    break
                else:
                    logl += np.log(contribution)
        return logl, []


# Helper functions
# ----------------

def simpson(integrand, dx=1):
    """1d simpson integration. Should give same answer as scipy.integrate.simps
    when integrand.shape[0] is odd."""
    out = np.sum(integrand[:-2:2] + integrand[2::2] + 4 * integrand[1::2])
    return (dx / 3.0) * out




def log_gaussian_given_r(r, sigma, n_dim=1):
    """
    Returns the natural log of a normalised, uncorrelated gaussian with
    equal variance in all n_dim dimensions.
    """
    logl = -0.5 * ((r ** 2) / (sigma ** 2))
    # normalise
    logl -= n_dim * np.log(sigma)
    logl -= np.log(2 * np.pi) * (n_dim / 2.0)
    return logl

def cpp_format_array(array):
    """Transforms array into string of type read in C++ likelihood config
    file."""
    format_dict = {'\n': '',
                   '[': '',
                   ']': ''}
    # Save input printoptions so we don't edit them
    printoptions_dict_in = np.get_printoptions()
    np.set_printoptions(
        suppress=True,  # supress standard form
        threshold=1000000,  # elements triggering summary
        floatmode='fixed',
        linewidth=10000000)
    vals_str = np.array2string(array.flatten(order='C'), precision=10,
                               separator=' ')
    for key, item in format_dict.items():
        vals_str = vals_str.replace(key, item)
    vals_str.replace('  ', ' ')
    vals_str.replace('  ', ' ')
    vals_str.replace('  ', ' ')
    assert 'e' not in vals_str, (
        'Should not use scientific notation! vals_str=' + vals_str)
    np.set_printoptions(**printoptions_dict_in)  # restore previous setting
    return vals_str
