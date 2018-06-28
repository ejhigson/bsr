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
import warnings
import numpy as np
import scipy.special
import bsr.basis_functions as bf


class BasisFuncFit(object):

    """Loglikelihood for fitting a sum of basis functions to the data."""

    def __init__(self, data, basis_func, nfunc, **kwargs):
        """
        Set up likelihood object's hyperparameter values.
        """
        self.adaptive = kwargs.pop('adaptive', False)
        self.global_bias = kwargs.pop(  # default True if nn, False otherwise
            'global_bias', basis_func.__name__[:2] == 'nn')
        if kwargs:
            raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
        self.data = data
        assert data['x_error_sigma'] is None or data['x2'] is None, (
            'Not yet set up to deal with x errors in 2d')
        self.basis_func = basis_func
        self.nfunc = nfunc
        self.nargs = len(bf.get_param_names(basis_func))
        self.ndim = self.nfunc * self.nargs
        if self.adaptive:
            self.ndim += 1
        if self.global_bias:
            self.ndim += 1

    def get_param_names(self):
        """Get list of parameter names as str."""
        bf_params = bf.get_param_names(self.basis_func)
        param_names = []
        if self.adaptive:
            param_names.append('B')
        for param in bf_params:
            for i in range(self.nfunc):
                param_names.append('{0}_{1}'.format(param, i + 1))
        if self.global_bias:
            param_names.append('C')
        assert len(param_names) == self.ndim
        return param_names

    def get_param_latex_names(self):
        """Get list of parameter names as str."""
        bf_params = bf.get_param_latex_names(
            self.basis_func)
        param_names = []
        if self.adaptive:
            param_names.append('$B$')
        for param in bf_params:
            assert param[-1] == '$'
            if param[-2] == '}':
                for i in range(self.nfunc):
                    param_names.append(param[:-2] + ',' + str(i + 1) + '}$')
            else:
                for i in range(self.nfunc):
                    param_names.append('{0}_{1}$'.format(param[:-1], i + 1))
        if self.global_bias:
            param_names.append('$C$')
        assert len(param_names) == self.ndim
        return param_names

    @staticmethod
    def log_gaussian_given_r(r, sigma, n_dim=1):
        """
        Returns the natural log of a normalised,  uncorrelated gaussian with
        equal variance in all n_dim dimensions.
        """
        logl = -0.5 * ((r ** 2) / (sigma ** 2))
        # normalise
        logl -= n_dim * np.log(sigma)
        logl -= np.log(2 * np.pi) * (n_dim / 2.0)
        return logl

    def integrand(self, X, Y, xj, yj):
        """Helper function for integrating."""
        expo = ((xj - X) / self.data['x_error_sigma']) ** 2
        expo += ((yj - Y) / self.data['y_error_sigma']) ** 2
        return np.exp(-0.5 * expo)

    def get_file_root(self, nlive, num_repeats, dynamic_goal=None):
        """Get a standard string for save names."""
        if self.adaptive:
            method = 'adaptive'
        else:
            method = 'vanilla'
        root_name = self.data['data_name'] + '_' + method
        root_name += '_' + self.basis_func.__name__
        root_name += '_' + str(self.nfunc) + 'funcs'
        root_name += '_' + str(nlive) + 'nlive'
        root_name += '_' + str(num_repeats) + 'reps'
        root_name += '_dg' + str(dynamic_goal)
        return root_name.replace('.', '_')

    def fit(self, theta, x1, x2=None):
        """
        Fit the data using the model and parameters theta.
        """
        # Deal with adaptive nfunc specification
        return bf.sum_basis_funcs(
            self.basis_func, theta, self.nfunc, x1, x2=x2,
            global_bias=self.global_bias, adaptive=self.adaptive)


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
            logl = np.sum(self.log_gaussian_given_r(
                abs(ytheta - self.data['y']), self.data['y_error_sigma']))
        else:
            # x errors require integration
            # First calculate normalisation constants for all points'
            # contributions in one go
            logl = np.log(2 * np.pi * self.data['y_error_sigma']
                          * self.data['x_error_sigma']
                          * (self.data['x1max'] - self.data['x1min']))
            logl *= self.data['y'].size
            # Now integrate:
            # number of points needs to be high (~1000) or polychord gives the
            # 'nondeterministic likelihood' warning as likelihoods evaluate
            # inconsistently due to rounding errors on the integration
            X = np.linspace(self.data['x1min'], self.data['x1max'], 1000)
            Y = self.fit(theta, X)
            for ind, y_ind in np.ndenumerate(self.data['y']):
                logl += np.log(scipy.integrate.simps(
                    self.integrand(X, Y, self.data['x1'][ind], y_ind), x=X))
        return logl, []
