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
import numpy as np
import scipy.special
import bsr.basis_functions


class BasisFuncSum(object):

    """Loglikelihood for fitting a sum of basis functions to the data."""

    def __init__(self, data, func, nfunc, **kwargs):
        """
        Set up likelihood object's hyperparameter values.

        Parameters
        ----------
        """
        self.adaptive = kwargs.pop('adaptive', False)
        self.global_bias = kwargs.pop('global_bias', False)
        if kwargs:
            raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
        self.data = data
        assert data['x_error_sigma'] is None or data['x2'] is None, (
            'Not yet set up to deal with x errors in 2d')
        self.func = func
        self.nfunc = nfunc
        self.nargs = len(bsr.basis_functions.get_func_params(func))
        self.ndim = self.nfunc * self.nargs
        if self.adaptive:
            self.ndim += 1
        if self.global_bias:
            self.ndim += 1

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

    def sum_basis_funcs(self, x1, args_arr_in, x2=None):
        """
        Sums basis functions.
        """
        # Deal with adaptive nfunc specification
        if self.adaptive:
            sum_max = int(np.round(args_arr_in[0]))
            args_arr = args_arr_in[1:]
        else:
            sum_max = self.nfunc
            args_arr = args_arr_in
        # Deal with global bias
        if self.global_bias:
            y = args_arr[-1]
            args_arr = args_arr[:-1]
        else:
            y = 0.0
        # Sum basis functions
        if x2 is None:
            for i in range(sum_max):
                y += self.func(x1, *args_arr[i::self.nfunc])
        else:
            for i in range(sum_max):
                y += self.func(x1, x2, *args_arr[i::self.nfunc])
        return y

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
            Any derived parameters.
        """
        # check there are the expected number of params
        assert theta.shape[0] == self.ndim, (
            'theta.shape[0]={0} != self.ndim={1}'.format(
                theta.shape[0], self.ndim))
        if self.data['x_error_sigma'] is None:
            # No x errors so no need to integrate
            ytheta = self.sum_basis_funcs(
                self.data['x1'], theta, x2=self.data['x2'])
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
            Y = self.sum_basis_funcs(X, theta)
            for ind, y_ind in np.ndenumerate(self.data['y']):
                logl += np.log(scipy.integrate.simps(
                    self.integrand(X, Y, self.data['x1'][ind], y_ind), x=X))
        return logl, []
