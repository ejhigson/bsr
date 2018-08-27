#!/usr/bin/env python
"""Basis functions for fitting."""

import inspect
import numpy as np


# Basis functions
# ---------------


def gg_1d(x, a, mu, sigma, beta):
    """1d generalised gaussian.

    Normalisation constant

    >>> const = beta / (2 * sigma * scipy.special.gamma(1.0 / beta))

    is excluded to avoid degeneracies between sigma and a, so the maxiumum of
    the basis function = a.
    """
    return a * np.exp(-((np.absolute(x - mu) / sigma) ** beta))


def gg_2d(x1, x2, a, mu1, mu2, sigma1, sigma2, beta1, beta2, omega):
    """2d generalised gaussian.

    omega is counterclockwise rotation angle in (x1, x2) plane."""
    # Rotate gen gaussian around the mean
    x1_new = np.cos(omega) * (x1 - mu1) - np.sin(omega) * (x2 - mu2)
    x2_new = np.sin(omega) * (x1 - mu1) + np.cos(omega) * (x2 - mu2)
    # NB we do not include means as x1_new and x2_new are relative to means
    return (a * gg_1d(x1_new, 1.0, 0, sigma1, beta1)
            * gg_1d(x2_new, 1.0, 0, sigma2, beta2))


def ta_1d(x, a, w_0, w_1):
    """1d tanh function."""
    return a * np.tanh(w_0 + (w_1 * x))


def ta_2d(x1, x2, a, w_0, w_1, w_2):
    """2d tanh function."""
    return a * np.tanh((w_1 * x1) + (w_2 * x2) + w_0)


def adfam_gg_ta_1d(x1, theta, nfunc, **kwargs):
    """Adaptive family selection between ta_1d and gg_1d."""
    assert kwargs.get('x2', None) is None
    assert kwargs.get('adaptive')
    try:
        family = int(np.round(theta[0]))
        assert family in [1, 2], (
            "family param T={} not in [1, 2]. theta[0]={}".format(
                family, theta[0]))
    except ValueError:
        if np.isnan(theta[0]):
            return np.nan * x1  # ensure same type and shape as x1
        else:
            raise
    if family == 1:
        return sum_basis_funcs(gg_1d, theta[1:], nfunc, x1, **kwargs)
    else:
        # ta_1d has 3 params (excluding x1), whereas gg_1d has 4
        return sum_basis_funcs(ta_1d, theta[1:-nfunc], nfunc, x1, **kwargs)

# Helper functions
# ----------------


def sum_basis_funcs(basis_func, args_iterable, nfunc, x1, **kwargs):
    """Sum up some basis functions."""
    x2 = kwargs.pop('x2', None)
    global_bias = kwargs.pop('global_bias', False)
    sigmoid = kwargs.pop('sigmoid', False)
    adaptive = kwargs.pop('adaptive', False)
    assert isinstance(nfunc, int), str(nfunc)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    if adaptive:
        try:
            sum_max = int(np.round(args_iterable[0]))
            assert 1 <= sum_max <= nfunc, (
                "sum_max={} nfunc={}".format(sum_max, nfunc))
        except ValueError:
            if np.isnan(args_iterable[0]):
                return np.nan * x1  # ensure same type and shape as x1
            else:
                raise
        args_arr = args_iterable[1:]
    else:
        sum_max = nfunc
        args_arr = args_iterable
    # Deal with global bias
    if global_bias:
        y = args_arr[0]
        args_arr = args_arr[1:]
    else:
        y = 0.0
    # Sum basis functions
    if x2 is None:
        for i in range(sum_max):
            y += basis_func(x1, *args_arr[i::nfunc])
    else:
        for i in range(sum_max):
            y += basis_func(x1, x2, *args_arr[i::nfunc])
    if sigmoid:
        y = sigmoid_func(y)
    return y


def get_bf_param_names(basis_func):
    """Get a list of the parameters of the bais function (excluding x
    inputs).

    Parameters
    ----------
    basis_func: function

    Returns
    -------
    param_names: list of strs
    """
    assert basis_func.__name__ in ['gg_1d', 'gg_2d', 'ta_1d', 'ta_2d']
    param_names = list(inspect.signature(basis_func).parameters)
    for par in ['x', 'x1', 'x2']:
        try:
            param_names.remove(par)
        except ValueError:
            pass
    return param_names


def get_param_latex_names(param_names):
    """Get a list of the parameters of LaTeX names for the params.

    Parameters
    ----------
    param_names: list of strs

    Returns
    -------
    latex_names: list of strs
    """
    latex_map = {'mu': r'\mu',
                 'mu1': r'\mu_{1}',
                 'mu2': r'\mu_{2}',
                 'sigma': r'\sigma',
                 'sigma1': r'\sigma_{1}',
                 'sigma2': r'\sigma_{2}',
                 'beta': r'\beta',
                 'beta1': r'\beta_{1}',
                 'beta2': r'\beta_{2}',
                 'omega': r'\Omega'}
    latex_names = []
    for param in param_names:
        try:
            latex_names.append(latex_map[param])
        except KeyError:
            if param[:2] == 'w_':
                latex_names.append('w_{' + param[2:] + '}')
            else:
                latex_names.append(param)
    return ['${}$'.format(name) for name in latex_names]


def sigmoid_func(y):
    """Sigmoid activation function."""
    return 1. / (1. + np.exp(-y))
