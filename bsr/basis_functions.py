#!/usr/bin/env python
"""Basis functions for fitting."""

import inspect
import numpy as np


def sigmoid_func(y):
    """Sigmoid activation function."""
    return 1. / (1. + np.exp(-y))


def sum_basis_funcs(basis_func, args_iterable, nfunc, x1, **kwargs):
    """Sum up some basis functions."""
    x2 = kwargs.pop('x2', None)
    global_bias = kwargs.pop('global_bias', basis_func.__name__[:2] == 'nn')
    sigmoid = kwargs.pop('sigmoid', basis_func.__name__ == 'nn_2d')
    adaptive = kwargs.pop('adaptive', False)
    assert isinstance(nfunc, int), str(nfunc)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    if adaptive:
        sum_max = int(np.round(args_iterable[0]))
        args_arr = args_iterable[1:]
    else:
        sum_max = nfunc
        args_arr = args_iterable
    # Deal with global bias
    if global_bias:
        y = args_arr[-1]
        args_arr = args_arr[:-1]
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


def get_param_names(basis_func):
    """Get a list of the parameters of the bais function (excluding x
    inputs).

    Parameters
    ----------
    basis_func: function

    Returns
    -------
    param_names: list of strs
    """
    param_names = list(inspect.signature(basis_func).parameters)
    for par in ['x', 'x1', 'x2']:
        try:
            param_names.remove(par)
        except ValueError:
            pass
    return param_names


def get_param_latex_names(basis_func):
    """Get a list of the parameters of LaTeX names for the params.

    Parameters
    ----------
    basis_func: function

    Returns
    -------
    param_names: list of strs
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
    param_names = get_param_names(basis_func)
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


def gg_1d(x, a, mu, sigma, beta):
    """1d generalised gaussian.

    Normalisation constant

    >>> const = a * beta / (2 * sigma * scipy.special.gamma(1.0 / beta))

    is excluded to avoid degeneracies between sigma and a, so the maxiumum of
    the basis function = a.
    """
    return a * np.exp((-1.0) * (np.absolute(x - mu) / sigma) ** beta)


def gg_2d(x1, x2, a, mu1, mu2, sigma1, sigma2, beta1, beta2, omega):
    """2d generalised gaussian"""
    # Rotate gen gaussian around the mean
    assert omega < 0.25 * np.pi and omega > -0.25 * np.pi, \
        "Angle=" + str(omega) + "must be in range +-pi/4=" + str(np.pi / 4)
    x1_new = x1 - mu1
    x2_new = x2 - mu2
    x1_new = np.cos(omega) * x1_new - np.sin(omega) * x2_new
    x2_new = np.sin(omega) * x1_new + np.cos(omega) * x2_new
    # NB we do not include means as x1_new and x2_new are relative to means
    return (a * gg_1d(x1_new, 1.0, 0, sigma1, beta1)
            * gg_1d(x2_new, 1.0, 0, sigma2, beta2))


def nn_1d(x, a, w_0, w_1):
    """1d neural network tanh."""
    return a * np.tanh(w_0 + (w_1 * x))


def nn_2d(x1, x2, a, w_0, w_1, w_2):
    """2d neural network tanh."""
    return a * np.tanh((w_1 * x1) + (w_2 * x2) + w_0)
