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
    global_bias = kwargs.pop('global_bias', False)
    sigmoid = kwargs.pop('sigmoid', False)
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


def prop_layer(inputs, w_arr, bias):
    """Propogate a neural network layer.

    outputs = dot(w_arr, inputs) + bias

    Parameters
    ----------
    inputs: 2d numpy array of dimension (n_input, 1)
    w_arr: 2d numpy array of dimension (n_input, n_output)
    bias: 2d numpy array of dimension (n_input, 1)
    """
    assert inputs.ndim == 2, inputs.ndim
    assert w_arr.ndim == 2, w_arr.ndim
    assert bias.ndim == 2, bias.ndim
    assert inputs.shape[1] == 1, inputs.shape
    assert bias.shape[1] == 1, bias.shape
    assert w_arr.shape == (bias.shape[0], inputs.shape[0]), (
        'w_arr.shape={}, bias.shape={}, inputs.shape={}'.format(
            w_arr.shape, bias.shape, inputs.shape))
    out = np.matmul(w_arr, inputs) + bias
    return out


def nn_num_params(n_nodes):
    """number of parameters a neural network needs."""
    assert isinstance(n_nodes, list)
    assert len(n_nodes) >= 2
    n_param = n_nodes[-1] + 1  # number of params for final activation
    for i, _ in enumerate(n_nodes[:-1]):
        n_param += (n_nodes[i] + 1) * n_nodes[i + 1]
    return n_param


def nn_split_params(params, n_nodes):
    """Split a 1d parameter vector into lists of the biases and weight arrays
    for each propogation step."""
    assert params.shape == (nn_num_params(n_nodes),), (
        'params.shape={} - I was expecting {}'.format(
            params.shape, nn_num_params(n_nodes)))
    w_arr_list = []
    bias_list = []
    # seperately extract the weights for mapping final layer to scalar output
    bias_final = np.atleast_2d(params[0])
    w_arr_final = np.atleast_2d(params[1:1 + n_nodes[-1]])
    n_par = n_nodes[-1] + 1  # counter for parameters already extracted
    for i, _ in enumerate(n_nodes[:-1]):
        # extract bias
        delta = n_nodes[i + 1]
        bias = params[n_par:n_par + delta]
        n_par += delta
        # reshape to 2d
        bias = np.atleast_2d(bias).T
        bias_list.append(bias)
        # extract w_arr
        delta = n_nodes[i] * n_nodes[i + 1]
        w_arr = params[n_par:n_par + delta]
        n_par += delta
        # reshape to 2d
        w_arr = w_arr.reshape((n_nodes[i + 1], n_nodes[i]), order='F')
        w_arr_list.append(w_arr)
    # add back in final scalar output as layer with 1 node
    w_arr_list.append(w_arr_final)
    bias_list.append(bias_final)
    assert (n_par,) == params.shape, (
        'np={}, whereas params.shape={}'.format(np, params.shape))
    return w_arr_list, bias_list


def nn_flatten_params(w_arr_list, bias_list):
    """Inverse of nn_split_params."""
    flat_list = [bias_list[-1].flatten(order='F'),
                 w_arr_list[-1].flatten(order='F')]
    for i, w_arr in enumerate(w_arr_list[:-1]):
        flat_list.append(bias_list[i].flatten(order='F'))
        flat_list.append(w_arr.flatten(order='F'))
    return np.concatenate(flat_list)


def nn_eval(x, params, nodes, **kwargs):
    """Get output from a neural network."""
    act_func = kwargs.pop('act_func', np.tanh)
    out_act_func = kwargs.pop('out_act_func', sigmoid_func)
    if isinstance(nodes, int):
        nodes = [nodes]
    assert isinstance(nodes, list), 'nodes={} is not list'.format(nodes)
    inputs = np.atleast_2d(x).T
    n_nodes = [inputs.shape[0]] + nodes
    w_arr_list, bias_list = nn_split_params(params, n_nodes)
    for i, w_arr in enumerate(w_arr_list):
        print(i)
        inputs = prop_layer(inputs, w_arr, bias_list[i])
        if i == len(w_arr_list) - 1:
            inputs = out_act_func(inputs)
        else:
            inputs = act_func(inputs)
    assert inputs.shape == (1, 1)
    return inputs[0, 0]


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


def get_bf_param_latex_names(basis_func):
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
    param_names = get_bf_param_names(basis_func)
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
    x1_new = x1 - mu1
    x2_new = x2 - mu2
    x1_new = np.cos(omega) * x1_new - np.sin(omega) * x2_new
    x2_new = np.sin(omega) * x1_new + np.cos(omega) * x2_new
    # NB we do not include means as x1_new and x2_new are relative to means
    return (a * gg_1d(x1_new, 1.0, 0, sigma1, beta1)
            * gg_1d(x2_new, 1.0, 0, sigma2, beta2))


def ta_1d(x, a, w_0, w_1):
    """1d tanh function."""
    return a * np.tanh(w_0 + (w_1 * x))


def ta_2d(x1, x2, a, w_0, w_1, w_2):
    """2d tanh function."""
    return a * np.tanh((w_1 * x1) + (w_2 * x2) + w_0)
