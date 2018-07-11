#!/usr/bin/env python
"""Neural network fitting functions."""

import copy
import numpy as np
import bsr.basis_functions as bf


def nn_fit(x, params, n_nodes, **kwargs):
    """Get output from a neural network."""
    act_func = kwargs.pop('act_func', np.tanh)
    out_act_func = kwargs.pop('out_act_func', bf.sigmoid_func)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    assert isinstance(n_nodes, list), 'n_nodes={} is not list'.format(n_nodes)
    assert len(n_nodes) >= 2, n_nodes
    if isinstance(x, (float, int)):
        inputs = np.atleast_2d(x)
    else:
        assert isinstance(x, np.ndarray)
        if x.ndim == 1:
            # Assume this is a single example (M=1). In principle it could be M
            # 1d examples - this will throw an assertion error as it will not
            # have n_nodes[0] == inputs.shape[0]
            inputs = np.atleast_2d(x).T
        else:
            assert x.ndim == 2, x.ndim
            inputs = copy.deepcopy(x)
    assert n_nodes[0] == inputs.shape[0], (
        'inputs.shape={} n_nodes={}'.format(inputs.shape, n_nodes))
    w_arr_list, bias_list = nn_split_params(params, n_nodes)
    for i, w_arr in enumerate(w_arr_list):
        inputs = prop_layer(inputs, w_arr, bias_list[i])
        if i == len(w_arr_list) - 1:
            inputs = out_act_func(inputs)
        else:
            inputs = act_func(inputs)
    assert inputs.ndim == 2 and inputs.shape[0] == 1
    if inputs.shape[1] == 1:
        return inputs[0, 0]
    else:
        return inputs[0, :]


def get_nn_param_names(n_nodes):
    """get names for the neural network parameters."""
    assert isinstance(n_nodes, list), 'n_nodes={} is not list'.format(n_nodes)
    param_names = ['a_{}'.format(i) for i in range(n_nodes[-1] + 1)]
    for layer, n_node_layer in enumerate(n_nodes[:-1]):
        for i_from in range(n_node_layer + 1):
            for i_too in range(1, n_nodes[layer + 1] + 1):
                param_names.append('w_{}_{}_{}'.format(i_from, i_too, layer))
    assert len(param_names) == nn_num_params(n_nodes), param_names
    return param_names


def adaptive_theta(theta, n_nodes):
    """Return a theta vector with the nodes that are not used zeroed out."""
    assert theta.shape == (1 + nn_num_params(n_nodes),)
    nfunc = int(np.round(int(theta[0])))
    theta = theta[1:]
    assert nfunc <= n_nodes[-1]
    for node in range(nfunc, n_nodes[-1]):
        theta[node + 1] = 0
    if len(n_nodes) > 2:
        assert len(set(n_nodes[1:])) == 1
        counter = n_nodes[-1] + 1
        for layer, n_node_layer in enumerate(n_nodes[:-1]):
            for i_from in range(n_node_layer + 1):
                for _ in range(1, n_nodes[layer + 1] + 1):
                    if layer > 0 and i_from > nfunc:
                        theta[counter] = 0
                    counter += 1
    return theta


def get_nn_param_latex_names(n_nodes):
    """get names for the neural network parameters."""
    assert isinstance(n_nodes, list), 'n_nodes={} is not list'.format(n_nodes)
    param_names = ['$a_{}$'.format(i) for i in range(n_nodes[-1] + 1)]
    for layer, n_node_layer in enumerate(n_nodes[:-1]):
        for i_from in range(n_node_layer + 1):
            for i_too in range(1, n_nodes[layer + 1] + 1):
                param_names.append('$w_{{{},{}}}^{{[{}]}}$'.format(
                    i_from, i_too, layer))
    assert len(param_names) == nn_num_params(n_nodes), param_names
    return param_names


def prop_layer(inputs, w_arr, bias):
    """Propogate a neural network layer.

    outputs = dot(w_arr, inputs) + bias

    This works for vectorised applications of M data points at once, as well as
    for a single data point (M=1).

    Parameters
    ----------
    inputs: 2d numpy array of shape (n_input, M)
    w_arr: 2d numpy array of shape (n_output, n_input)
    bias: 2d numpy array of shape (n_output, 1)

    Returns
    -------
    out: 2d numpy array of shape (n_output, M)
    """
    check_shapes(inputs, w_arr, bias)
    out = np.matmul(w_arr, inputs) + bias
    assert out.shape == (w_arr.shape[0], inputs.shape[1]), (
        'out.shape={}, w_arr.shape={}, inputs.shape={}'.format(
            out.shape, w_arr.shape, inputs.shape))
    return out


def check_shapes(inputs, w_arr, bias):
    """Check the parameters for a neural network propogation step all have the
    correct shapes.

    This works for vectorised applications of M data points at once, as well as
    for a single data point (M=1).

    Parameters
    ----------
    inputs: 2d numpy array of shape (n_input, M)
    w_arr: 2d numpy array of shape (n_output, n_input)
    bias: 2d numpy array of shape (n_output, 1)
    """
    assert inputs.ndim == 2, inputs.ndim
    assert w_arr.ndim == 2, w_arr.ndim
    assert bias.ndim == 2, bias.ndim
    assert w_arr.shape == (bias.shape[0], inputs.shape[0]), (
        'w_arr.shape={}, bias.shape={}, inputs.shape={}'.format(
            w_arr.shape, bias.shape, inputs.shape))


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
