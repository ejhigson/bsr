#!/usr/bin/env python
"""Neural network fitting functions."""

import numpy as np
import bsr.basis_functions as bf


def nn_fit(x, params, nodes, **kwargs):
    """Get output from a neural network."""
    act_func = kwargs.pop('act_func', np.tanh)
    out_act_func = kwargs.pop('out_act_func', bf.sigmoid_func)
    assert isinstance(nodes, list), 'nodes={} is not list'.format(nodes)
    inputs = np.atleast_2d(x).T
    n_nodes = [inputs.shape[0]] + nodes
    w_arr_list, bias_list = nn_split_params(params, n_nodes)
    for i, w_arr in enumerate(w_arr_list):
        inputs = prop_layer(inputs, w_arr, bias_list[i])
        if i == len(w_arr_list) - 1:
            inputs = out_act_func(inputs)
        else:
            inputs = act_func(inputs)
    assert inputs.shape == (1, 1)
    return inputs[0, 0]


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


def get_nn_param_latex_names(n_nodes):
    """get names for the neural network parameters."""
    assert isinstance(n_nodes, list), 'n_nodes={} is not list'.format(n_nodes)
    param_names = ['a_{}'.format(i) for i in range(n_nodes[-1] + 1)]
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
