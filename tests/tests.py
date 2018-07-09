#!/usr/bin/env python
"""
Test suite for the bsr package.
"""
# import os
# import shutil
# import warnings
import unittest
import numpy as np
import numpy.testing
import bsr.basis_functions as bf
import bsr.neural_networks as nn


class TestBasisFunctions(unittest.TestCase):

    """Tests for the basis_functions.py module."""

    def test_gg_1d(self):
        """Check gg_1d basis function values."""
        args = tuple(range(1, 6))
        self.assertAlmostEqual(bf.gg_1d(*args),
                               1.9384664689526883)

    def test_gg_2d(self):
        """Check gg_2d basis function values."""
        args = tuple(range(1, 11))
        self.assertAlmostEqual(bf.gg_2d(*args),
                               2.999954698792266)

    def test_ta_1d(self):
        """Check ta_1d basis function values."""
        args = tuple(range(1, 5))
        self.assertAlmostEqual(bf.ta_1d(*args),
                               1.9999966738878894)

    def test_ta_2d(self):
        """Check ta_2d basis function values."""
        args = tuple(range(-3, 3))
        self.assertAlmostEqual(bf.ta_2d(*args),
                               0.9999983369439447)

    def test_sigmoid_func(self):
        """Check sigmoid_func values."""
        self.assertAlmostEqual(bf.sigmoid_func(1.),
                               0.7310585786300049)

    def test_sum_basis_func(self):
        """Check summing basis funcs, including adaptive."""
        nfunc = 2
        basis_func = bf.gg_1d
        x1 = 0.5
        x2 = None
        # make params
        state = np.random.get_state()  # save initial random state
        np.random.seed(0)
        nargs = len(bf.get_bf_param_names(basis_func))
        theta = np.random.random((nargs * nfunc) + 1)
        # check adaptive using all funcs
        theta[0] = nfunc
        out = bf.sum_basis_funcs(
            basis_func, theta, nfunc, x1, x2=x2, adaptive=True)
        self.assertAlmostEqual(out, 1.152357037165534)
        # check adaptive using 1 func is consistent
        theta[0] = 1
        out = bf.sum_basis_funcs(
            basis_func, theta, nfunc, x1, x2=x2, adaptive=True)
        self.assertAlmostEqual(out, basis_func(x1, *theta[1::nfunc]))
        # return to original random state
        np.random.set_state(state)

    def test_names(self):
        """Check the parameter naming functions."""
        for func in [bf.gg_1d, bf.gg_2d, bf.ta_1d, bf.ta_2d]:
            names = bf.get_bf_param_names(func)
            self.assertEqual(len(names),
                             len(bf.get_param_latex_names(names)))


class TestNeuralNetworks(unittest.TestCase):

    """Tests for the neural_networks.py module."""

    def test_nn_fit(self):
        """Check the neural network fitting function's output values, including
        against sums of tanh basis functions."""
        state = np.random.get_state()  # save initial random state
        # 1d input and 1 hidden layer
        # ---------------------------
        np.random.seed(1)
        nodes = [3]
        x = np.random.random()
        n_params = nn.nn_num_params([1] + nodes)
        self.assertEqual(n_params, 10)
        theta = np.random.random(n_params)
        out_nn = nn.nn_fit(x, theta, nodes)
        self.assertAlmostEqual(0.712630649588386, out_nn)
        # Check vs tanh basis function sum (should be equivalent)
        out_bf = bf.sum_basis_funcs(
            bf.ta_1d, theta, nodes[0], x, x2=None, global_bias=True,
            sigmoid=True, adaptive=False)
        self.assertAlmostEqual(out_bf, out_nn)
        # 2d input and 1 hidden layer
        # ---------------------------
        np.random.seed(2)
        nodes = [3]
        x = np.random.random(2)
        n_params = nn.nn_num_params([2] + nodes)
        self.assertEqual(n_params, 13)
        theta = np.random.random(n_params)
        out_nn = nn.nn_fit(x, theta, nodes)
        self.assertAlmostEqual(0.7594362318597511, out_nn)
        # Check vs tanh basis function sum (should be equivalent)
        out_bf = bf.sum_basis_funcs(
            bf.ta_2d, theta, nodes[0], x[0], x2=x[1], global_bias=True,
            sigmoid=True, adaptive=False)
        self.assertAlmostEqual(out_bf, out_nn)
        # 2d input and 2 hidden layers
        # ----------------------------
        np.random.seed(22)
        nodes = [3, 4]
        x = np.random.random(2)
        n_params = nn.nn_num_params([2] + nodes)
        self.assertEqual(n_params, 30)
        theta = np.random.random(n_params)
        out_nn = nn.nn_fit(x, theta, nodes)
        self.assertAlmostEqual(0.8798621458937803, out_nn)
        # 4d input and 4 hidden layers
        # ----------------------------
        np.random.seed(44)
        nodes = [3, 4, 5, 2]
        x = np.random.random(4)
        n_params = nn.nn_num_params([4] + nodes)
        self.assertEqual(n_params, 71)
        theta = np.random.random(n_params)
        out_nn = nn.nn_fit(x, theta, nodes)
        self.assertAlmostEqual(0.7952126284097402, out_nn)
        np.random.set_state(state)  # return to original random state

    def test_param_names(self):
        """Check parameter naming functions."""
        n_nodes = [2, 3]
        n_params = nn.nn_num_params(n_nodes)
        self.assertEqual(len(nn.get_nn_param_names(n_nodes)), n_params)
        self.assertEqual(len(nn.get_nn_param_latex_names(n_nodes)), n_params)

    def test_nn_flatten_parameters(self):
        """Check parameter naming functions."""
        n_nodes = [2, 4]
        n_params = nn.nn_num_params(n_nodes)
        theta = np.random.random(n_params)
        w_arr_list, bias_list = nn.nn_split_params(theta, n_nodes)
        self.assertEqual(len(w_arr_list), len(bias_list))
        theta_flat = nn.nn_flatten_params(w_arr_list, bias_list)
        numpy.testing.assert_array_equal(theta, theta_flat)


if __name__ == '__main__':
    unittest.main()
