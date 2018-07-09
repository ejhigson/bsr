#!/usr/bin/env python
"""
Test suite for the bsr package.
"""
import unittest
import numpy as np
import numpy.testing
import scipy.special
import bsr.basis_functions as bf
import bsr.neural_networks as nn
import bsr.priors
import bsr.data
import bsr.likelihoods


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
        n_nodes = [1, 3]
        x = np.random.random(n_nodes[0])
        n_params = nn.nn_num_params(n_nodes)
        self.assertEqual(n_params, 10)
        theta = np.random.random(n_params)
        out_nn = nn.nn_fit(x, theta, n_nodes)
        self.assertAlmostEqual(0.712630649588386, out_nn)
        # Check vs tanh basis function sum (should be equivalent)
        out_bf = bf.sum_basis_funcs(
            bf.ta_1d, theta, n_nodes[1], x, x2=None, global_bias=True,
            sigmoid=True, adaptive=False)
        self.assertAlmostEqual(out_bf, out_nn)
        # 2d input and 1 hidden layer
        # ---------------------------
        np.random.seed(2)
        n_nodes = [2, 3]
        x = np.random.random(n_nodes[0])
        n_params = nn.nn_num_params(n_nodes)
        self.assertEqual(n_params, 13)
        theta = np.random.random(n_params)
        out_nn = nn.nn_fit(x, theta, n_nodes)
        self.assertAlmostEqual(0.7594362318597511, out_nn)
        # Check vs tanh basis function sum (should be equivalent)
        out_bf = bf.sum_basis_funcs(
            bf.ta_2d, theta, n_nodes[1], x[0], x2=x[1], global_bias=True,
            sigmoid=True, adaptive=False)
        self.assertAlmostEqual(out_bf, out_nn)
        # 2d input and 2 hidden layers
        # ----------------------------
        np.random.seed(22)
        n_nodes = [2, 3, 4]
        x = np.random.random(n_nodes[0])
        n_params = nn.nn_num_params(n_nodes)
        self.assertEqual(n_params, 30)
        theta = np.random.random(n_params)
        out_nn = nn.nn_fit(x, theta, n_nodes)
        self.assertAlmostEqual(0.8798621458937803, out_nn)
        # 4d input and 4 hidden layers
        # ----------------------------
        np.random.seed(44)
        n_nodes = [4, 3, 4, 5, 2]
        x = np.random.random(n_nodes[0])
        n_params = nn.nn_num_params(n_nodes)
        self.assertEqual(n_params, 71)
        theta = np.random.random(n_params)
        out_nn = nn.nn_fit(x, theta, n_nodes)
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


class TestData(unittest.TestCase):

    """Tests for the data.py module."""

    def test_generate_data(self):
        """Check data generated is consistent."""
        data_func = bf.gg_1d
        data_type = 1
        y_error_sigma = 0.05
        x_error_sigma = 0.05
        keys = ['x1',
                'x1_no_noise',
                'x1min',
                'x1max',
                'x2',
                'x2min',
                'x2max',
                'y',
                'y_no_noise',
                'data_name',
                'func_name',
                'x_error_sigma',
                'data_type',
                'func',
                'x1_no_noise',
                'args',
                'random_seed',
                'y_error_sigma']
        data = bsr.data.generate_data(data_func, data_type, y_error_sigma,
                                      x_error_sigma=x_error_sigma)
        self.assertEqual(set(data.keys()), set(keys))
        # try with image
        y_error_sigma = 0.05
        x_error_sigma = None
        data_func = bsr.data.get_image
        data_type = 2
        data = bsr.data.generate_data(data_func, data_type, y_error_sigma,
                                      x_error_sigma=x_error_sigma,
                                      file_dir='tests/')
        self.assertEqual(set(data.keys()), set(keys))


class TestPriors(unittest.TestCase):

    """Tests for the priors.py module."""

    @staticmethod
    def test_uniform():
        """Check uniform prior."""
        prior_scale = 5
        hypercube = np.random.random(5)
        theta_prior = bsr.priors.Uniform(-prior_scale, prior_scale)(hypercube)
        theta_check = (hypercube * 2 * prior_scale) - prior_scale
        numpy.testing.assert_allclose(theta_prior, theta_check)

    @staticmethod
    def test_gaussian():
        """Check spherically symmetric Gaussian prior centred on the origin."""
        prior_scale = 5
        hypercube = np.random.random(5)
        theta_prior = bsr.priors.Gaussian(prior_scale)(hypercube)
        theta_check = (scipy.special.erfinv(hypercube * 2 - 1) *
                       prior_scale * np.sqrt(2))
        numpy.testing.assert_allclose(theta_prior, theta_check)

    @staticmethod
    def test_exponential():
        """Check the exponential prior."""
        prior_scale = 5
        hypercube = np.random.random(5)
        theta_prior = bsr.priors.Exponential(prior_scale)(hypercube)
        theta_check = -np.log(1 - hypercube) / prior_scale
        numpy.testing.assert_allclose(theta_prior, theta_check)

    @staticmethod
    def test_forced_identifiability():
        """Check the forced identifiability (forced ordering) transform.
        Note that the PolyChord paper contains a typo in the formulae."""
        n = 5
        hypercube = np.random.random(n)
        theta_func = bsr.priors.forced_identifiability_transform(
            hypercube)

        def forced_ident_transform(x):
            """PyPolyChord version of the forced identifiability transform.
            Note that the PolyChord paper contains a typo, but this equation
            is correct."""
            n = len(x)
            theta = numpy.zeros(n)
            theta[n - 1] = x[n - 1] ** (1. / n)
            for i in range(n - 2, -1, -1):
                theta[i] = x[i] ** (1. / (i + 1)) * theta[i + 1]
            return theta

        theta_check = forced_ident_transform(hypercube)
        numpy.testing.assert_allclose(theta_func, theta_check)

    @staticmethod
    def test_default_priors():
        """Check the numerical values from the default prior."""
        # save initial random state
        state = np.random.get_state()
        np.random.seed(0)
        # Test gg_2d prior
        nfunc = 2
        func = bf.gg_2d
        prior = bsr.priors.get_default_prior(func, nfunc, adaptive=True)
        cube = np.random.random(sum(prior.block_sizes))
        expected = np.array([
            1.597627, 0.40513, 0.7489, 0.544883, 0.423655, 0.645894,
            0.437587, 1.111762, 1.657456, 0.241801, 0.784448, 1.505348,
            1.678866, 5.196508, 0.147371, -0.648536, -0.753639])
        numpy.testing.assert_allclose(prior(cube), expected,
                                      rtol=1e-06, atol=1e-06)
        # Test nn prior
        n_nodes = [2, 3]
        np.random.seed(0)
        cube = np.random.random(nn.nn_num_params(n_nodes))
        prior = bsr.priors.get_default_prior(
            nn.nn_fit, n_nodes, adaptive=False)
        expected = bsr.priors.Gaussian(10)(cube)
        numpy.testing.assert_allclose(prior(cube), expected,
                                      rtol=1e-06, atol=1e-06)
        # return to original random state
        np.random.set_state(state)


class TestLikelihoods(unittest.TestCase):

    """Tests for the likelihoods.py module."""

    def test_nn_ta_fit(self):
        """Check the neural network fitting function's output values, including
        against sums of tanh basis functions."""
        state = np.random.get_state()  # save initial random state
        # 1d input and 1 hidden layer
        # ---------------------------
        np.random.seed(1)
        n_nodes = [1, 3]
        x = np.random.random(n_nodes[0])
        data = bsr.data.generate_data(bf.gg_1d, 1, 0.05, x_error_sigma=0.05)
        ta_likelihood = bsr.likelihoods.FittingLikelihood(
            data, bf.ta_1d, n_nodes[1])
        nn_likelihood = bsr.likelihoods.FittingLikelihood(
            data, nn.nn_fit, n_nodes)
        n_params = nn.nn_num_params(n_nodes)
        theta = np.random.random(n_params)
        theta[0] = 0  # Correct for global bias (not present in ta)
        out_ta = bf.sigmoid_func(ta_likelihood.fit(theta[1:], x))
        out_nn = nn_likelihood.fit(theta, x)
        self.assertAlmostEqual(out_ta, out_nn)
        # check names
        names_ta = ta_likelihood.get_param_names()
        names_nn = nn_likelihood.get_param_names()
        for i, name_nn in enumerate(names_nn[1:]):
            if name_nn[0] == 'a':
                assert name_nn == names_ta[i]
            else:
                assert name_nn == names_ta[i] + '_0'
        # 2d input and 1 hidden layer
        # ---------------------------
        n_nodes = [2, 3]
        x = np.random.random(n_nodes[0])
        data = bsr.data.generate_data(bf.gg_2d, 1, 0.1)
        ta_likelihood = bsr.likelihoods.FittingLikelihood(
            data, bf.ta_2d, n_nodes[1])
        nn_likelihood = bsr.likelihoods.FittingLikelihood(
            data, nn.nn_fit, n_nodes)
        n_params = nn.nn_num_params(n_nodes)
        theta = np.random.random(n_params)
        theta[0] = 0  # Correct for global bias (not present in ta)
        out_ta = bf.sigmoid_func(ta_likelihood.fit(theta[1:], x[0], x2=x[1]))
        print(x.shape, np.atleast_2d(x).T.shape)
        out_nn = nn_likelihood.fit(theta, x[0], x2=x[1])
        self.assertAlmostEqual(out_ta, out_nn)
        # check names
        names_ta = ta_likelihood.get_param_names()
        names_nn = nn_likelihood.get_param_names()
        for i, name_nn in enumerate(names_nn[1:]):
            if name_nn[0] == 'a':
                assert name_nn == names_ta[i]
            else:
                assert name_nn == names_ta[i] + '_0'
        # return to original random state
        np.random.set_state(state)


if __name__ == '__main__':
    unittest.main()
