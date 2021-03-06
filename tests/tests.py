#!/usr/bin/env python
"""
Test suite for the bsr package.
"""
import unittest
import numpy as np
import numpy.testing
import scipy.special
import dyPolyChord.python_priors
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
                               2.9730199861219813)

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
        # check error handling if theta[0] is nan
        theta[0] = np.nan
        out = bf.sum_basis_funcs(
            basis_func, theta, nfunc, x1, x2=x2, adaptive=True)
        self.assertTrue(np.isnan(out))
        # return to original random state
        np.random.set_state(state)

    def test_adfam_gg_ta_1d(self):
        """Check 1d adaptive family selection is working ok."""
        nfunc = 3
        x1 = np.random.random()
        theta = np.random.random(2 + (nfunc * 4))
        theta[1] = 2.3  # 2 funcs should be used
        # check T=1
        theta[0] = 1.1
        self.assertAlmostEqual(
            bf.adfam_gg_ta_1d(x1, theta, nfunc, adaptive=True),
            bf.sum_basis_funcs(bf.gg_1d, theta[1:], nfunc, x1, adaptive=True))
        # check T=2
        theta[0] = 2.1
        self.assertAlmostEqual(
            bf.adfam_gg_ta_1d(x1, theta, nfunc, adaptive=True),
            bf.sum_basis_funcs(bf.ta_1d, theta[1:-nfunc], nfunc, x1,
                               adaptive=True))
        # check error handling when theta[0] is nan
        theta[0] = np.nan
        self.assertTrue(
            np.isnan(bf.adfam_gg_ta_1d(x1, theta, nfunc, adaptive=True)))


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
        self.assertAlmostEqual(0.6771527317141491, out_nn)
        # check with float arg
        self.assertAlmostEqual(out_nn, nn.nn_fit(x[0], theta, n_nodes))
        # check with array input
        self.assertAlmostEqual(
            out_nn,
            nn.nn_fit(np.append(x, -10).reshape((1, 2)), theta, n_nodes)[0])
        # check adaptive - prepend n_nodes so all params used
        theta_ad = np.insert(theta, 0, n_nodes[-1])
        self.assertAlmostEqual(
            out_nn, nn.nn_fit(x, theta_ad, n_nodes, adaptive=True))
        # Check vs tanh basis function sum (should be equivalent)
        theta_bf = np.zeros(theta.shape)
        theta_bf[0] = theta[-1]  # global bias at front for bf but back for nn
        theta_bf[1:] = theta[:-1] # other params
        out_bf = bf.sum_basis_funcs(
            bf.ta_1d, theta_bf, n_nodes[1], x, x2=None, global_bias=True,
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
        self.assertAlmostEqual(0.703846220499679, out_nn)
        # Check vs tanh basis function sum (should be equivalent)
        theta_bf = np.zeros(theta.shape)
        theta_bf[0] = theta[-1]  # global bias at front for bf but back for nn
        theta_bf[1:] = theta[:-1] # other params
        out_bf = bf.sum_basis_funcs(
            bf.ta_2d, theta_bf, n_nodes[1], x[0], x2=x[1], global_bias=True,
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
        self.assertAlmostEqual(0.9149156893056427, out_nn)
        # 4d input and 4 hidden layers
        # ----------------------------
        np.random.seed(44)
        n_nodes = [4, 3, 4, 5, 2]
        x = np.random.random(n_nodes[0])
        n_params = nn.nn_num_params(n_nodes)
        self.assertEqual(n_params, 71)
        theta = np.random.random(n_params)
        out_nn = nn.nn_fit(x, theta, n_nodes)
        self.assertAlmostEqual(0.821340316965994, out_nn)
        np.random.set_state(state)  # return to original random state

    def test_nn_fit_wrappers(self):
        """Check wrapper funcs."""
        # nn_1l
        # -----
        state = np.random.get_state()  # save initial random state
        np.random.seed(0)
        n_nodes = [2, 5]
        x = np.random.random(n_nodes[0])
        theta = np.random.random(nn.nn_num_params(n_nodes))
        self.assertEqual(
            nn.nn_fit(x, theta, n_nodes),
            nn.nn_1l(x, theta, n_nodes))
        # nn_2l
        # -----
        n_nodes = [2, 4, 4]
        x = np.random.random(n_nodes[0])
        theta = np.random.random(nn.nn_num_params(n_nodes))
        self.assertEqual(
            nn.nn_fit(x, theta, n_nodes),
            nn.nn_2l(x, theta, n_nodes))
        # nn_adl
        # ------
        n_nodes = [2, 6, 6]
        x = np.random.random(n_nodes[0])
        theta = np.random.random(nn.nn_num_params(n_nodes) + 2)
        theta[1] = 3  # nfunc adaptive
        # n_layer = 1
        theta[0] = 1
        n_nodes_1l = [n_nodes[0], n_nodes[-1]]
        self.assertEqual(
            nn.nn_adl(x, theta, n_nodes, adaptive=True),
            nn.nn_1l(x, theta[1:2 + nn.nn_num_params(n_nodes_1l)],
                     n_nodes_1l, adaptive=True))
        np.random.set_state(state)  # return to original random state
        # n_layer = 2
        theta[0] = 2
        self.assertEqual(
            nn.nn_adl(x, theta, n_nodes, adaptive=True),
            nn.nn_2l(x, theta[1:], n_nodes, adaptive=True))
        # First param is nan
        theta[0] = np.nan
        self.assertTrue(np.isnan(nn.nn_adl(
            x, theta, n_nodes, adaptive=True)))
        self.assertTrue(np.isnan(nn.nn_adl(
            x[0], theta, n_nodes, adaptive=True)))

    def test_param_names(self):
        """Check parameter naming functions."""
        n_nodes = [2, 3]
        n_params = nn.nn_num_params(n_nodes)
        self.assertEqual(
            len(nn.get_nn_param_names(n_nodes, use_hyper=True)),
            n_params + 1)
        self.assertEqual(len(nn.get_nn_param_latex_names(
            n_nodes, use_hyper=True)), n_params + 1)

    def test_nn_flatten_parameters(self):
        """Check parameter naming functions."""
        n_nodes = [2, 4]
        n_params = nn.nn_num_params(n_nodes)
        theta = np.random.random(n_params)
        w_arr_list, bias_list = nn.nn_split_params(theta, n_nodes)
        self.assertEqual(len(w_arr_list), len(bias_list))
        theta_flat = nn.nn_flatten_params(w_arr_list, bias_list)
        numpy.testing.assert_array_equal(theta, theta_flat)

    def test_adaptive_theta(self):
        """Chech the adaptive_theta function for zeroing out nodes gives the
        same answer as a vanilla neural network."""
        # settings
        x = np.random.random(2)
        n_nodes_a = [2, 5, 5]
        nfunc_max = 4
        value = np.random.random()  # give all theta components the same value
        # get adaptive
        theta_a = np.full(nn.nn_num_params(n_nodes_a) + 1, value)
        theta_a[0] = nfunc_max
        theta_a = nn.adaptive_theta(theta_a, n_nodes_a)
        out_a = nn.nn_fit(x, theta_a, n_nodes_a)
        # get vanilla
        n_nodes_v = [2] + ([nfunc_max] * len(n_nodes_a[1:]))
        theta_v = np.full(nn.nn_num_params(n_nodes_v), value)
        out_v = nn.nn_fit(x, theta_v, n_nodes_v)
        print(out_v, out_a)
        self.assertAlmostEqual(out_a, out_v)
        # Nan handling
        theta_a = np.full(nn.nn_num_params(n_nodes_a) + 1, value)
        theta_a[0] = np.nan
        self.assertTrue(np.all(np.isnan(
            nn.adaptive_theta(theta_a, n_nodes_a))))



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
        # check if x_error_sigma = 0 it is set to None
        datax0 = bsr.data.generate_data(data_func, data_type, y_error_sigma,
                                        x_error_sigma=0)
        assert datax0['x_error_sigma'] is None
        # try with image
        y_error_sigma = 0.05
        x_error_sigma = None
        data_func = bsr.data.get_image
        data_type = 2
        data = bsr.data.generate_data(data_func, data_type, y_error_sigma,
                                      x_error_sigma=x_error_sigma,
                                      file_dir='tests/')
        self.assertEqual(set(data.keys()), set(keys))

    def test_get_data_args(self):
        """Check right number of args is supplied."""
        for data_func in [bf.gg_1d, bf.gg_2d, bf.ta_1d]:
            for nfunc in [1, 2, 3]:
                args = bsr.data.get_data_args(data_func, nfunc)
                self.assertEqual(len(args),
                                 nfunc * len(bf.get_bf_param_names(data_func)))
        self.assertRaises(AssertionError, bsr.data.get_data_args,
                          bf.gg_1d, 100)


class TestPriors(unittest.TestCase):

    """Tests for the priors.py module. Some of these tests also check the
    dyPolyChord python priors - this dosn't take long and protects against
    unexpected changes to dyPolyChord."""

    @staticmethod
    def test_uniform():
        """Check uniform prior."""
        prior_scale = 5
        hypercube = np.random.random(5)
        theta_prior = dyPolyChord.python_priors.Uniform(
            -prior_scale, prior_scale)(hypercube)
        theta_check = (hypercube * 2 * prior_scale) - prior_scale
        numpy.testing.assert_allclose(theta_prior, theta_check)

    @staticmethod
    def test_power_uniform():
        """Check prior for some power of theta uniformly distributed"""
        cube = np.random.random(10)
        for power in [-2, 3]:
            power = -2
            maximum = 20
            minimum = 0.1
            theta = dyPolyChord.python_priors.PowerUniform(
                minimum, maximum, power=power)(cube)
            # Check this vs doing a uniform prior and transforming
            # Note if power < 0, the high to low order of X is inverted
            umin = min(minimum ** (1 / power), maximum ** (1 / power))
            umax = max(minimum ** (1 / power), maximum ** (1 / power))
            test_prior = dyPolyChord.python_priors.Uniform(umin, umax)
            if power < 0:
                theta_check = test_prior(1 - cube) ** power
            else:
                theta_check = test_prior(cube) ** power
            numpy.testing.assert_allclose(theta, theta_check)


    @staticmethod
    def test_gaussian():
        """Check spherically symmetric Gaussian prior centred on the origin."""
        prior_scale = 5
        hypercube = np.random.random(5)
        theta_prior = dyPolyChord.python_priors.Gaussian(
            prior_scale)(hypercube)
        theta_check = (scipy.special.erfinv(hypercube * 2 - 1) *
                       prior_scale * np.sqrt(2))
        numpy.testing.assert_allclose(theta_prior, theta_check)

    @staticmethod
    def test_exponential():
        """Check the exponential prior."""
        prior_scale = 5
        hypercube = np.random.random(5)
        theta_prior = dyPolyChord.python_priors.Exponential(
            prior_scale)(hypercube)
        theta_check = -np.log(1 - hypercube) / prior_scale
        numpy.testing.assert_allclose(theta_prior, theta_check)

    @staticmethod
    def test_forced_identifiability():
        """Check the forced identifiability (forced ordering) transform.
        Note that the PolyChord paper contains a typo in the formulae."""
        n = 5
        hypercube = np.random.random(n)
        theta_func = dyPolyChord.python_priors.forced_identifiability(
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
    def test_default_gg_2d_prior():
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
            1.59762701, 0.81025994, 1.49779981, 0.54488318, 0.4236548,
            0.64589411, 0.43758721, 0.44913331, 0.4829215, 0.21021751,
            0.40211077, 1.50534822, 1.67886569, 5.19650831, 0.14737071,
            -0.64853578, -0.75363918])
        numpy.testing.assert_allclose(prior(cube), expected,
                                      rtol=1e-06, atol=1e-06)
        # return to original random state
        np.random.set_state(state)

    @staticmethod
    def test_default_gg_1d_prior():
        """Check the numerical values from the default prior."""
        # save initial random state
        state = np.random.get_state()
        np.random.seed(0)
        # Test gg_2d prior
        nfunc = 2
        func = bf.gg_1d
        prior = bsr.priors.get_default_prior(func, nfunc, adaptive=True)
        cube = np.random.random(sum(prior.block_sizes))
        expected = np.array([
            1.59762701, 0.81025994, 1.49779981, 0.54488318, 0.4236548,
            0.65651729, 0.45445959, 4.44704883, 6.62982436])
        numpy.testing.assert_allclose(prior(cube), expected,
                                      rtol=1e-06, atol=1e-06)
        # return to original random state
        np.random.set_state(state)

    @staticmethod
    def test_default_ta_1d_prior():
        """Check the numerical values from the default prior."""
        # save initial random state
        state = np.random.get_state()
        np.random.seed(0)
        nfunc = 2
        func = bf.ta_1d
        prior = bsr.priors.get_default_prior(func, nfunc, adaptive=True)
        cube = np.random.random(sum(prior.block_sizes))
        expected = np.array([
            1.59762701, 3.82104718, 6.08477118, 0.56371926, -0.96276154,
            1.87129405, -0.7854468])
        numpy.testing.assert_allclose(prior(cube), expected,
                                      rtol=1e-06, atol=1e-06)
        # return to original random state
        np.random.set_state(state)

    def test_default_adfam_gg_ta_1d_prior(self):
        """Check the numerical values from the default prior."""
        # save initial random state
        state = np.random.get_state()
        np.random.seed(0)
        # Test gg_2d prior
        nfunc = 2
        func = bf.adfam_gg_ta_1d
        prior = bsr.priors.get_default_prior(func, nfunc, adaptive=True)
        cube = np.random.random(sum(
            prior.long_prior.block_sizes) + 1)
        # test T=1
        cube[0] = 0.25
        expected = np.array([
            1.00000000, 1.93037873, 0.58867373, 1.3400294, 0.4236548,
            0.64589411, 0.45445959, 0.89501981, 6.62982436, 0.9672042])
        numpy.testing.assert_allclose(prior(cube), expected,
                                      rtol=1e-06, atol=1e-06)
        # test T=2
        cube[0] = 0.75
        expected = np.array([
            2.00000000, 1.93037873, 2.95095234, 5.61028777, -0.96276154,
            1.87129405, -0.7854468, 6.18006141, 6.62982436, 0.9672042])
        numpy.testing.assert_allclose(prior(cube), expected,
                                      rtol=1e-06, atol=1e-06)
        # test error handling of nans
        cube[0] = np.nan
        self.assertTrue(np.all(np.isnan(prior(cube))))
        # return to original random state
        np.random.set_state(state)


    @staticmethod
    def test_default_nn_adl_prior():
        """Check the numerical values from the default prior."""
        # save initial random state
        state = np.random.get_state()
        np.random.seed(0)
        # Test gg_2d prior
        nfunc = [2, 4, 4]
        func = nn.nn_adl
        prior = bsr.priors.get_default_prior(func, nfunc, adaptive=True)
        cube = np.random.random(sum(
            prior.long_prior.block_sizes) + 1)
        # test T=1
        # --------
        cube[0] = 0.25
        expected = np.array([
            1.00000000, 3.36075747, 0.43188003, 0.76276923, 1.15291132,
            0.92665462, -0.15708936, 1.23601228, 1.7948696, -0.29645446,
            0.81242131, 0.07249227, 0.17139794, 1.44375920, -1.46811821,
            -1.35864661, -2.04925901, 0.96456985, 0.76598286, 2.72665814,
            2.02601434, 0.83861949, -0.09670745, 0.77398164, -1.18365704,
            0.35824769, -1.06537432, 1.59522395, 0.05479303, -0.2155688,
            -0.62936332, 0.75286237, -0.11013708, 0.17238851, -2.07941089,
            0.2992766, 0.28478540, 0.29743817, 1.58703886, 0.74491857])
        numpy.testing.assert_allclose(prior(cube), expected,
                                      rtol=1e-06, atol=1e-06)
        # Check is the same as nn_1l prior
        nfunc_1l = [nfunc[0], nfunc[-1]]
        nparam_1l = nn.nn_num_params(nfunc_1l) + 1
        prior_1l = bsr.priors.get_default_prior(
            nn.nn_1l, nfunc_1l, adaptive=True)
        numpy.testing.assert_allclose(
            prior(cube)[1:1 + nparam_1l], prior_1l(cube[1:1 + nparam_1l]),
            rtol=1e-06, atol=1e-06)
        # test T=2
        # --------
        cube[0] = 0.75
        expected = np.array([
            2.00000000, 3.36075747, -0.42842354, 0.13678332, 0.6778078,
            0.37425881, -0.15708936, 1.23601228, 1.7948696, -0.29645446,
            0.81242131, 0.07249227, 0.17139794, 1.4437592, -1.46811821,
            -1.35864661, -2.04925901, 0.96456985, 0.76598286, 1.12644856,
            2.02601434, 0.83861949, -0.09670745, 0.77398164, -1.18365704,
            0.35824769, -1.06537432, 1.59522395, 0.05479303, -0.21556880,
            -0.62936332, 0.75286237, -0.11013708, 0.17238851, -2.07941089,
            0.29927660, 0.28478540, 0.29743817, 1.58703886, 0.74491857])
        numpy.testing.assert_allclose(prior(cube), expected,
                                      rtol=1e-06, atol=1e-06)
        # Check is the same as nn_2l prior
        prior_2l = bsr.priors.get_default_prior(
            nn.nn_2l, nfunc, adaptive=True)
        numpy.testing.assert_allclose(
            prior(cube)[1:], prior_2l(cube[1:]), rtol=1e-06, atol=1e-06)
        # return to original random state
        np.random.set_state(state)


    @staticmethod
    def test_nn_prior():
        """Check default neural network prior."""
        # Test nn prior
        n_nodes = [2, 3]
        sigma_w = 1
        state = np.random.get_state()
        np.random.seed(0)
        # Vanilla
        cube = np.random.random(nn.nn_num_params(n_nodes) + 1)
        prior = bsr.priors.get_default_prior(
            nn.nn_fit, n_nodes, adaptive=False)
        expected = np.zeros(cube.shape)
        expected[:n_nodes[-1]] = dyPolyChord.python_priors.Gaussian(
            sigma_w, sort=True, half=True)(cube[:n_nodes[-1]])
        expected[n_nodes[-1]:-1] = dyPolyChord.python_priors.Gaussian(
            sigma_w, sort=False)(cube[n_nodes[-1]:-1])
        expected[-1] = dyPolyChord.python_priors.PowerUniform(
            0.1, 20, power=-2)(cube[-1])
        numpy.testing.assert_allclose(
            prior(cube), expected, rtol=1e-06, atol=1e-06)
        # Adaptive
        cube = np.random.random(nn.nn_num_params(n_nodes) + 2)
        prior = bsr.priors.get_default_prior(
            nn.nn_fit, n_nodes, adaptive=True)
        expected = np.zeros(cube.shape)
        expected[:n_nodes[-1] + 1] = dyPolyChord.python_priors.Gaussian(
            sigma_w, sort=True, adaptive=True, half=True)(
                cube[:n_nodes[-1] + 1])
        expected[n_nodes[-1] + 1:] = dyPolyChord.python_priors.Gaussian(
            sigma_w)(cube[n_nodes[-1] + 1:])
        # Get sigma_w from prior and scale weights
        expected[-1] = dyPolyChord.python_priors.PowerUniform(
            0.1, 20, power=-2)(cube[-1])
        numpy.testing.assert_allclose(prior(cube), expected,
                                      rtol=1e-06, atol=1e-06)
        # Multiple hidden layers
        n_nodes = [2, 4, 4]
        cube = np.random.random(nn.nn_num_params(n_nodes) + 1)
        expected = np.zeros(cube.shape)
        prior = bsr.priors.get_default_prior(
            nn.nn_2l, n_nodes, adaptive=False)
        expected[:n_nodes[-1]] = dyPolyChord.python_priors.Gaussian(
            sigma_w, sort=True, half=False)(cube[:n_nodes[-1]])
        expected[n_nodes[-1]:-1] = dyPolyChord.python_priors.Gaussian(
            sigma_w, sort=False)(cube[n_nodes[-1]:-1])
        expected[-1] = dyPolyChord.python_priors.PowerUniform(
            0.1, 20, power=-2)(cube[-1])
        numpy.testing.assert_allclose(
            prior(cube), expected, rtol=1e-06, atol=1e-06)
        # return to original random state
        np.random.set_state(state)


class TestLikelihoods(unittest.TestCase):

    """Tests for the likelihoods.py module."""

    def test_gg_1d_fit(self):
        """Basic tests of fitting likelihood with and without X errors."""
        theta = np.asarray([0.1, 0.1, 0.1, 2])
        # Try with x errors
        data = bsr.data.generate_data(bf.gg_1d, 1, 0.05, x_error_sigma=0.05)
        likelihood = bsr.likelihoods.FittingLikelihood(
            data, bf.gg_1d, 1)
        logl = likelihood(theta)[0]
        self.assertAlmostEqual(logl, -3644.2492288935546)
        # Try without x errors
        data = bsr.data.generate_data(bf.gg_1d, 1, 0.1, x_error_sigma=None)
        likelihood = bsr.likelihoods.FittingLikelihood(
            data, bf.gg_1d, 1)
        logl = likelihood(theta)[0]
        self.assertAlmostEqual(logl, -996.0402722959848)


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
            data, nn.nn_fit, n_nodes, use_hyper=False)
        n_params = nn.nn_num_params(n_nodes)
        theta = np.random.random(n_params)
        theta[-1] = 0  # Correct for global bias (not present in ta)
        out_ta = bf.sigmoid_func(ta_likelihood.fit(theta[:-1], x))
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
            data, nn.nn_fit, n_nodes, use_hyper=False)
        n_params = nn.nn_num_params(n_nodes)
        theta = np.random.random(n_params)
        theta[-1] = 0  # Correct for global bias (not present in ta)
        out_ta = bf.sigmoid_func(ta_likelihood.fit(theta[:-1], x[0], x2=x[1]))
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
