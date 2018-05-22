#!/usr/bin/env python
"""
Python priors for use with PolyChord.

PolyChord v1.14 requires priors to be callables with parameter and return
types:

Parameters
----------
hypercube: float or 1d numpy array
    Parameter positions in the prior hypercube.

Returns
-------
theta: float or 1d numpy array
    Corresponding physical parameter coordinates.

Input hypercube values numpy array is mapped to physical space using the
inverse CDF (cumulative distribution function) of each parameter.
See the PolyChord papers for more details.

We use classes with the prior defined in the __call__ property, as
this provides convenient way of storing other information such as
hyperparameter values. The objects be used in the same way as functions
due to python's "duck typing" (or alternatively you can just define prior
functions).
"""
import numpy as np
import scipy
import bsr.basis_functions


def get_default_prior(func, nfunc, adaptive=False, **kwargs):
    """Construct a default set of priors for the basis function."""
    assert func.__name__ in ['gg_1d', 'gg_2d', 'nn_1d', 'nn_2d'], (
        'not yet set up for {}'.format(func.__name__))
    nfunc_min = kwargs.pop('nfunc_min', 1)
    global_bias = kwargs.pop(  # default True if nn, False otherwise
        'global_bias', func.__name__[:2] == 'nn')
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    # specify default priors
    if func.__name__ in ['gg_1d', 'gg_2d']:
        priors_dict = {'a':     SortedUniform(0, 1.0),
                       'mu':    Uniform(0, 1.0),
                       'sigma': Uniform(0.01, 1.0),
                       'beta':  Uniform(0.1, 10.0)}
        if adaptive:
            priors_dict['a'] = AdaptiveSortedUniform(
                0, 1.0, nfunc_min=nfunc_min)
        assert not global_bias
        if func.__name__ == 'gg_2d':
            for param in ['mu', 'sigma', 'beta']:
                priors_dict[param + '1'] = priors_dict[param]
                priors_dict[param + '2'] = priors_dict[param]
            priors_dict['omega'] = Uniform(-0.25 * np.pi, 0.25 * np.pi)
    elif func.__name__ in ['nn_1d', 'nn_2d']:
        priors_dict = {'a':           SortedUniform(0, 10),
                       'w_0':         Uniform(-10, 10),
                       'w_1':         Uniform(-10, 10),
                       'w_2':         Uniform(-10, 10),
                       'global_bias': Uniform(-10, 10)}
        if adaptive:
            priors_dict['a'] = AdaptiveSortedUniform(
                0, 250, nfunc_min=nfunc_min)
    # Get a list of the priors we want
    args = bsr.basis_functions.get_param_names(func)
    prior_blocks = [priors_dict[arg] for arg in args]
    block_sizes = [nfunc] * len(args)
    if adaptive:
        block_sizes[0] += 1
    if global_bias:
        prior_blocks.append(priors_dict['global_bias'])
        block_sizes.append(1)
    return BlockPrior(prior_blocks, block_sizes)


class Gaussian(object):

    """Symmetric Gaussian prior centred on the origin."""

    def __init__(self, sigma=10.0):
        """
        Set up prior object's hyperparameter values.

        Parameters
        ----------
        sigma: float
            Standard deviation of Gaussian prior in each parameter.
        """
        self.sigma = sigma

    def __call__(self, hypercube):
        """
        Map hypercube values to physical parameter values.

        Parameters
        ----------
        hypercube: list of floats
            Point coordinate on unit hypercube (in probabily space).
            See the PolyChord papers for more details.

        Returns
        -------
        theta: list of floats
            Physical parameter values corresponding to hypercube.
        """
        theta = scipy.special.erfinv(2 * hypercube - 1)
        theta *= self.sigma * np.sqrt(2)
        return theta


class Uniform(object):

    """Uniform prior."""

    def __init__(self, minimum=0.0, maximum=1.0):
        """
        Set up prior object's hyperparameter values.

        Prior is uniform in [minimum, maximum] in each parameter.

        Parameters
        ----------
        minimum: float
        maximum: float
        """
        assert maximum > minimum
        self.maximum = maximum
        self.minimum = minimum

    def __call__(self, hypercube):
        """
        Map hypercube values to physical parameter values.

        Parameters
        ----------
        hypercube: list of floats
            Point coordinate on unit hypercube (in probabily space).
            See the PolyChord papers for more details.

        Returns
        -------
        theta: list of floats
            Physical parameter values corresponding to hypercube.
        """
        return self.minimum + (self.maximum - self.minimum) * hypercube


class SortedUniform(Uniform):

    """Uniform prior with sorting imposed so values have decreasing size."""

    def __call__(self, cube):
        """
        Map hypercube values to physical parameter values.

        Parameters
        ----------
        hypercube: list of floats
            Point coordinate on unit hypercube (in probabily space).
            See the PolyChord papers for more details.

        Returns
        -------
        theta: list of floats
            Physical parameter values corresponding to hypercube.
        """
        theta = np.zeros(cube.shape)
        theta[-1] = cube[-1] ** (1. / cube.shape[0])
        for n in range(cube.shape[0] - 2, -1, -1):
            theta[n] = cube[n] ** (1. / (n + 1)) * theta[n + 1]
        return Uniform.__call__(self, theta)


class AdaptiveSortedUniform(SortedUniform):

    """Adaptive sorted uniform prior."""

    def __init__(self, minimum, maximum, nfunc_min=1):
        """
        Set up prior object's hyperparameter values.

        Parameters
        ----------
        minimum: float
        maximum: float
        nfunc_min: int, optional
        """
        SortedUniform.__init__(self, minimum, maximum)
        self.minimum = minimum
        self.maximum = maximum
        self.nfunc_min = nfunc_min

    def __call__(self, cube):
        """
        Map hypercube values to physical parameter values.

        Parameters
        ----------
        hypercube: list of floats
            Point coordinate on unit hypercube (in probabily space).
            See the PolyChord papers for more details.

        Returns
        -------
        theta: list of floats
            Physical parameter values corresponding to hypercube.
        """
        # First get integer number of funcs
        theta = np.zeros(cube.shape)
        nfunc_max = cube.shape[0] - 1
        # first component is a number of funcs
        theta[0] = ((self.nfunc_min - 0.5)
                    + (1.0 + nfunc_max - self.nfunc_min) * cube[0])
        nfunc = int(np.round(theta[0]))
        # perform SortedUniform on the next nfunc components
        theta[1:1 + nfunc] = SortedUniform.__call__(self, cube[1:1 + nfunc])
        # do uniform prior on remaining components
        if len(cube) > 1 + nfunc:
            theta[1 + nfunc:] = (
                self.minimum + (self.maximum - self.minimum)
                * cube[1 + nfunc:])
        return theta


class BlockPrior(object):

    """Prior object which applies a list of priors to different blocks within
    the parameters."""

    def __init__(self, prior_blocks, block_sizes):
        """Store prior and size of each block."""
        assert len(prior_blocks) == len(block_sizes)
        self.prior_blocks = prior_blocks
        self.block_sizes = block_sizes

    def __call__(self, cube):
        """
        Map hypercube values to physical parameter values.

        Parameters
        ----------
        hypercube: list of floats
            Point coordinate on unit hypercube (in probabily space).
            See the PolyChord papers for more details.

        Returns
        -------
        theta: list of floats
            Physical parameter values corresponding to hypercube.
        """
        theta = np.zeros(cube.shape)
        start = 0
        end = 0
        for i, prior in enumerate(self.prior_blocks):
            end += self.block_sizes[i]
            theta[start:end] = prior(cube[start:end])
            start += self.block_sizes[i]
        return theta
