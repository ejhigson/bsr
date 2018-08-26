#!/usr/bin/env python
"""
Python priors for use with PolyChord.

PolyChord v1.14 requires priors to be callables with parameter and return
types:

Parameters
----------
hypercube: 1d numpy array
    Parameter positions in the prior hypercube.

Returns
-------
theta: 1d numpy array
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
import bsr.basis_functions as bf
import bsr.neural_networks as nn
import dyPolyChord.python_priors


def get_default_prior(func, nfunc, **kwargs):
    """Construct a default set of priors for the basis function."""
    global_bias = kwargs.pop('global_bias', False)
    adaptive = kwargs.pop('adaptive', False)
    sigma_w = kwargs.pop('sigma_w', 5)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    assert not global_bias
    # specify default priors
    if func.__name__[:2] == 'nn':
        # Neural network prior. Here nfunc is a list of numbers of nodes
        # Note all physical coordinates are scaled by hyperparameter (final
        # parameter in prior) and hence use sigma=1
        assert isinstance(nfunc, list)
        assert len(nfunc) >= 2
        prior_blocks = []
        block_sizes = []
        # Add sorted adaptive parameter on output weights
        prior_blocks.append(dyPolyChord.python_priors.Gaussian(
            1.0, adaptive=adaptive, sort=True, half=len(nfunc) == 2))
        if adaptive:
            block_sizes.append(nfunc[-1] + 1)
        else:
            block_sizes.append(nfunc[-1])
        # Priors on remaining weights
        prior_blocks.append(dyPolyChord.python_priors.Gaussian(1.0))
        block_sizes.append(nn.nn_num_params(nfunc) - nfunc[-1])
        # Priors on hyperparameter
        prior_blocks.append(dyPolyChord.python_priors.PowerUniform(
            0.1, 20, power=-2))
        block_sizes.append(1)
        return dyPolyChord.python_priors.BlockPrior(prior_blocks, block_sizes)
    elif func.__name__ == 'adfam_gg_ta_1d':
        assert adaptive
        # Need to explicitly provide all args rather than use **kwargs as
        # kwargs is now empty due to poping
        gg_prior = get_default_prior(bf.gg_1d, nfunc, global_bias=global_bias,
                                     adaptive=adaptive)
        ta_prior = get_default_prior(bf.ta_1d, nfunc, global_bias=global_bias,
                                     adaptive=adaptive)
        return AdFamPrior(gg_prior, ta_prior, nfunc)
    elif func.__name__ in ['gg_1d', 'gg_2d', 'ta_1d', 'ta_2d']:
        if func.__name__ in ['gg_1d', 'gg_2d']:
            priors_dict = {
                'a':     dyPolyChord.python_priors.Exponential(
                    1.0, adaptive=adaptive, sort=True),
                'mu':    dyPolyChord.python_priors.Uniform(0.0, 1.0),
                # 0.03 is approx pixel size in 32x32
                'sigma': dyPolyChord.python_priors.Uniform(0.03, 1.0),
                'beta':  dyPolyChord.python_priors.Exponential(0.5)}
            if func.__name__ == 'gg_2d':
                # reduce max sigma from 1.0 to 0.5 for 2d case
                priors_dict['sigma'] = dyPolyChord.python_priors.Uniform(
                    0.03, 0.5)
                for param in ['mu', 'sigma', 'beta']:
                    priors_dict[param + '1'] = priors_dict[param]
                    priors_dict[param + '2'] = priors_dict[param]
                    del priors_dict[param]  # del so error thrown if used
                priors_dict['omega'] = dyPolyChord.python_priors.Uniform(
                    -0.25 * np.pi, 0.25 * np.pi)
        elif func.__name__ in ['ta_1d', 'ta_2d']:
            priors_dict = {
                'a':   dyPolyChord.python_priors.Gaussian(
                    sigma_w, adaptive=adaptive,
                    sort=True, half=True),
                'w_0': dyPolyChord.python_priors.Gaussian(sigma_w),
                'w_1': dyPolyChord.python_priors.Gaussian(sigma_w),
                'w_2': dyPolyChord.python_priors.Gaussian(sigma_w)}
        # Get a list of the priors we want
        args = bf.get_bf_param_names(func)
        prior_blocks = [priors_dict[arg] for arg in args]
        block_sizes = [nfunc] * len(args)
        if adaptive:
            block_sizes[0] += 1
        return dyPolyChord.python_priors.BlockPrior(prior_blocks, block_sizes)
    else:
        raise AssertionError('not yet set up for {}'.format(func.__name__))


class AdFamPrior(object):

    """Prior for adaptive selection between different families of basis
    functions. First coordinate selects family, then different priors are
    applied to the remaining coordinates depending on its value."""

    def __init__(self, gg_1d_prior, ta_1d_prior, nfunc):
        """Store the different blocks and block sizes for each family."""
        self.gg_1d_prior = gg_1d_prior
        self.ta_1d_prior = ta_1d_prior
        self.nfunc = nfunc

    def __call__(self, cube):
        """
        Map hypercube values to physical parameter values.

        Parameters
        ----------
        hypercube: 1d numpy array
            Point coordinate on unit hypercube (in probabily space).
            See the PolyChord papers for more details.

        Returns
        -------
        theta: 1d numpy array
            Physical parameter values corresponding to hypercube.
        """
        theta = np.zeros(cube.shape)
        theta[0] = dyPolyChord.python_priors.Uniform(0.5, 2.5)(cube[0])
        # Calculate gg prior even if func is ta, so parameters unused by ta
        # are drawn from the gg prior
        theta[1:] = self.gg_1d_prior(cube[1:])
        if theta[0] >= 1.5:
            theta[1:-self.nfunc] = self.ta_1d_prior(cube[1:-self.nfunc])
        return theta
