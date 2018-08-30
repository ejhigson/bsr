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
import copy
import numpy as np
import bsr.basis_functions as bf
import bsr.neural_networks as nn
import dyPolyChord.python_priors
import dyPolyChord.polychord_utils


def get_default_prior(func, nfunc, **kwargs):
    """Construct a default set of priors for the basis function."""
    adaptive = kwargs.pop('adaptive', False)
    sigma_w = kwargs.pop('sigma_w', 5)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    # specify default priors
    if func.__name__ == 'nn_adl':
        assert isinstance(nfunc, list)
        assert len(nfunc) == 3
        assert adaptive
        nfunc_1l = [nfunc[0], nfunc[-1]]
        return AdFamPrior([
            get_default_prior(nn.nn_1l, nfunc_1l, adaptive=adaptive),
            get_default_prior(nn.nn_2l, nfunc, adaptive=adaptive)])
    elif func.__name__[:2] == 'nn':
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
        return AdFamPrior([
            get_default_prior(bf.gg_1d, nfunc, adaptive=adaptive),
            get_default_prior(bf.ta_1d, nfunc, adaptive=adaptive)])
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

    def __init__(self, block_prior_list):
        """Store the block priors for each family.

        Parameters
        ----------
        block_prior_list: list of block prior objects
            Must have length 2.
            First element corresponds to family T=1 and second element to
            family T=2.
        """
        assert len(block_prior_list) == 2, len(block_prior_list)
        # Find out which list is longer
        prior_sizes = [sum(pri.block_sizes) for pri in block_prior_list]
        if prior_sizes[0] >= prior_sizes[1]:
            self.long_prior = block_prior_list[0]
            self.long_prior_t = 1
            self.short_prior = block_prior_list[1]
            self.short_prior_size = prior_sizes[1]
        else:
            self.long_prior = block_prior_list[1]
            self.long_prior_t = 2
            self.short_prior = block_prior_list[0]
            self.short_prior_size = prior_sizes[0]

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
        # Apply the longer of the two priors, as we want the trailing values to
        # be filled in with this prior even if the shorter perio is chosen
        theta[1:] = self.long_prior(cube[1:])
        try:
            nfam = int(np.round(theta[0]))
        except ValueError:
            return np.full(cube.shape, np.nan)
        assert nfam in [1, 2], nfam
        if nfam != self.long_prior_t:
            theta[1:self.short_prior_size + 1] = self.short_prior(
                cube[1:self.short_prior_size + 1])
        return theta


def bsr_prior_to_str(prior_obj):
    """Helper for mapping priors to PolyChord ini strings which deals with
    adaptive family priors."""
    if type(prior_obj).__name__ == 'BlockPrior':
        return dyPolyChord.polychord_utils.python_block_prior_to_str(prior_obj)
    assert type(prior_obj).__name__ == 'AdFamPrior', (
        'Unexpected input object type: {}'.format(
            type(prior_obj).__name__))
    # Check this is a Neural Networks adfam prior (not set up for 1d basis
    # functions one)
    assert type(prior_obj.long_prior.prior_blocks[0]).__name__ == 'Gaussian', (
        type(prior_obj.long_prior.prior_blocks[0]).__name__)
    assert prior_obj.long_prior.prior_blocks[0].sort
    assert prior_obj.long_prior.prior_blocks[0].adaptive
    assert not prior_obj.long_prior.prior_blocks[0].half
    temp_prior = copy.deepcopy(prior_obj.long_prior)
    temp_prior.block_sizes[0] += 1
    string = dyPolyChord.polychord_utils.python_block_prior_to_str(temp_prior)
    return string.replace('adaptive_sorted_gaussian',
                          'nn_adaptive_layer_gaussian')
