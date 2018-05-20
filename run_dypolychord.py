#!/usr/bin/env python
"""Generate results with dyPolyChord."""
import copy
import os
from mpi4py import MPI  # initialise MPI
import nestcheck.parallel_utils
import dyPolyChord.output_processing
import dyPolyChord.pypolychord_utils
import dyPolyChord
import bsr.likelihoods
import bsr.priors
import bsr.basis_functions as bf
import bsr.data


def main():
    """Get dyPolyChord runs (and standard nested sampling runs to compare them
    to)."""
    # Set up problem
    # --------------
    data = bsr.data.generate_data(bf.gg_2d, 1, 0.1, x_error_sigma=None)
    fit_func = bf.nn_2d
    adaptive = False
    if adaptive:
        nfunc_list = [4]
    else:
        nfunc_list = list(range(1, 5))
    # run settings
    start_ind = 0
    end_ind = start_ind + 1
    dynamic_goal = None
    nlive_per_dim = 20
    num_repeats_per_dim = 2
    # concurret futures parallel
    parallel = False
    max_workers = 6
    # dynamic settings - only used if dynamic_goal is not None
    seed_increment = 100
    ninit = 50
    clean = True
    init_step = ninit
    comm = MPI.COMM_WORLD
    # comm = None
    # PolyChord settings
    settings_dict = {
        'max_ndead': -1,
        'do_clustering': True,
        'posteriors': False,
        'equals': False,
        'base_dir': 'chains',
        'feedback': 1,
        'precision_criterion': 0.001,
        'nlives': {},
        'write_dead': True,
        'write_stats': True,
        'write_paramnames': False,
        'write_prior': False,
        'write_live': False,
        'write_resume': False,
        'read_resume': False,
        'cluster_posteriors': False,
        'boost_posterior': 0.0}
    for nfunc in nfunc_list:
        # Make likelihood, prior and run func
        likelihood = bsr.likelihoods.BasisFuncFit(
            data, fit_func, nfunc, adaptive=adaptive)
        prior = bsr.priors.get_default_prior(
            fit_func, nfunc, adaptive=adaptive)
        assert likelihood.ndim == sum(prior.block_sizes)
        # set nlive and num_repeats using ndim
        settings_dict['nlive'] = nlive_per_dim * likelihood.ndim
        settings_dict['num_repeats'] = num_repeats_per_dim * likelihood.ndim
        # make list of settings dictionaries for the different repeats
        file_root = likelihood.get_file_root(
            settings_dict['nlive'], settings_dict['num_repeats'],
            dynamic_goal=dynamic_goal)
        settings_list = []
        for extra_root in range(start_ind + 1, end_ind + 1):
            settings = copy.deepcopy(settings_dict)
            if comm is None or comm.Get_size() == 1:
                settings['seed'] = extra_root * (10 ** 3)
            else:
                settings['seed'] = -1
            settings['file_root'] = file_root + '_' + str(extra_root).zfill(3)
            settings_list.append(settings)
        # Before running in parallel make sure base_dir exists, as if multiple
        # threads try to make one at the same time mkdir throws an error.
        if not os.path.exists(settings_dict['base_dir']):
            os.makedirs(settings_dict['base_dir'])
        # Do the nested sampling
        # ----------------------
        run_func = dyPolyChord.pypolychord_utils.RunPyPolyChord(
            likelihood, prior, likelihood.ndim)
        if dynamic_goal is None:
            # For standard nested sampling just run PolyChord
            nestcheck.parallel_utils.parallel_apply(
                run_func, settings_list,
                max_workers=max_workers,
                parallel=parallel,
                tqdm_kwargs={'desc': 'dg=' + str(dynamic_goal),
                             'leave': True})
        else:
            # Dynamic nested sampling with dyPolyChord
            nestcheck.parallel_utils.parallel_apply(
                dyPolyChord.run_dypolychord, settings_list,
                max_workers=max_workers,
                func_pre_args=(run_func, dynamic_goal),
                func_kwargs={'ninit': ninit,
                             'init_step': init_step,
                             'seed_increment': seed_increment,
                             'clean': clean,
                             'nlive_const': settings_dict['nlive'],
                             'comm': comm},
                parallel=parallel,
                tqdm_kwargs={'desc': 'dg=' + str(dynamic_goal),
                             'leave': True})


if __name__ == '__main__':
    main()
