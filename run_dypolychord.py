#!/usr/bin/env python
"""Generate results with dyPolyChord."""
import copy
import os
import nestcheck.parallel_utils
import dyPolyChord.output_processing
import dyPolyChord.pypolychord_utils
import dyPolyChord
import bsr.likelihoods
import bsr.priors
import bsr.basis_functions as bf
import bsr.get_data


def main():
    """Get dyPolyChord runs (and standard nested sampling runs to compare them
    to)."""
    # Set up problem
    # --------------
    data = bsr.get_data.generate_data(bf.gg_1d, 1, 0.05)
    fit_func = bf.gg_1d
    nfunc = 1
    adaptive = False
    global_bias = False
    likelihood = bsr.likelihoods.BasisFuncSum(
        data, fit_func, nfunc, adaptive=adaptive, global_bias=global_bias)
    prior = bsr.priors.get_default_prior(
        fit_func, nfunc, adaptive=adaptive, global_bias=global_bias)
    assert likelihood.ndim == sum(prior.block_sizes)
    run_func = dyPolyChord.pypolychord_utils.RunPyPolyChord(
        likelihood, prior, likelihood.ndim)
    # run settings
    start_ind = 0
    end_ind = 1
    parallel = False
    max_workers = 6
    dg_list = [None]
    nlive_const = 100
    num_repeats = 5
    # dynamic settings - only used if dynamic_goal is not None
    seed_increment = 1
    ninit = 100
    clean = True
    init_step = ninit
    # PolyChord settings
    settings_dict = {
        'max_ndead': -1,
        'do_clustering': True,
        'posteriors': False,
        'equals': False,
        'num_repeats': num_repeats,
        'base_dir': 'chains',
        'feedback': 1,
        'precision_criterion': 0.001,
        'nlive': nlive_const,
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
    for dynamic_goal in dg_list:
        # make list of settings dictionaries for the different repeats
        file_root = likelihood.get_file_root(
            settings_dict['nlive'], settings_dict['num_repeats'],
            dynamic_goal=dynamic_goal)
        settings_list = []
        for extra_root in range(start_ind + 1, end_ind + 1):
            settings = copy.deepcopy(settings_dict)
            settings['seed'] = extra_root * (10 ** 3)
            settings['file_root'] = file_root + '_' + str(extra_root).zfill(3)
            settings_list.append(settings)
        # Before running in parallel make sure base_dir exists, as if multiple
        # threads try to make one at the same time mkdir throws an error.
        if not os.path.exists(settings_dict['base_dir']):
            os.makedirs(settings_dict['base_dir'])
        # Do the nested sampling
        # ----------------------
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
                             'nlive_const': nlive_const},
                parallel=parallel,
                tqdm_kwargs={'desc': 'dg=' + str(dynamic_goal),
                             'leave': True})


if __name__ == '__main__':
    main()
