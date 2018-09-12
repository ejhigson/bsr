#!/usr/bin/env python
"""Generate results using dyPolyChord."""
# pylint: disable=wrong-import-position
import copy
import os
import time
import sys
import traceback
from mpi4py import MPI  # initialise MPI
if 'ejh81' in os.getcwd().split('/'):  # running on cluster
    # Use non-interactive matplotlib backend. Must set this before pyplot
    # import.
    import matplotlib
    matplotlib.use('pdf')
import matplotlib.pyplot as plt
import nestcheck.parallel_utils
import dyPolyChord.output_processing
import dyPolyChord.pypolychord_utils
import dyPolyChord.polychord_utils
import bsr.basis_functions as bf
import bsr.neural_networks as nn  # pylint: disable=unused-import
import bsr.data
import bsr.results_utils
import bsr.results_tables
import bsr.paper_plots


def main():  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    """Get dyPolyChord runs (and standard nested sampling runs to compare them
    to). Optionally also make results plots. Edit the settings variables at the
    start of the function to determine what to run.

    Compiled C++ vs Python likelihoods
    ----------------------------------

    Whether to use the Python or compiled C++ likelihoods is determined by the
    "compiled" (bool) variable. The results should be the same but the C++
    likelihood is faster. All results can be run in either Python or C++ except
    those using bf.adfam_gg_ta_1d which are currently only set up in Python.

    The C++ likelihood is contained in the "CC_ini_likelihood.cpp" file and
    must be compiled before use. With PolyChord v1.15 already installed with
    MPI, this can be done with the following commands:

        $ cp CC_ini_likelihood.cpp
        ~/path_to_pc/PolyChord/likelihoods/CC_ini/CC_ini_likelihood.cpp
        $ cd ~/path_to_pc/PolyChord
        $ make polychord_CC_ini

    This creates an executable at PolyChord/bin/polychord_CC_ini. The ex_path
    variable must be updated to point to this. For more details see the
    PolyChord and dyPolyChord documentation.

    Parallelisation for compiled C++ likelihoods is performed through the
    mpi_str argument, and compute_results.py (this module) should not be
    executed with an mpirun command. In contrast Python likelihoods are run
    with parallelisation using an mpirun command on this module; for example

        $ mpirun -np 6 python3 compute_results.py

    Data Sets and Fitting Functions
    -------------------------------

    The problem to be solved is specifice in a tuple of the form:

        problem_tup = (fit_func, data_func, data_type)

    where

    * fit_func is the function with which to fit the data (e.g. a basis
      function or neural network).
    * data_func specifies the type of data (basis function from which the true
      signal was sampled, or alternatively bsr.data.get_image for astronomical
      images).
    * data_type specifies which of the data sets using that data_func to use.
      For signals made from a mixture model of basis functions, this is the
      number of mixture components. For astronomical images it specifies which
      of the .jpeg images in bsr/images/ to use.

    The problem tuples corresponding to the results in the paper are:

    Figure 6a: (bf.gg_1d, bf.gg_1d, 1)
    Figure 6b: (bf.gg_1d, bf.gg_1d, 2)
    Figure 6c: (bf.gg_1d, bf.gg_1d, 3)
    Figure 8a: (bf.ta_1d, bf.ta_1d, 1)
    Figure 8b: (bf.ta_1d, bf.ta_1d, 2)
    Figure 8c: (bf.ta_1d, bf.ta_1d, 3)
    Figure 10a: (bf.adfam_gg_ta_1d, bf.gg_1d, 1)
    Figure 10b: (bf.adfam_gg_ta_1d, bf.ta_1d, 1)
    Figure 11a: (bf.gg_2d, bf.gg_2d, 1)
    Figure 11b: (bf.gg_2d, bf.gg_2d, 2)
    Figure 11c: (bf.gg_2d, bf.gg_2d, 3)
    Figure 13a: (bf.gg_2d, bsr.data.get_image, 1)
    Figure 13b: (bf.gg_2d, bsr.data.get_image, 2)
    Figure 13c: (bf.gg_2d, bsr.data.get_image, 3)
    Figure 16: (nn.nn_adl, bf.gg_2d, 3)
    Figure 18a: (nn.nn_2l, bsr.data.get_image, 2)
    Figure 18b: (nn.nn_2l, bsr.data.get_image, 3)

    Vanilla, adaptive and dynamic adaptive methods
    ----------------------------------------------

    The method with which to solve it is specified in a tuple of the form:

        method_tup = (adaptive, dynamic_goal, nlive, num_repeats)

    where

    * adaptive (bool) is whether or not to use the adaptive method.
    * dynamic_goal is None for standard nested sampling and otherwise
      corresponds to dyPolyChord's dynamic_goal setting.
    * nlive is the number of live points.
    * num_repeats is the PolyChord/dyPolyChord num_repeats setting.

    In the paper, all fits of a 1-dimensional data use

        method_tups_1d = [(False, None, 200, 100),
                          (True, None, 1000, 100),
                          (True, 1, 1000, 100)]

    and fits of 2-dimensional images use

        method_tups_2d = [(False, None, 400, 250),
                          (True, None, 2000, 250),
                          (True, 1, 2000, 250)].

    In each case the 3 list elements correspond to the vanilla, adaptive and
    dynamic_adaptive settings.
    """
    # ###################################################################
    # Settings
    # ###################################################################
    # problem
    # -------
    fit_func_list = [nn.nn_1l]
    data_func_list = [bf.gg_2d]
    data_type_list = [1, 2, 3]
    inds = list(range(1, 6))
    # method
    # ------
    # 1d data
    method_tups_1d = [(False, None, 200, 100),
                      (True, None, 1000, 100),
                      (True, 1, 1000, 100)]
    # 2d data
    method_tups_2d = [(False, None, 400, 250),
                      (True, None, 2000, 250),
                      (True, 1, 2000, 250)]
    # Runtime settings
    # ----------------
    compiled = True  # Whether to use compiled C++ likelihood or Python
    run_vanilla = True  # Whether or not to run vanilla results
    run_ad_none = True  # Whether or not to run adaptive results
    run_ad_dg_1 = True  # Whether or not to run dynamic adaptive results
    process_results = True  # Whether or not to plot results
    # PolyChord settings
    # ------------------
    use_mpi = True  # only affects compiled == True
    settings_dict = {
        'max_ndead': -1,
        'do_clustering': True,
        'posteriors': False,
        'equals': False,
        'base_dir': 'chains',
        'feedback': -1,
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
    # ###################################################################
    # Run program
    # ###################################################################
    problem_tups = []
    for fit_func in fit_func_list:
        for data_func in data_func_list:
            for data_type in data_type_list:
                problem_tups.append((fit_func, data_func, data_type))
    # Initialise MPI
    comm = MPI.COMM_WORLD
    if compiled:
        assert comm.Get_size() == 1, (
            'compiled likelihoods should use mpi via mpi_str. '
            'comm.Get_size()={}').format(comm.Get_size())
        mpi_str = None
        # Work out if we are running on cluster or laptop
        if 'ed' in os.getcwd().split('/'):
            ex_path = '/home/ed/code/'
            if use_mpi:
                mpi_str = 'mpirun -np 6'
        else:
            ex_path = '/home/ejh81/'
            if use_mpi:
                mpi_str = 'mpirun -np 32'
        ex_path += 'PolyChord/bin/polychord_CC_ini'
    # Before running in parallel make sure base_dir exists, as if
    # multiple threads try to make one at the same time mkdir
    # throws an error.
    if comm.Get_rank() == 0:
        print('compiled={} mpi={}'.format(
            compiled, comm.Get_size() if not compiled else mpi_str))
        sys.stdout.flush()  # Make sure message prints
        try:
            if not os.path.exists(settings_dict['base_dir']):
                os.makedirs(settings_dict['base_dir'])
            if not os.path.exists(settings_dict['base_dir'] + '/clusters'):
                os.makedirs(settings_dict['base_dir'] + '/clusters')
        except:  # pylint: disable=bare-except
            if comm is None or comm.Get_size() == 1:
                raise
            else:
                # print error info
                traceback.print_exc(file=sys.stdout)
                print('Error in process with rank == 0: forcing MPI abort.')
                sys.stdout.flush()  # Make sure message prints before abort
                comm.Abort(1)  # Force all processes to abort
    # Generate nested sampling runs
    # -----------------------------
    # pylint: disable=too-many-nested-blocks
    for problem_tup in problem_tups:
        fit_func, data_func, data_type = problem_tup
        if '1d' in fit_func.__name__:
            method_tups = method_tups_1d
        else:
            method_tups = method_tups_2d
        problem_key = bsr.results_utils.get_problem_key(*problem_tup)
        base_dict = bsr.results_utils.make_base_dict(
            [problem_tup], method_tups)
        for method_key in method_tups:
            adaptive, dynamic_goal, nlive, num_repeats = method_key
            if not run_vanilla and not adaptive:
                continue
            if not run_ad_none and adaptive and dynamic_goal is None:
                continue
            if not run_ad_dg_1 and adaptive and dynamic_goal == 1:
                continue
            if fit_func.__name__ in ['adfam_gg_ta_1d', 'nn_adl']:
                if not adaptive:
                    continue
            elif fit_func.__name__[-2:] == '1d':
                if fit_func.__name__ != data_func.__name__:
                    # Only run for vanilla with data type=1
                    if adaptive or data_type != 1:
                        continue
            if comm.Get_rank() == 0:
                print('Starting: {} {} nrun={}'
                      .format(problem_key, method_key, len(inds)))
                sys.stdout.flush()  # Make sure message prints
            prob_meth_dict = base_dict[problem_key][method_key]
            for i, likelihood in enumerate(prob_meth_dict['likelihood_list']):
                # Make likelihood, prior and run func
                prior = prob_meth_dict['prior_list'][i]
                # Do the nested sampling
                # ----------------------
                settings_dict['nlive'] = nlive
                settings_dict['num_repeats'] = num_repeats
                # make list of settings dictionaries for the different repeats
                file_root = likelihood.get_file_root(
                    settings_dict['nlive'], settings_dict['num_repeats'],
                    dynamic_goal=dynamic_goal)
                settings_list = []
                for extra_root in inds:
                    settings = copy.deepcopy(settings_dict)
                    if comm is None or comm.Get_size() == 1:
                        settings['seed'] = extra_root * (10 ** 3)
                    else:
                        settings['seed'] = -1
                    settings['file_root'] = (file_root + '_'
                                             + str(extra_root).zfill(3))
                    settings_list.append(settings)
                # concurret futures parallel settings
                parallel = False  # if already running with MPI, set to False
                parallel_warning = False
                max_workers = 6
                tqdm_kwargs = {'disable': True}
                if not compiled:
                    run_func = dyPolyChord.pypolychord_utils.RunPyPolyChord(
                        likelihood, prior, likelihood.ndim)
                else:
                    prior_str = bsr.priors.bsr_prior_to_str(prior)
                    run_func = (dyPolyChord.polychord_utils
                                .RunCompiledPolyChord(
                                    ex_path, prior_str, mpi_str=mpi_str,
                                    config_str=likelihood.cpp_config_str()))
                start_time = time.time()
                if dynamic_goal is None:
                    # For standard nested sampling just run PolyChord
                    nestcheck.parallel_utils.parallel_apply(
                        run_func, settings_list,
                        parallel=parallel, parallel_warning=parallel_warning,
                        max_workers=max_workers, tqdm_kwargs=tqdm_kwargs)
                else:
                    ninit = int(settings_dict['nlive'] // 2)
                    init_step = ninit * 5
                    if comm.Get_size() > 1:
                        seed_increment = -1
                    else:
                        seed_increment = 1
                    # Dynamic nested sampling with dyPolyChord
                    nestcheck.parallel_utils.parallel_apply(
                        dyPolyChord.run_dynamic_ns.run_dypolychord,
                        settings_list,
                        func_pre_args=(run_func, dynamic_goal),
                        func_kwargs={'ninit': ninit,
                                     'init_step': init_step,
                                     'seed_increment': seed_increment,
                                     'clean': True,
                                     'nlive_const': settings_dict['nlive'],
                                     'comm': None if compiled else comm},
                        parallel=parallel, parallel_warning=parallel_warning,
                        max_workers=max_workers, tqdm_kwargs=tqdm_kwargs)
                # Remove <root>_dead.txt as we only need <root>_dead-birth.txt
                if comm.Get_rank() == 0:
                    try:
                        for settings in settings_list:
                            file_root = os.path.join(settings['base_dir'],
                                                     settings['file_root'])
                            os.remove(file_root + '_dead.txt')
                        end_time = time.time()
                        print('{} {} nrun={} nfunc={} took {:.2f}sec'.format(
                            problem_key, method_key, len(inds),
                            likelihood.nfunc, (end_time - start_time)))
                        sys.stdout.flush()  # Make sure message prints
                    except:  # pylint: disable=bare-except
                        if comm is None or comm.Get_size() == 1:
                            raise
                        else:
                            # print error info
                            traceback.print_exc(file=sys.stdout)
                            # force all MPI processes to abort to avoid hanging
                            comm.Abort(1)
    # Generate results and make plots
    # -------------------------------
    if comm.Get_rank() == 0 and process_results:
        print('Runs are all finished')
        print('Now try plotting results')
        try:
            print('inds run:', inds)
            inds_to_plot = list(range(1, max(inds) + 1))
            print('inds_to_plot:', inds_to_plot)
            for problem_tup in problem_tups:
                fit_func, data_func, data_type = problem_tup
                problem_key = bsr.results_utils.get_problem_key(*problem_tup)
                if '1d' in fit_func.__name__:
                    method_tups = method_tups_1d
                else:
                    method_tups = method_tups_2d
                print('Starting: {} nrun={}'
                      .format(problem_key, len(inds_to_plot)))
                # Load data
                results_dict = bsr.results_utils.load_data(
                    [problem_tup], method_tups, inds_to_plot)
                nrun_list = []
                for meth_key in method_tups:
                    nrun = len(
                        results_dict[problem_key][meth_key]['run_list_sep'])
                    nrun_list.append(nrun)
                    if nrun != len(inds_to_plot):
                        print('{}: only got {} runs'.format(meth_key, nrun))
                # Get and cache results dfs
                results_df = bsr.results_tables.get_results_df(
                    results_dict, load=False)
                # Make plots
                _ = bsr.paper_plots.odds(
                    results_df, nruns=min(nrun_list))
                _ = bsr.paper_plots.multi(
                    results_dict,
                    meth_condition=(lambda x: x[1] == 1))  # only dyn adapt
                _ = bsr.paper_plots.split(
                    results_dict,
                    meth_condition=(lambda x: not x[0]))  # only plot Vanilla
                plt.close('all')
                sys.stdout.flush()  # Make sure message prints
        except:  # pylint: disable=bare-except
            if comm is None or comm.Get_size() == 1:
                raise
            else:
                # print error info
                traceback.print_exc(file=sys.stdout)
                sys.stdout.flush()  # Make sure message prints
                # force all MPI processes to abort to avoid hanging
                comm.Abort(1)


if __name__ == '__main__':
    main()
