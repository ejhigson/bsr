#!/usr/bin/env python
"""Utilites for loading and processing results for the paper."""
import nestcheck.data_processing
import nestcheck.ns_run_utils
import bsr.likelihoods
import bsr.priors
import bsr.data
import bsr.basis_functions as bf


def load_data(problem_tups, method_tups, inds, **kwargs):
    """Load run data."""
    sep_runs = kwargs.pop('sep_runs', True)
    base_dir = kwargs.pop('base_dir', 'chains')
    results_dict = make_base_dict(problem_tups, method_tups, **kwargs)
    for problem_tup in problem_tups:
        prob_key = get_problem_key(*problem_tup)
        print(prob_key)
        for method_key in method_tups:
            _, dynamic_goal, nlive, num_repeats = method_key
            likelihood_list = \
                results_dict[prob_key][method_key]['likelihood_list']
            run_list = []
            run_list_sep = []
            for likelihood in likelihood_list:
                root = likelihood.get_file_root(nlive, num_repeats,
                                                dynamic_goal)
                batch = nestcheck.data_processing.batch_process_data(
                    [root + '_' + str(i).zfill(3) for i in inds],
                    parallel=False, parallel_warning=False,
                    tqdm_kwargs={'disable': True}, base_dir=base_dir,
                    errors_to_handle=(AssertionError, OSError))
                if sep_runs:
                    run_list_sep.append(batch)
                try:
                    run_list.append(nestcheck.ns_run_utils.combine_ns_runs(
                        batch))
                except ValueError:
                    # Occurs when no runs loaded
                    pass
            try:
                sep_list = []
                assert len(likelihood_list) == len(run_list)
                results_dict[prob_key][method_key]['run_list'] = run_list
                if run_list_sep:
                    assert all([len(rl) == len(inds) for rl in run_list_sep])
                    # Reformat run_list_sep so each element has same format as
                    # run list
                    for nrep in range(len(inds)):
                        sep_list.append([rl[nrep] for rl in run_list_sep])
                    assert len(sep_list) == len(inds)
                    assert all([len(rl) == len(run_list) for rl in sep_list])
                    results_dict[prob_key][method_key]['run_list_sep'] = \
                        sep_list
            except AssertionError:
                if sep_list:  # dont catch problems in sep_list
                    raise
                print('runs missing for method_key={}'.format(method_key))
                # delete method keys with no data
                del results_dict[prob_key][method_key]
            # delete problem keys with no method keys
        if not results_dict[prob_key]:
            del results_dict[prob_key]
    return results_dict


def label_given_method_str(method):
    """Get method labels for bar charts and tables."""
    # get label
    label = ''
    if method.split('_')[0] == 'True':
        label += 'adaptive'
    else:
        assert method.split('_')[0] == 'False', method
        label += 'vanilla'
    if method.split('_')[1] != 'None':
        label = 'dynamic ' + label
    return label


def nfunc_list_union(problem_data):
    """Given dict of results for different methods on a given problem, find
    union of all the nfuncs considered."""
    nfunc_list_list = []
    for key, item in problem_data.items():
        if key[0]:  # adaptive
            assert len(item['nfunc_list']) == 1
            try:
                nfunc_list_list += list(range(1, item['nfunc_list'][0] + 1))
            except TypeError:
                nfl = item['nfunc_list'][0]
                assert isinstance(nfl, list)
                assert len(set(nfl[1:])) == 1  # hidden layers has same #nodes
                n_hidden = len(nfl) - 1
                for i in range(1, nfl[1] + 1):
                    nfunc_list_list.append(nfl[:1] + [i] * n_hidden)
        else:
            nfunc_list_list += item['nfunc_list']
    try:
        return sorted(list(set(nfunc_list_list)))
    except TypeError:
        # For neural networks where nfuncs is a list, we need to convert the
        # lists to tuples in order to eliminate duplicates then convert them
        # back to lists
        nfunc_list_list = [tuple(nfl) for nfl in nfunc_list_list]
        nfunc_list_list = list(set(nfunc_list_list))
        nfunc_list_list = [list(nfl) for nfl in nfunc_list_list]
        for nfl in nfunc_list_list:
            assert isinstance(nfl, list)
            assert nfl[0] == nfunc_list_list[0][0]  # same dim of input
            assert len(nfl) == len(nfunc_list_list[0])  # same number of layers
            assert len(set(nfl[1:])) == 1  # each hidden layer has same #nodes
        return sorted(nfunc_list_list, key=lambda x: x[1])


def get_nfunc_list(fit_func, adaptive, data, nfunc_max_dict):
    """Makes nfunc list."""
    nfunc_max = nfunc_max_dict[fit_func.__name__[:2]]
    if adaptive:
        nfunc_list = [nfunc_max]
    else:
        nfunc_list = list(range(1, nfunc_max + 1))
    if fit_func.__name__[:2] == 'nn':
        if fit_func.__name__[-2:] == '2l':
            layers = 2
        else:
            layers = 1
        nfunc_list = [layers * [nf] for nf in nfunc_list]
        if data['x2'] is None:
            nfunc_list = [[1] + nf for nf in nfunc_list]
        else:
            nfunc_list = [[2] + nf for nf in nfunc_list]
    return nfunc_list


def make_base_dict(problem_tups, method_tups, **kwargs):
    """Creates a nested dictionary structure for different problems (data and
    fit function) and methods."""
    nfunc_max_dict = kwargs.pop('nfunc_max_dict',
                                {'gg': 5, 'ta': 5, 'nn': 10, 'ad': 5})
    y_sigma_2d = kwargs.pop('y_sigma_2d', 0.2)
    y_sigma_1d = kwargs.pop('y_sigma_1d', 0.05)
    x_sigma_1d = kwargs.pop('x_sigma_1d', 0.05)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    output = {}
    for fit_func, data_func, data_type in problem_tups:
        problem_key = get_problem_key(fit_func, data_func, data_type)
        output[problem_key] = {}
        # Get data
        if data_func.__name__[-2:] == '1d':
            y_error_sigma = y_sigma_1d
            x_error_sigma = x_sigma_1d
        else:
            y_error_sigma = y_sigma_2d
            x_error_sigma = None
        data = bsr.data.generate_data(
            data_func, data_type, y_error_sigma,
            x_error_sigma=x_error_sigma)
        for method_key in method_tups:
            output[problem_key][method_key] = {}
            adaptive = method_key[0]
            nfunc_list = get_nfunc_list(
                fit_func, adaptive, data, nfunc_max_dict)
            likelihood_list = []
            prior_list = []
            if fit_func.__name__[:5] == 'adfam' and not adaptive:
                for nfunc in nfunc_list:
                    likelihood_list.append(bsr.likelihoods.FittingLikelihood(
                        data, bf.gg_1d, nfunc, adaptive=adaptive))
                    prior_list.append(bsr.priors.get_default_prior(
                        bf.gg_1d, nfunc, adaptive=adaptive))
                for nfunc in nfunc_list:
                    likelihood_list.append(bsr.likelihoods.FittingLikelihood(
                        data, bf.ta_1d, nfunc, adaptive=adaptive))
                    prior_list.append(bsr.priors.get_default_prior(
                        bf.ta_1d, nfunc, adaptive=adaptive))
            else:
                for nfunc in nfunc_list:
                    # Make likelihood, prior and run func
                    likelihood_list.append(bsr.likelihoods.FittingLikelihood(
                        data, fit_func, nfunc, adaptive=adaptive))
                    prior_list.append(bsr.priors.get_default_prior(
                        fit_func, nfunc, adaptive=adaptive))
            output[problem_key][method_key]['nfunc_list'] = nfunc_list
            output[problem_key][method_key]['likelihood_list'] = \
                likelihood_list
            output[problem_key][method_key]['prior_list'] = prior_list
    return output


def root_given_key(prob_key):
    """Turn the tuple keys into a string."""
    root = ''
    for i, info in enumerate(prob_key):
        if i != 0:
            root += '_'
        root += str(info)
    return root.replace('.', '_')


def get_problem_key(fit_func, data_func, data_type):
    """Map the problem tuple (which contains functions) to a key tuple."""
    return (fit_func.__name__, data_func.__name__, data_type)
