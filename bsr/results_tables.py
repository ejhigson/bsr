#!/usr/bin/env python
"""Functions for plotting the results."""
import functools
import numpy as np
import pandas as pd
import scipy.special
import nestcheck.estimators
import nestcheck.ns_run_utils
import nestcheck.error_analysis
import nestcheck.parallel_utils
import nestcheck.io_utils
import bsr.priors
import bsr.results_utils


def adaptive_logz(run, logw=None, nfunc=1, adfam_t=None):
    """Get the logz assigned to nfunc basis functions from an adaptive run.
    Note that the absolute value does not correspond to that from a similar
    vanilla run, but the relative value can be used when calculating Bayes
    factors."""
    if logw is None:
        logw = nestcheck.ns_run_utils.get_logw(run)
    if isinstance(nfunc, list):
        nfunc = nfunc[-1]
    points = logw[select_adaptive_inds(run['theta'], nfunc, nfam=adfam_t)]
    if points.shape == (0,):
        return -np.inf
    else:
        return scipy.special.logsumexp(points)


def select_adaptive_inds(theta, nfunc, nfam=None):
    """Returns boolian mask of theta components which have the input B (and
    optionally also T) values.
    """
    if nfam is None:
        samp_nfunc = np.round(theta[:, 0]).astype(int)
        return samp_nfunc == nfunc
    else:
        samp_nfam = np.round(theta[:, 0]).astype(int)
        samp_nfunc = np.round(theta[:, 1]).astype(int)
        return np.logical_and((samp_nfunc == nfunc),
                              (samp_nfam == nfam))

def get_results_df(results_dict, **kwargs):
    """get results dataframe."""
    n_simulate = kwargs.pop('n_simulate', 10)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    df_list = []
    for prob_key, prob_data in results_dict.items():
        adfam = ('adfam' in prob_key[0])
        nfunc_list = bsr.results_utils.nfunc_list_union(prob_data)
        if any(isinstance(nf, list) for nf in nfunc_list):
            assert all(isinstance(nf, list) for nf in nfunc_list), nfunc_list
            nfunc_list = [nf[-1] for nf in nfunc_list]
        prob_key_str = bsr.results_utils.root_given_key(prob_key)
        for meth_key, meth_data in prob_data.items():
            meth_key_str = bsr.results_utils.root_given_key(meth_key)
            save_name = 'cache/results_df_{}_{}_{}sim'.format(
                prob_key_str, meth_key_str, n_simulate)
            df_temp = get_method_df(
                meth_data, meth_key[0], save_name=save_name,
                n_simulate=n_simulate, nfunc_list=nfunc_list,
                adfam=adfam)
            df_temp['method key'] = meth_key_str
            df_temp['problem key'] = prob_key_str
            df_list.append(df_temp)
    df = pd.concat(df_list)
    names = list(df.index.names)
    df.set_index(['problem key', 'method key'], append=True, inplace=True)
    return df.reorder_levels(['problem key', 'method key'] + names)


@nestcheck.io_utils.save_load_result
def get_method_df(meth_data, adaptive, **kwargs):
    """Results for a given method."""
    df = get_bayes_df(meth_data['run_list'], adaptive,
                      meth_data['run_list_sep'], **kwargs)
    nsample = sum([run['logl'].shape[0] for run in meth_data['run_list']])
    nlike = sum([run['output']['nlike'] for run in meth_data['run_list']])
    df['nsample'] = [nsample] + [np.nan] * 3
    df['nlike'] = [nlike] + [np.nan] * 3
    return df


def get_bayes_df(run_list, adaptive, run_list_sep, **kwargs):
    """Dataframe of Bayes factors."""
    n_simulate = kwargs.pop('n_simulate')
    nfunc_list = kwargs.pop('nfunc_list')
    adfam = kwargs.pop('adfam')
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    if not adaptive:
        # vanilla bayes
        bayes = np.asarray([nestcheck.estimators.logz(run) for run in
                            run_list])
        # std calculation of different runs parallelised and uses simulated
        # weights method as it gives the correct results for logZ
        bayes_unc = nestcheck.parallel_utils.parallel_apply(
            nestcheck.error_analysis.run_std_simulate, run_list,
            func_args=([nestcheck.estimators.logz],),
            func_kwargs={'n_simulate': n_simulate})
        bayes_unc = np.squeeze(np.asarray(bayes_unc))
        # Get an alternative std estimate from the variation of the
        # component runs
        bayes_split = np.zeros(bayes.shape)
        bayes_split_unc = np.zeros(bayes_unc.shape)
        for i, rls in enumerate(run_list_sep):
            bayes_temp = np.asarray([nestcheck.estimators.logz(run)
                                     for run in rls])
            bayes_split[i] = np.mean(bayes_temp)
            bayes_split_unc[i] = (np.std(bayes_temp, ddof=1)
                                  / np.sqrt(bayes_temp.shape[0]))
    else:
        # Adaptive bayes
        assert len(run_list) == 1
        if adfam:
            funcs = [functools.partial(adaptive_logz, nfunc=nf, adfam_t=1)
                     for nf in nfunc_list]
            funcs += [functools.partial(adaptive_logz, nfunc=nf, adfam_t=2)
                      for nf in nfunc_list]
        else:
            funcs = [functools.partial(adaptive_logz, nfunc=nf) for nf in
                     nfunc_list]
        bayes = nestcheck.ns_run_utils.run_estimators(run_list[0], funcs)
        bayes_unc = nestcheck.error_analysis.run_std_bootstrap(
            run_list[0], funcs, n_simulate=n_simulate)
        # Get an alternative std estimate from the variation of the
        # component runs
        bayes_split_vals = nestcheck.parallel_utils.parallel_apply(
            nestcheck.ns_run_utils.run_estimators, run_list_sep[0],
            func_args=(funcs,))
        # vstack so each row represents a run and each column a number of
        # functions
        bayes_split_vals = np.vstack(bayes_split_vals)
        bayes_split = np.mean(bayes_split_vals, axis=0)
        bayes_split_unc = np.std(bayes_split_vals, ddof=1, axis=0)
        bayes_split_unc /= np.sqrt(len(run_list_sep[0]))
        assert bayes_split_vals.shape[0] == len(run_list_sep[0])
        expected_nbayes = len(nfunc_list)
        if adfam:
            expected_nbayes *= 2
        assert bayes_split.shape[0] == expected_nbayes
        assert bayes_split_unc.shape[0] == expected_nbayes
    bayes -= bayes.max()
    bayes_split -= bayes_split.max()
    assert bayes.shape == bayes_unc.shape, [bayes.shape, bayes_unc.shape]
    assert bayes.shape == bayes_split.shape, [bayes.shape, bayes_split.shape]
    assert bayes.shape == bayes_split_unc.shape, (
        [bayes.shape, bayes_split_unc.shape])
    columns = list(range(1, bayes.shape[0] + 1))
    # Now put into df
    df_comb = pd.DataFrame(np.vstack([bayes, bayes_unc]),
                           index=['value', 'uncertainty'],
                           columns=columns)
    df_comb['calculation type'] = 'odds comb'
    df_split = pd.DataFrame(np.vstack([bayes_split, bayes_split_unc]),
                            index=['value', 'uncertainty'],
                            columns=columns)
    df_split['calculation type'] = 'odds split'
    df = pd.concat([df_comb, df_split])
    df.index.name = 'result type'
    df.set_index('calculation type', append=True, inplace=True)
    return df.reorder_levels(['calculation type', 'result type'])
