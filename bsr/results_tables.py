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
import nestcheck.pandas_functions
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


def get_log_odds(run_list, nfunc_list, **kwargs):
    """Returns array of log odds ratios"""
    adfam = kwargs.pop('adfam', False)
    adaptive = kwargs.pop('adaptive', len(run_list) == 1)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    # Get funcs
    if adaptive:
        assert len(run_list) == 1, len(run_list)
        if adfam:
            funcs = [functools.partial(adaptive_logz, nfunc=nf, adfam_t=1)
                     for nf in nfunc_list]
            funcs += [functools.partial(adaptive_logz, nfunc=nf, adfam_t=2)
                      for nf in nfunc_list]
        else:
            funcs = [functools.partial(adaptive_logz, nfunc=nf) for nf in
                     nfunc_list]
    else:
        funcs = [nestcheck.estimators.logz]
    # Calculate values
    logzs = [nestcheck.ns_run_utils.run_estimators(run, funcs) for run in
             run_list]
    logzs = np.concatenate(logzs)
    log_odds = logzs - scipy.special.logsumexp(logzs)
    return log_odds


def get_log_odds_bs_resamp(run_list, nfunc_list, n_simulate=10, **kwargs):
    """BS resamples of get_log_odds."""
    threads_list = [nestcheck.ns_run_utils.get_run_threads(run) for run in
                    run_list]
    out_list = []
    for _ in range(n_simulate):
        run_list_temp = [nestcheck.error_analysis.bootstrap_resample_run(
            {}, threads) for threads in threads_list]
        out_list.append(get_log_odds(run_list_temp, nfunc_list, **kwargs))
    return np.vstack(out_list)


def get_bayes_df(run_list, run_list_sep, **kwargs):
    """Dataframe of Bayes factors."""
    n_simulate = kwargs.pop('n_simulate', 10)
    adaptive = kwargs.pop('adaptive')
    nfunc_list = kwargs.pop('nfunc_list')
    adfam = kwargs.pop('adfam')
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    log_odds = get_log_odds(run_list, nfunc_list, adaptive=adaptive,
                            adfam=adfam)
    columns = list(range(1, log_odds.shape[0] + 1))
    log_odds_bs = get_log_odds_bs_resamp(
        run_list, nfunc_list, adaptive=adaptive, adfam=adfam,
        n_simulate=n_simulate)
    df = nestcheck.pandas_functions.summary_df_from_array(
        log_odds_bs, columns)
    df = df.loc[df.index.get_level_values('calculation type') == 'std', :]
    df.index.set_levels('log odds ' + df.index.levels[0].astype(str),
                        level=0, inplace=True)
    df.loc[('log odds', 'value'), :] = log_odds
    df.loc[('log odds', 'uncertainty'), :] = \
        df.loc[('log odds std', 'value'), :]
    df.loc[('odds', 'value'), :] = np.exp(log_odds)
    df.loc[('odds', 'uncertainty'), :] = np.std(
        np.exp(log_odds_bs), axis=0, ddof=1)
    # add split values
    sep_log_odds = []
    sep_bs_stds = []
    for rl in run_list_sep:
        sep_log_odds.append(get_log_odds(
            rl, nfunc_list, adaptive=adaptive, adfam=adfam))
        bs_temp = get_log_odds_bs_resamp(
            rl, nfunc_list, adaptive=adaptive, adfam=adfam,
            n_simulate=n_simulate)
        sep_bs_stds.append(np.std(bs_temp, axis=0, ddof=1))
    sep_df = nestcheck.pandas_functions.summary_df_from_list(
        sep_log_odds, columns)
    sep_df.index.set_levels(
        'log odds sep ' + sep_df.index.levels[0].astype(str),
        level=0, inplace=True)
    df = pd.concat([df, sep_df])
    # Add sep runs implementation errors
    sep_bs = nestcheck.pandas_functions.summary_df_from_list(
        sep_bs_stds, columns)
    for rt in ['value', 'uncertainty']:
        df.loc[('sep bs std mean', rt), :] = sep_bs.loc[('mean', rt)]
    # get implementation stds
    imp_std, imp_std_unc, imp_frac, imp_frac_unc = \
        nestcheck.error_analysis.implementation_std(
            df.loc[('log odds sep std', 'value')].values,
            df.loc[('log odds sep std', 'uncertainty')].values,
            df.loc[('sep bs std mean', 'value')].values,
            df.loc[('sep bs std mean', 'uncertainty')].values)
    df.loc[('sep implementation std', 'value'), df.columns] = imp_std
    df.loc[('sep implementation std', 'uncertainty'), df.columns] = imp_std_unc
    df.loc[('sep implementation std frac', 'value'), :] = imp_frac
    df.loc[('sep implementation std frac', 'uncertainty'), :] = imp_frac_unc
    return df.sort_index()


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
    df_list = []
    for prob_key, prob_data in results_dict.items():
        meth_df_list = []
        adfam = ('adfam' in prob_key[0])
        nfunc_list = bsr.results_utils.nfunc_list_union(prob_data)
        if any(isinstance(nf, list) for nf in nfunc_list):
            assert all(isinstance(nf, list) for nf in nfunc_list), nfunc_list
            nfunc_list = [nf[-1] for nf in nfunc_list]
        prob_str = bsr.results_utils.root_given_key(prob_key)
        for meth_key, meth_data in prob_data.items():
            meth_str = bsr.results_utils.root_given_key(meth_key)
            save_name = 'cache/results_df_{}_{}_{}sim'.format(
                prob_str, meth_str, n_simulate)
            df_temp = get_method_df(
                meth_data, adaptive=meth_key[0], save_name=save_name,
                n_simulate=n_simulate, nfunc_list=nfunc_list,
                adfam=adfam, **kwargs)
            df_temp['method key'] = meth_str
            df_temp['problem key'] = prob_str
            meth_df_list.append(df_temp)
        df = pd.concat(meth_df_list)
        names = list(df.index.names)
        df.set_index(['problem key', 'method key'], append=True, inplace=True)
        df = df.reorder_levels(['problem key', 'method key'] + names)
        # Add eff gains
        vanilla_keys = [key for key in prob_data.keys() if not key[0]]
        if len(vanilla_keys) != 1:
            print('not adding gains as no single vanilla method',
                  prob_data.keys())
        else:
            van_str = bsr.results_utils.root_given_key(vanilla_keys[0])
            van_nsamp = df.loc[(prob_str, van_str, 'nsample', 'value')]
            for measure in ['log odds std', 'log odds sep std']:
                gain_type = measure.replace('std', 'gain')
                van_val = df.loc[(prob_str, van_str, measure, 'value')]
                van_unc = df.loc[(prob_str, van_str, measure, 'uncertainty')]
                for meth_key in prob_data.keys():
                    meth_str = bsr.results_utils.root_given_key(meth_key)
                    meth_nsamp = df.loc[(prob_str, meth_str, 'nsample', 'value')]
                    meth_val = df.loc[(prob_str, meth_str, measure, 'value')]
                    meth_unc = df.loc[(prob_str, meth_str, measure, 'uncertainty')]
                    nsamp_ratio = van_nsamp / meth_nsamp
                    gain, gain_unc = nestcheck.pandas_functions.get_eff_gain(
                        van_val, van_unc, meth_val, meth_unc,
                        adjust=nsamp_ratio)
                    df.loc[(prob_str, meth_str, gain_type, 'value'), :] = gain
                    df.loc[(prob_str, meth_str, gain_type, 'uncertainty'), :] = \
                        gain_unc
        df_list.append(df)
    return pd.concat(df_list)


@nestcheck.io_utils.save_load_result
def get_method_df(meth_data, **kwargs):
    """Results for a given method."""
    df = get_bayes_df(meth_data['run_list'], meth_data['run_list_sep'],
                      **kwargs)
    nsample = sum([run['logl'].shape[0] for run in meth_data['run_list']])
    nlike = sum([run['output']['nlike'] for run in meth_data['run_list']])
    df.loc[('nsample', 'value'), :] = nsample
    df.loc[('nlike', 'value'), :] = nlike
    return df
