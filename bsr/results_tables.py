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


def get_y_mean(run_list, likelihood_list, x1, x2=None):
    """Evaluate mean output y at coordinates x1,x2."""
    assert len(run_list) == len(likelihood_list)
    if len(run_list) == 1:
        factors = [1]
    else:
        logzs = np.asarray([nestcheck.estimators.logz(run) for run in
                            run_list])
        factors = np.exp(logzs - logzs.max())
        factors /= np.sum(factors)
    y_mean = np.zeros(x1.shape)
    for i, run in enumerate(run_list):
        w_rel = nestcheck.ns_run_utils.get_w_rel(run)
        y_mean += factors[i] * likelihood_list[i].fit_mean(
            run['theta'], x1, x2, w_rel=w_rel)
    return y_mean


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

def adaptive_prob(run, logw=None, **kwargs):
    """Convenience wrapper for computing adaptive Z as fraction of the total
    Z."""
    if logw is None:
        logw = nestcheck.ns_run_utils.get_logw(run)
    ad_logz = adaptive_logz(run, logw=logw, **kwargs)
    return np.exp(ad_logz - scipy.special.logsumexp(logw))

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
            # Get family prob using adaptive_logz=1 func, as T is stored in the
            # first column
            funcs.append(functools.partial(adaptive_logz, nfunc=1))
        else:
            funcs = [functools.partial(adaptive_logz, nfunc=nf) for nf in
                     nfunc_list]
    else:
        funcs = [nestcheck.estimators.logz]
    # Calculate values
    logzs_list = [nestcheck.ns_run_utils.run_estimators(run, funcs)
                  for run in run_list]
    log_odds = np.concatenate(logzs_list)
    if not adfam:
        log_odds -= scipy.special.logsumexp(log_odds)
        # check the probabilities approximately sum to 1
        prob_sum = np.exp(scipy.special.logsumexp(log_odds))
    else:
        # Dont include final element in sum as this is the total for all N,T
        # combos with T=1, so it will lead to double counting
        log_odds -= scipy.special.logsumexp(log_odds[:-1])
        prob_sum = np.exp(scipy.special.logsumexp(log_odds[:-1]))
    assert np.isclose(prob_sum, 1), prob_sum
    return log_odds


def get_log_odds_bs_resamp(run_list, nfunc_list, n_simulate=10, **kwargs):
    """BS resamples of get_log_odds."""
    # NB thread list must be in same order as run list, so can't use parallel
    # apply
    threads_list = [nestcheck.ns_run_utils.get_run_threads(run) for run in
                    run_list]
    out_list = nestcheck.parallel_utils.parallel_apply(
        log_odds_bs_resamp_helper, list(range(n_simulate)),
        func_args=(threads_list, nfunc_list), func_kwargs=kwargs,
        tqdm_kwargs={'disable': True})
    return np.vstack(out_list)


def log_odds_bs_resamp_helper(_, threads_list, nfunc_list, **kwargs):
    """Helper for parallelising."""
    run_list_temp = [nestcheck.error_analysis.bootstrap_resample_run(
        {}, threads) for threads in threads_list]
    return get_log_odds(run_list_temp, nfunc_list, **kwargs)


def get_bayes_df(run_list, run_list_sep, **kwargs):
    """Dataframe of Bayes factors."""
    n_simulate = kwargs.pop('n_simulate', 5)  # 10
    adaptive = kwargs.pop('adaptive')
    nfunc_list = kwargs.pop('nfunc_list')
    adfam = kwargs.pop('adfam')
    inc_log_odds = kwargs.pop('inc_log_odds', False)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    # Get log odds ratios from combined runs
    log_odds = get_log_odds(run_list, nfunc_list, adaptive=adaptive,
                            adfam=adfam)
    log_odds_resamps = get_log_odds_bs_resamp(
        run_list, nfunc_list, adaptive=adaptive, adfam=adfam,
        n_simulate=n_simulate)
    # Get log odds ratios from split runs
    sep_log_odds = []
    sep_log_odds_resamps = []
    for rl in run_list_sep:
        sep_log_odds.append(get_log_odds(
            rl, nfunc_list, adaptive=adaptive, adfam=adfam))
        sep_log_odds_resamps.append(get_log_odds_bs_resamp(
            rl, nfunc_list, adaptive=adaptive, adfam=adfam,
            n_simulate=n_simulate))
    # Get info df
    col_names = [r'$P(N={})$'.format(i + 1) for
                 i in range(log_odds.shape[0])]
    if adfam:
        col_names[-1] = r'$P(T={})$'.format(1)
    df = get_sep_comb_df(
        np.exp(log_odds),
        np.exp(log_odds_resamps),
        [np.exp(arr) for arr in sep_log_odds],
        [np.exp(arr) for arr in sep_log_odds_resamps],
        col_names=col_names)
    if inc_log_odds:
        col_names = [r'$\log P(N={})$'.format(i + 1) for
                     i in range(log_odds.shape[0])]
        if adfam:
            col_names[-1] = r'$P(T={})$'.format(1)
        log_odds_df = get_sep_comb_df(
            log_odds,
            log_odds_resamps,
            sep_log_odds,
            sep_log_odds_resamps,
            col_names)
        df = pd.concat([df, log_odds_df], axis=1)
    return df


def get_sep_comb_df(values, bs_resamps, sep_values, sep_bs_resamps,
                    col_names):
    """Get dataframe information on combined and seperate runs.

    Parameters
    ----------
    values: 1d numpy array
    bs_resamps: 2d numpy array
    sep_values: list of 1d numpy arrays
    sep_bs_resamps: list of 2d numpy arrays
    """
    # Check shapes
    assert values.shape[0] == bs_resamps.shape[1]
    assert len(sep_values) == len(sep_bs_resamps)
    assert len(set([ar.shape for ar in sep_values])) == 1
    assert len(set([ar.shape for ar in sep_bs_resamps])) == 1
    assert values.shape == sep_values[0].shape
    assert bs_resamps.shape == sep_bs_resamps[0].shape
    assert len(col_names) == values.shape[0]
    # Create empty MultiIndex dataframe using nestcheck's summary df
    df = nestcheck.pandas_functions.summary_df_from_array(
        bs_resamps, col_names)
    # Note estunated uncertainty on combined = bs std
    df.loc[('combined', 'value'), :] = values
    df.loc[('combined', 'uncertainty'), :] = df.loc[('std', 'value'), :]
    df.loc[('bs std', 'value'), :] = df.loc[('std', 'value'), :]
    df.loc[('bs std', 'uncertainty'), :] = df.loc[('std', 'uncertainty'), :]
    df = df.iloc[-4:]
    # Add seperate values
    sep_df = nestcheck.pandas_functions.summary_df_from_list(
        sep_values, col_names)
    sep_df.index.set_levels(
        'sep ' + sep_df.index.levels[0].astype(str),
        level=0, inplace=True)
    df = pd.concat([df, sep_df])
    # Get mean sep bs std and its uncertainty
    sep_bs_df = nestcheck.pandas_functions.summary_df_from_list(
        [np.std(arr, axis=0, ddof=1) for arr in sep_bs_resamps], col_names)
    for rt in ['value', 'uncertainty']:
        df.loc[('sep bs std mean', rt), :] = sep_bs_df.loc[('mean', rt)]
    # Add sep runs implementation errors
    imp_std, imp_std_unc, imp_frac, imp_frac_unc = \
        nestcheck.error_analysis.implementation_std(
            df.loc[('sep std', 'value')].values,
            df.loc[('sep std', 'uncertainty')].values,
            df.loc[('sep bs std mean', 'value')].values,
            df.loc[('sep bs std mean', 'uncertainty')].values)
    df.loc[('sep imp std', 'value'), df.columns] = imp_std
    df.loc[('sep imp std', 'uncertainty'), df.columns] = imp_std_unc
    df.loc[('sep imp std frac', 'value'), :] = imp_frac
    df.loc[('sep imp std frac', 'uncertainty'), :] = imp_frac_unc
    return df # .sort_index()


def select_adaptive_inds(theta, nfunc, nfam=None):
    """Returns boolian mask of theta components which have the input N (and
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
    df_dict = {}
    for prob_key, prob_data in results_dict.items():
        adfam = prob_key[0] in ['adfam_gg_ta_1d', 'nn_adl']
        nfunc_list = bsr.results_utils.nfunc_list_union(prob_data)
        if any(isinstance(nf, list) for nf in nfunc_list):
            assert all(isinstance(nf, list) for nf in nfunc_list), nfunc_list
            nfunc_list = [nf[-1] for nf in nfunc_list]
        prob_str = bsr.results_utils.root_given_key(prob_key)
        prob_df_dict = {}
        for meth_key, meth_data in prob_data.items():
            meth_str = bsr.results_utils.root_given_key(meth_key)
            save_name = 'cache/results_df_{}_{}_{}sim'.format(
                prob_str, meth_str, n_simulate)
            prob_df_dict[meth_str] = get_method_df(
                meth_data, adaptive=meth_key[0], save_name=save_name,
                n_simulate=n_simulate, nfunc_list=nfunc_list,
                adfam=adfam, **kwargs)
        # Add efficiency gains
        vanilla_keys = [key for key in prob_data.keys() if not key[0]]
        if len(vanilla_keys) == 1:
            van_str = bsr.results_utils.root_given_key(vanilla_keys[0])
            van_nsamp = prob_df_dict[van_str]['nsample'].loc[
                ('combined', 'value')]
            for measure in ['bs std', 'sep std']:
                van_val = prob_df_dict[van_str].loc[(measure, 'value')]
                van_unc = prob_df_dict[van_str].loc[(measure, 'uncertainty')]
                gain_type = measure.replace('std', 'gain')
                for meth_str in prob_df_dict:
                    meth_nsamp = prob_df_dict[meth_str]['nsample'].loc[
                        ('combined', 'value')]
                    meth_val = prob_df_dict[meth_str].loc[(measure, 'value')]
                    meth_unc = prob_df_dict[meth_str].loc[
                        (measure, 'uncertainty')]
                    nsamp_ratio = van_nsamp / meth_nsamp
                    gain, gain_unc = nestcheck.pandas_functions.get_eff_gain(
                        van_val, van_unc, meth_val, meth_unc,
                        adjust=nsamp_ratio)
                    prob_df_dict[meth_str].loc[(gain_type, 'value'), :] = gain
                    prob_df_dict[meth_str].loc[
                        (gain_type, 'uncertainty'), :] = gain_unc
        else:
            print('not adding gains as no single vanilla method',
                  prob_data.keys())
        df_dict[prob_str] = pd.concat(prob_df_dict)
    return pd.concat(df_dict)


@nestcheck.io_utils.save_load_result
def get_method_df(meth_data, **kwargs):
    """Results for a given method."""
    df = get_bayes_df(meth_data['run_list'], meth_data['run_list_sep'],
                      **kwargs)
    df['nsample'] = np.full(df.shape[0], np.nan)
    df['nlike'] = np.full(df.shape[0], np.nan)
    df.loc[('combined', 'value'), 'nsample'] = sum(
        [run['logl'].shape[0] for run in meth_data['run_list']])
    try:
        df.loc[('combined', 'value'), 'nlike'] = sum(
            [run['output']['nlike'] for run in meth_data['run_list']])
    except KeyError:
        pass
    return df
