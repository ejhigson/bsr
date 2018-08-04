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
import bsr.priors
import bsr.results_utils


def adaptive_logz(run, logw=None, nfunc=1):
    """Get the logz assigned to nfunc basis functions from an adaptive run.
    Note that the absolute value does not correspond to that from a similar
    vanilla run, but the relative value can be used when calculating Bayes
    factors."""
    if logw is None:
        logw = nestcheck.ns_run_utils.get_logw(run)
    vals = run['theta'][:, 0]
    if isinstance(nfunc, list):
        nfunc = nfunc[-1]
    points = logw[((nfunc - 0.5) <= vals) & (vals < (nfunc + 0.5))]
    if points.shape == (0,):
        return -np.inf
    else:
        return scipy.special.logsumexp(points)


def get_bayes_df(problem_data, **kwargs):
    """Dataframe of Bayes factors."""
    adfam = kwargs.pop('adfam', False)
    n_simulate = kwargs.pop('n_simulate', 5)
    if adfam:
        nfunc_list = list(range(1, 11))
    else:
        nfunc_list = bsr.results_utils.nfunc_list_union(problem_data)
    if any(isinstance(nf, list) for nf in nfunc_list):
        assert all(isinstance(nf, list) for nf in nfunc_list), nfunc_list
        nfunc_list = [nf[-1] for nf in nfunc_list]
    df_list = []
    for meth_key, meth_data in problem_data.items():
        adaptive = meth_key[0]
        run_list = meth_data['run_list']
        if not adaptive:
            # vanilla bayes
            bayes = np.asarray([nestcheck.estimators.logz(run) for run in
                                run_list])
            bayes -= bayes.max()
            # std calculation of different runs parallelised and uses simulated
            # weights method as it gives the correct results for logZ
            bayes_stds = nestcheck.parallel_utils.parallel_apply(
                nestcheck.error_analysis.run_std_simulate, run_list,
                func_args=([nestcheck.estimators.logz],),
                func_kwargs={'n_simulate': n_simulate})
            bayes_stds = np.squeeze(np.asarray(bayes_stds))
        else:
            # Adaptive bayes
            assert len(run_list) == 1
            if adfam:
                theta = run_list[0]['theta']
                # remove T column and add 5 to adaptive column when T=2
                theta[np.where(theta[:, 0] >= 1.5), 1] += 5
                theta = theta[:, 1:]
                run_list[0]['theta'] = theta
            funcs = [functools.partial(adaptive_logz, nfunc=nf) for nf in
                     nfunc_list]
            bayes = nestcheck.ns_run_utils.run_estimators(run_list[0], funcs)
            bayes -= bayes.max()
            bayes_stds = nestcheck.error_analysis.run_std_bootstrap(
                run_list[0], funcs, n_simulate=n_simulate)
        # Now put into df
        df_temp = pd.DataFrame(np.vstack([bayes, bayes_stds]),
                               index=['value', 'uncertainty'],
                               columns=nfunc_list)
        df_temp.index.name = 'result type'
        df_temp['method key'] = bsr.results_utils.root_given_key(meth_key)
        df_list.append(df_temp)
    df = pd.concat(df_list)
    df.set_index('method key', append=True, inplace=True)
    return df.reorder_levels(['method key', 'result type'])
