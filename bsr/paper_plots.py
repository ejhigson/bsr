#!/usr/bin/env python
"""Utility functions for making plots in the paper, including saved settings."""
import matplotlib
import bsr.plotting

# Set plot size, font and fontsize to match LaTeX template
# --------------------------------------------------------
# NB A4 paper is 8.27 × 11.69 inches (=210 × 297 mm)
# Font: \T1/ntxtlf/m/n/9
# Caption font: \T1/ntxtlf/m/n/8
# Footnote font: \T1/ntxtlf/m/n/8
# Abstract font: \T1/ntxtlf/m/n/10
TEXTWIDTH = 6.97522 * 0.99  # make 1% smaller to ensure everything fits
TEXTHEIGHT = 9.43869 * 0.99  # make 1% smaller to ensure everything fits
COLWIDTH = 3.32153
DPI = 400  # needed for colorplots (had to rasterize to remove white lines)
MATPLOTLIB_SETTINGS = {'text.usetex': True,
                       'font.family': ['serif'],
                       'font.serif': ['Times New Roman'],
                       'font.size': 8}


def check_matplotlib_settings():
    """Print warning if settings are not as expected."""
    for key, value in MATPLOTLIB_SETTINGS.items():
        if matplotlib.rcParams.get(key) != value:
            print('{}={} - the paper plots use {}'.format(
                key, matplotlib.rcParams.get(key), value))


def odds(results_df, **kwargs):
    """Make a bar chart of the odds."""
    nruns = kwargs.pop('nruns', None)
    max_unc = kwargs.pop('max_unc', True)
    check_matplotlib_settings()
    bayes_fig_list = []
    for prob_key in set(results_df.index.get_level_values('problem key')):
        # Get cols
        cols = [col for col in results_df.columns if 'P(' in col]
        cols = [col for col in cols if 'N=' in col]
        if 'nn_adl' in prob_key:
            exclude_n_le = 3
        else:
            exclude_n_le = 0
        for i in range(1, exclude_n_le + 1):
            cols = [col for col in cols if 'N={})'.format(i) not in col]
        df_temp = results_df.xs(prob_key, level='problem key')[cols]
        df_temp = df_temp.dropna(axis=1, how='all')
        adfam = ('adfam' in prob_key) or ('nn_adl' in prob_key)
        figwidth = TEXTWIDTH * 0.4
        if 'nn_adl' in prob_key:
            figwidth = 3.05
        fig = bsr.plotting.plot_bars(
            df_temp, figsize=(figwidth, 1.7),
            adfam=adfam, nn_xlabel=('nn' in prob_key), max_unc=max_unc,
            exclude_n_le=exclude_n_le, **kwargs)
        filename = 'plots/odds_{}'.format(prob_key)
        if nruns is not None:
            filename += '_{}runs'.format(nruns)
        if not max_unc:
            filename += '_pnsunc'
        filename += '.pdf'
        print('saving to', filename)
        fig.savefig(filename)
        bayes_fig_list.append(fig)
    return bayes_fig_list


def multi(results_dict, **kwargs):
    """Multi plot with default settings."""
    prob_condition = kwargs.pop('prob_condition', lambda x: True)
    meth_condition = kwargs.pop('meth_condition', lambda x: True)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    check_matplotlib_settings()
    multi_fig_list = []
    for prob_key, problem_data in results_dict.items():
        if prob_condition(prob_key):
            nfunc_list = bsr.results_utils.nfunc_list_union(problem_data)
            adfam = prob_key[0] in ['adfam_gg_ta_1d', 'nn_adl']
            # Use true signal title unless data is from astro image
            true_signal = 'image' not in prob_key[1]
            for meth_key, meth_data in problem_data.items():
                if meth_condition(meth_key):
                    print(prob_key, meth_key)
                    fig = bsr.plotting.plot_runs(
                        meth_data['likelihood_list'], meth_data['run_list'],
                        nfunc_list, combine=True, plot_data=True, adfam=adfam,
                        ntrim=500, figsize=(TEXTWIDTH * 0.6, 1.7),
                        true_signal=true_signal)
                    filename = 'plots/multi_{}_{}runs.pdf'.format(
                        bsr.results_utils.root_given_key(prob_key + meth_key),
                        len(meth_data['run_list_sep']))
                    print('saving to', filename)
                    fig.savefig(filename, dpi=DPI)
                    multi_fig_list.append(fig)
    return multi_fig_list


def split(results_dict, **kwargs):
    """Multi plot with default settings."""
    prob_condition = kwargs.pop('prob_condition', lambda x: True)
    meth_condition = kwargs.pop('meth_condition', lambda x: True)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    check_matplotlib_settings()
    split_fig_list = []
    for prob_key, problem_data in results_dict.items():
        if prob_condition(prob_key):
            adfam = prob_key[0] in ['adfam_gg_ta_1d', 'nn_adl']
            adfam_nn = prob_key[0] == 'nn_adl'
            nfunc_list = bsr.results_utils.nfunc_list_union(problem_data)
            nrow = 1
            figheight = 1.45
            # Max 5 plots per row
            if not adfam:
                nrow = (len(nfunc_list) + 4) // 5
            else:
                nrow = ((len(nfunc_list) * 2) + 4) // 5
            figheight += (nrow - 1) * 1.35  # extra space for extra rows
            figwidth = TEXTWIDTH * 0.9
            if '1d' in prob_key[0]:
                # Add extra space for color bar and numerical scale
                figheight += 0.2
                figwidth += 0.2
            for meth_key, meth_data in problem_data.items():
                if meth_condition(meth_key):
                    print(prob_key, meth_key)
                    fig = bsr.plotting.plot_runs(
                        meth_data['likelihood_list'], meth_data['run_list'],
                        nfunc_list, combine=False, adfam=adfam,
                        plot_data=False, ntrim=500, nrow=nrow,
                        adfam_nn=adfam_nn,
                        figsize=(figwidth, figheight))
                    filename = 'plots/split_{}_{}runs.pdf'.format(
                        bsr.results_utils.root_given_key(prob_key + meth_key),
                        len(meth_data['run_list_sep']))
                    print('saving to', filename)
                    fig.savefig(filename, dpi=DPI)
                    split_fig_list.append(fig)
    return split_fig_list
