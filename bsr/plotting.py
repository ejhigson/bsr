#!/usr/bin/env python
"""Functions for plotting the results."""
import copy
import warnings
import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import fgivenx
import fgivenx.plot
import nestcheck.estimators
import nestcheck.ns_run_utils
import nestcheck.error_analysis
import nestcheck.parallel_utils
import bsr.priors
import bsr.basis_functions as bf
import bsr.results_utils
import bsr.results_tables


def plot_bayes(df, **kwargs):
    """Make a bar chart of vanilla and adaptive Bayes factors, including their
    error bars."""
    assert len(df.index.names) == 2
    assert df.index.names[-1] == 'result type', df.index.names
    adfam = kwargs.pop('adfam', False)
    method_list = kwargs.pop(
        'method_list',
        sort_method_list(list(set(df.index.get_level_values(0)))))
    title = kwargs.pop('title', 'log posterior odds ratios')
    ymin = kwargs.pop('ymin', -10)
    figsize = kwargs.pop('figsize', (3, 2))
    colors = kwargs.pop('colors', ['lightgrey', 'grey', 'black', 'darkblue',
                                   'darkred'])
    nn_xlabel = kwargs.pop('nn_xlabel', False)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    # Make the plot
    tot_width = 0.75  # the total width of all the bars
    bar_width = tot_width / len(method_list)
    bar_centres = np.arange(len(method_list)) * bar_width
    bar_centres -= (tot_width - bar_width) * 0.5
    ind = np.arange(len(df.columns))  # the x locations for the groups
    bars = []
    labels = []
    fig, ax = plt.subplots(figsize=figsize)
    for i, method in enumerate(method_list):
        # get label
        label = ''
        if method.split('_')[0] == 'True':
            label += 'adaptive'
        else:
            assert method.split('_')[0] == 'False', method
            label += 'vanilla'
        if method.split('_')[1] != 'None':
            label += ' dg={}'.format(method.split('_')[1])
        labels.append(label)
        # plot bar
        bars.append(ax.bar(
            ind + bar_centres[i], df.loc[(method, 'value')], bar_width,
            yerr=df.loc[(method, 'uncertainty')], color=colors[i]))
    if title is not None:
        ax.set_title(title)
    ax.set_xticks(ind)
    if adfam:
        assert len(df.columns) % 2 == 0, len(df.columns)
        assert list(df.columns) == list(range(1, len(df.columns) + 1)), (
            df.columns)
        n_per_fam = len(df.columns) // 2
        xlabels = []
        for t in [1, 2]:
            for b in range(1, n_per_fam + 1):
                xlabels.append('{},{}'.format(t, b))
        fig.axes[0].set_xticklabels(xlabels)
        fig.axes[0].axvline(x=n_per_fam - 0.5, color='black', linestyle=':')
        ax.set_xlabel('family and number $T,B$')
    else:
        if nn_xlabel:
            ax.set_xlabel('nodes per hidden layer $B$')
        else:
            ax.set_xlabel('number of basis functions $B$')
        ax.set_xticklabels(['${}$'.format(nf) for nf in df.columns])
    if ymin == -10:
        ax.set_yticks([0, -5, -10])
    ax.legend([ba[0] for ba in bars], labels)
    ax.set_ylim([ymin, 0])
    adjust = {'top': 1 - (0.2 / figsize[1]),
              'bottom': 0.4 / figsize[1],
              'left': (0.3 / figsize[0]),
              'right': 1 - (0.01 / figsize[0])}
    fig.subplots_adjust(**adjust)
    return fig


def plot_runs(likelihood_list, run_list, nfunc_list=None, **kwargs):
    """Wrapper for plotting ns runs (automatically tests if they are
    1d or 2d)."""
    adfam = kwargs.get('adfam', False)
    if not isinstance(likelihood_list, list):
        likelihood_list = [likelihood_list]
    if not isinstance(run_list, list):
        run_list = [run_list]
    assert len(run_list) == len(likelihood_list), '{}!={}'.format(
        len(run_list), len(likelihood_list))
    if 'titles' not in kwargs:
        kwargs['titles'] = get_default_titles(
            likelihood_list, kwargs.get('plot_data', False),
            kwargs.get('combine', True), nfunc_list, adfam=adfam)
    if len(likelihood_list) >= 1:
        for like in likelihood_list[1:]:
            try:
                assert like.data == likelihood_list[0].data
            except ValueError:
                print('ValueError comparing data')
        kwargs['data'] = likelihood_list[0].data
    if kwargs['data']['y'].ndim == 1:
        fig = plot_1d_runs(likelihood_list, run_list, nfunc_list=nfunc_list,
                           **kwargs)
    elif kwargs['data']['y'].ndim == 2:
        kwargs.pop('ntrim', None)  # remove as no longer needed
        kwargs.pop('adfam', None)  # remove as no longer needed
        fig = plot_2d_runs(likelihood_list, run_list, **kwargs)
    return fig


def get_default_titles(likelihood_list, combine, plot_data,
                       nfunc_list, adfam=False):
    """Get some default titles for the plots."""
    titles = []
    if plot_data:
        titles += ['true signal', 'noisy data']
    if combine:
        titles.append('fit')
    else:
        if adfam:
            for nfunc in nfunc_list:
                titles.append('$T,B=1,{}$'.format(nfunc))
            for nfunc in nfunc_list:
                titles.append('$T,B=2,{}$'.format(nfunc))
        else:
            for nfunc in nfunc_list:
                if isinstance(nfunc, list):
                    # Display only the hidden layers' number of nodes
                    titles.append('$B={}$'.format(nfunc[-1]))
                else:
                    titles.append('$B={}$'.format(nfunc))
    return titles


def plot_2d_runs(likelihood_list, run_list, **kwargs):
    """Wrapper for plotting nested sampling runs with 2d data as colormaps."""
    data = kwargs.pop('data')
    combine = kwargs.pop('combine', False)
    plot_data = kwargs.pop('plot_data', False)
    y_list = []
    if plot_data:
        y_list.append(data['y_no_noise'])
        y_list.append(data['y'])
    if combine:
        logzs = np.asarray([nestcheck.estimators.logz(run) for run in
                            run_list])
        factors = np.exp(logzs - logzs.max())
        factors /= np.sum(factors)
        y_mean = np.zeros(data['x1'].shape)
        for i, run in enumerate(run_list):
            w_rel = nestcheck.ns_run_utils.get_w_rel(run)
            y_mean += factors[i] * likelihood_list[i].fit_mean(
                run['theta'], data['x1'], data['x2'], w_rel=w_rel)
        y_list.append(y_mean)
    else:
        if len(run_list) == 1 and likelihood_list[0].adaptive:
            run = run_list[0]
            samp_nfuncs = np.round(run['theta'][:, 0]).astype(int)
            nfunc_list = kwargs.pop('nfunc_list', list(np.unique(samp_nfuncs)))
            print(nfunc_list)
            logw = nestcheck.ns_run_utils.get_logw(run)
            for nf in nfunc_list:
                inds = np.where(samp_nfuncs == nf)[0]
                w_rel = logw[inds]
                # exp after selecting inds to avoid overflow
                w_rel = np.exp(w_rel - w_rel.max())
                y_list.append(likelihood_list[0].fit_mean(
                    run['theta'][inds, :], data['x1'],
                    data['x2'], w_rel=w_rel))
        else:
            for i, run in enumerate(run_list):
                w_rel = nestcheck.ns_run_utils.get_w_rel(run)
                y_list.append(likelihood_list[i].fit_mean(
                    run['theta'], data['x1'], data['x2'], w_rel=w_rel))
    print('ymax:', [y.max() for y in y_list])
    print('ymin:', [y.min() for y in y_list])
    return plot_colormap(y_list, data['x1'], data['x2'], **kwargs)


def plot_1d_runs(likelihood_list, run_list, **kwargs):
    """Get samples, weights and funcs then feed into plot_1d_grid."""
    combine = kwargs.pop('combine', False)
    nfunc_list = kwargs.pop('nfunc_list', None)
    funcs = [like.fit_fgivenx for like in likelihood_list]
    samples = [run['theta'] for run in run_list]
    weights = [nestcheck.ns_run_utils.get_w_rel(run) for run in run_list]
    if combine:
        if len(run_list) > 1:
            logzs = [nestcheck.estimators.logz(run) for run in run_list]
            fig = plot_1d_grid([funcs], [samples], [weights], logzs=[logzs],
                               **kwargs)
        else:
            fig = plot_1d_grid(funcs, samples, weights, **kwargs)
    else:
        if len(run_list) == 1 and likelihood_list[0].adaptive:
            fig = plot_1d_adaptive_multi(
                likelihood_list[0].fit_fgivenx, run_list[0],
                nfunc_list=nfunc_list, **kwargs)
        else:
            fig = plot_1d_grid(funcs, samples, weights, **kwargs)
    return fig


def plot_colormap(y_list, x1, x2, **kwargs):
    """Make 2d square colormaps.

    If y_list is not a list, it is plotted on a single colormap.
    """
    if not isinstance(y_list, list):
        y_list = [y_list]
    nrow = kwargs.pop('nrow', 1)
    ncol = int(np.ceil(len(y_list) / nrow))
    colorbar_aspect = kwargs.pop('colorbar_aspect', 40)
    titles = kwargs.pop('titles', None)
    figsize = kwargs.pop('figsize', (2.1 * ncol + 1, 2.4 * nrow))
    fig = plt.figure(figsize=figsize)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    gs = gridspec.GridSpec(
        nrow, ncol + 1,
        width_ratios=[colorbar_aspect] * ncol + [1],
        height_ratios=[1] * nrow)
    for i, y in enumerate(y_list):
        col = i % ncol
        row = i // ncol
        ax = plt.subplot(gs[row, col])
        im = ax.pcolor(x1, x2, y, vmin=0, vmax=1, linewidth=0,
                       rasterized=True)
        ax.set(aspect='equal')
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
        if titles is not None:
            ax.set_title(titles[i])
        if row == nrow - 1:
            ax.set_xlabel('$x_1$')
        if col == 0:
            ax.set_ylabel('$x_2$')
            cax = plt.subplot(gs[row, -1])
            plt.colorbar(im, cax=cax)
            cax.set(aspect=colorbar_aspect)
        if titles is not None:
            ax.set_title(titles[i])
    fig = adjust_spacing(fig, gs)
    return fig


def plot_1d_adaptive_multi(func, run, **kwargs):
    """Helper function for getting samples and weights for each number of
    basis functions from a 1d adaptive run."""
    nfunc_list = kwargs.pop('nfunc_list')
    adfam = kwargs.pop('adfam', False)
    if adfam:
        nfam_list = [1, 2]
    else:
        nfam_list = [None]
    inds_list = []
    for nfam in nfam_list:
        for nfunc in nfunc_list:
            inds_list.append(bsr.results_tables.select_adaptive_inds(
                run['theta'], nfunc, nfam=nfam))
    logw = nestcheck.ns_run_utils.get_logw(run)
    samples = []
    weights = []
    for inds in inds_list:
        samples.append(run['theta'][inds, :])
        logw_temp = logw[inds]
        weights.append(np.exp(logw_temp - logw_temp.max()))
    return plot_1d_grid([func] * len(samples), samples, weights, **kwargs)


def plot_1d_grid(funcs, samples, weights, **kwargs):  # pylint: disable=too-many-branches,too-many-statements
    """Plot a bunch of fgivenx subplots"""
    if not isinstance(funcs, list):
        funcs = [funcs]
    if not isinstance(samples, list):
        samples = [samples]
    if not isinstance(weights, list):
        weights = [weights]
    assert len(funcs) == len(samples) == len(weights)
    logzs = kwargs.pop('logzs', [None] * len(funcs))
    plot_data = kwargs.pop('plot_data', False)
    data = kwargs.pop('data', None)
    x = kwargs.pop('x', np.linspace(0.0, 1.0, 100))
    nplots = len(funcs)
    if plot_data:
        nplots += 2
    nrow = kwargs.pop('nrow', 1)
    ncol = int(np.ceil(nplots / nrow))
    titles = kwargs.pop('titles', None)
    figsize = kwargs.pop('figsize', (2.1 * ncol + 1, 2.4 * nrow))
    colorbar_aspect = kwargs.pop('colorbar_aspect', 40)
    data_color = kwargs.pop('data_color', 'darkred')
    kwargs.pop('adfam', None)  # remove as no longer needed
    # ncolorbar = kwargs.pop('ncolorbar', 1)
    # Make figure
    gs = gridspec.GridSpec(
        nrow, ncol + 1, width_ratios=[colorbar_aspect] * ncol + [1],
        height_ratios=[1] * nrow)
    fig = kwargs.pop('fig', None)
    if fig is None:
        fig = plt.figure(figsize=figsize)
        for i in range(nplots):
            col = i % ncol
            row = i // ncol
            ax = plt.subplot(gs[row, col])
    cbar_list = []  # for storing colorbars
    for i in range(nplots):
        ax = fig.axes[i]
        col = i % ncol
        row = i // ncol
        # If plot_data we also want to plot the true function and the
        # noisy data, and need to shift the other plots along
        if plot_data and i == 0:
            if data['data_type'] != 1:
                for nf in range(data['data_type']):
                    if data['func'].__name__[:2] != 'nn':
                        comp = data['func'](
                            x, *data['args'][nf::data['data_type']])
                        ax.plot(x, comp, color=data_color, linestyle=':')
            y_true = bf.sum_basis_funcs(
                data['func'], np.asarray(copy.deepcopy(data['args'])),
                data['data_type'], x)
            ax.plot(x, y_true, color=data_color)
        elif plot_data and i == 1:
            ax.errorbar(data['x1'], data['y'], yerr=data['y_error_sigma'],
                        xerr=data['x_error_sigma'], fmt='none',
                        ecolor=data_color)
        elif plot_data:  # for i >= 2
            cbar_list.append(fgivenx_plot(
                funcs[i - 2], x, samples[i - 2], ax, weights=weights[i - 2],
                logzs=logzs[i - 2], y=x, **kwargs))
            # # plot MAP
            # ax.plot(x, funcs[i - 2](x, samples[i - 2][-1, :]), color='black')
        else:
            cbar_list.append(fgivenx_plot(
                funcs[i], x, samples[i], ax, weights=weights[i],
                logzs=logzs[i], y=x, **kwargs))
            # # plot MAP
            # ax.plot(x, funcs[i](x, samples[i][-1, :]), color='black')
        if col == 0:
            ax.set_ylabel('$y$')
        # Format the axis and ticks
        # -------------------------
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([x.min(), x.max()])
        # ax.set(aspect='equal')
        prune = 'upper' if col != (ncol - 1) else None
        # prune = None
        ax.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(nbins=5, prune=prune))
        if titles is not None:
            ax.set_title(titles[i])
        if row == nrow - 1:
            ax.set_xlabel('$x$')
        else:
            ax.set_xticklabels([])
        if col != 0:
            ax.set_yticklabels([])
    for cbar in cbar_list:
        # If there are not enough samples in the plot, the colourbar will fail
        # with a RuntimeError. This can occur for undersampled models with the
        # adpative method. If it happens, just keep trying cbars until we reach
        # a plot with enough samples
        try:
            for row in range(nrow):
                cbar_plot = plt.colorbar(cbar, cax=plt.subplot(gs[row, -1]),
                                         ticks=[1, 2, 3])
                cbar_plot.solids.set_edgecolor('face')
                cbar_plot.ax.set_yticklabels(
                    [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])
            break
        except RuntimeError:
            pass
    fig = adjust_spacing(fig, gs)
    return fig


def adjust_spacing(fig, gs):
    """Adjust plotgrid position to make sure plots are square and use all
    the available space."""
    figsize = fig.get_size_inches()
    wspace = 0.1
    if gs is not None:
        wr = gs.get_width_ratios()
        hr = gs.get_height_ratios()
        gs.update(wspace=wspace)
        if len(hr) > 1:
            gs.update(hspace=0.5)
    else:
        wr = [1]
    margins = {'top': 0.2, 'bottom': 0.4, 'left': 0.45, 'right': 0.3}
    # fit squared vertically into space left after top and bottom margins
    adjust = {}
    adjust['bottom'] = margins['bottom'] / figsize[1]
    adjust['top'] = 1 - (margins['top'] / figsize[1])
    side_length = figsize[1] - (adjust['top'] - adjust['bottom'])
    width = ((sum(wr) / max(wr)) + (len(wr) - 1) * wspace) * side_length
    space = figsize[0] - (width + margins['left'] + margins['right'])
    if space < 0:
        warnings.warn('I need {} more horizonal inches to fit square plots'
                      .format(space), UserWarning)
        space = 0
    adjust['left'] = (margins['left'] + space / 2) / figsize[0]
    adjust['right'] = 1 - ((margins['right'] + space / 2) / figsize[0])
    fig.subplots_adjust(**adjust)
    return fig


def fgivenx_plot(func, x, thetas, ax, **kwargs):
    """
    Adds fgivenx plot to ax.

    Parameters
    ----------
    func: function or list of functions
        Model is y = func(x, theta)
    x: 1d numpy array
        Support
    ax: matplotlib axis object
    logzs: list of flots or None, optional
        For multimodal analysis, use a list of functions, thetas, weights
        and logZs. fgivenx checks each has the same length and weights samples
        from each model accordingly - see its docs for more details.
    weights: 1d numpy array or None
        Relative weights of each row of theta (if not evenly weighted).
    colormap: matplotlib colormap, optional
        Colors to plot fgivenx distribution.
    y: 1d numpy array, optional
        Specify actual y grid to use. If None, a grid is calculated based on
        the max and min y values and ny.
    ny: int, optional
        Size of y-axis grid for fgivenx plots. Not used if y is not None.
    cache: str or None
        Root for fgivenx caching (no caching if None).
    parallel: bool, optional
        fgivenx parallel option.
    rasterize_contours: bool, optional
        fgivenx rasterize_contours option.
    smooth: bool, optional
        fgivenx smooth option.
    tqdm_kwargs: dict, optional
        Keyword arguments to pass to the tqdm progress bar when it is used in
        fgivenx while plotting contours.
    ntrim: int or None, optional
        Max number of samples to trim to.

    Returns
    -------
    cbar: matplotlib colorbar
        For use in higher order functions.
    """
    logzs = kwargs.pop('logzs', None)
    weights = kwargs.pop('weights', None)
    colormap = kwargs.pop('colormap', plt.get_cmap('Reds_r'))
    y = kwargs.pop('y', None)
    ny = kwargs.pop('ny', x.shape[0])
    cache = kwargs.pop('cache', None)
    parallel = kwargs.pop('parallel', True)
    rasterize_contours = kwargs.pop('rasterize_contours', False)
    smooth = kwargs.pop('smooth', False)
    tqdm_kwargs = kwargs.pop('tqdm_kwargs', {'leave': False})
    ntrim = kwargs.pop('ntrim', None)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    try:
        y, pmf = fgivenx.compute_pmf(
            func, x, thetas, logZ=logzs, weights=weights, parallel=parallel,
            ntrim=ntrim, ny=ny, y=y, cache=cache, tqdm_kwargs=tqdm_kwargs)
        cbar = fgivenx.plot.plot(
            x, y, pmf, ax, colors=colormap, smooth=smooth,
            rasterize_contours=rasterize_contours)
        return cbar
    except ValueError:
        warnings.warn(
            ('ValueError in compute_pmf. Expected # samples={}'.format(
                np.sum(weights) / weights.max())), UserWarning)
        return None


def sort_method_list(unsorted):
    """Sort methods so vanilla runs come before adaptive runs and standard
    nested sampling comes before dynamic nested sampling."""
    out = []
    out += sorted([meth for meth in unsorted
                   if ('False' in meth and 'None' in meth)])
    out += sorted([meth for meth in unsorted
                   if ('False' in meth and 'None' not in meth)])
    out += sorted([meth for meth in unsorted
                   if ('True' in meth and 'None' in meth)])
    out += sorted([meth for meth in unsorted
    	           if ('True' in meth and 'None' not in meth)])
    assert set(out) == set(unsorted), [set(out), set(unsorted)]
    return out


def plot_prior(likelihood, nsamples):
    """Plots the default prior for the likelihood object's
    parameters."""
    prior = bsr.priors.get_default_prior(
        likelihood.function, likelihood.nfunc, adaptive=likelihood.adaptive)
    hypercube_samps = np.random.random((nsamples, likelihood.ndim))
    thetas = np.apply_along_axis(prior, 1, hypercube_samps)
    if likelihood.data['x2'] is None:
        fig = plot_1d_grid(likelihood.fit_fgivenx, thetas, None)
    else:
        y = likelihood.fit_mean(
            thetas, likelihood.data['x1'], likelihood.data['x2'])
        print('ymax={} ymean={}'.format(y.max(), np.mean(y)))
        fig = plot_colormap(y, likelihood.data['x1'], likelihood.data['x2'])
    return fig
