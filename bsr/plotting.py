#!/usr/bin/env python
"""Functions for plotting the results."""
import warnings
import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import fgivenx
import fgivenx.plot
import nestcheck.estimators
import nestcheck.ns_run_utils
import bsr.priors


def plot_runs(likelihood_list, run_list, **kwargs):
    """Wrapper for plotting ns runs (automatically tests if they are
    1d or 2d)."""
    if not isinstance(likelihood_list, list):
        likelihood_list = [likelihood_list]
    if not isinstance(run_list, list):
        run_list = [run_list]
    assert len(run_list) == len(likelihood_list)
    assert likelihood_list[0].basis_func.__name__[-2:] in ['1d', '2d']
    if len(likelihood_list) >= 1:
        for like in likelihood_list[1:]:
            assert like.data == likelihood_list[0].data
    kwargs['data'] = likelihood_list[0].data
    if 'titles' not in kwargs:
        kwargs['titles'] = get_default_titles(
            likelihood_list, kwargs.get('plot_data', False),
            kwargs.get('combine', True))
    if likelihood_list[0].basis_func.__name__[-2:] == '1d':
        fig = plot_1d_runs(likelihood_list, run_list, **kwargs)
    elif likelihood_list[0].basis_func.__name__[-2:] == '2d':
        fig = plot_2d_runs(likelihood_list, run_list, **kwargs)
    return fig


def get_default_titles(likelihood_list, combine, plot_data):
    """Get some default titles for the plots."""
    titles = []
    if plot_data:
        titles += ['true signal', 'noisy data']
    if combine:
        titles.append('fit')
    else:
        if len(likelihood_list) == 1 and likelihood_list[0].adaptive:
            # As we don't know which funcs are being plotted, just assume they
            # start with B=1 and add a longer list than we actuall need
            titles += ['$B={}$'.format(i) for i in range(1, 12)]
        else:
            titles += ['$B={}$'.format(like.nfunc) for like in
                       likelihood_list]
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
    combine = kwargs.pop('combine', False)
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
                likelihood_list[0].fit_fgivenx, run_list[0], **kwargs)
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
    gs.update(wspace=0.1, hspace=0.2)
    return fig


def plot_1d_adaptive_multi(func, run, nfunc_list=None, **kwargs):
    samp_nfuncs = np.round(run['theta'][:, 0]).astype(int)
    if nfunc_list is None:
        nfunc_list = list(np.unique(samp_nfuncs))
    funcs = [func] * len(nfunc_list)
    samples = []
    weights = []
    logw = nestcheck.ns_run_utils.get_logw(run)
    for nf in nfunc_list:
        inds = np.where(samp_nfuncs == nf)[0]
        samples.append(run['theta'][inds, :])
        logw_temp = logw[inds]
        weights.append(np.exp(logw_temp - logw_temp.max()))
    return plot_1d_grid(funcs, samples, weights, **kwargs)


def plot_1d_grid(funcs, samples, weights, **kwargs):
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
    # ncolorbar = kwargs.pop('ncolorbar', 1)
    # Make figure
    gs = gridspec.GridSpec(
        nrow, ncol + 1,
        width_ratios=[colorbar_aspect] * ncol + [1],
        height_ratios=[1] * nrow)
    fig = kwargs.pop('fig', None)
    if fig is None:
        fig = plt.figure(figsize=figsize)
        for i in range(nplots):
            col = i % ncol
            row = i // ncol
            ax = plt.subplot(gs[row, col])
    for i in range(nplots):
        ax = fig.axes[i]
        col = i % ncol
        row = i // ncol
        # If plot_data we also want to plot the true function and the
        # noisy data, and need to shift the other plots along
        if not plot_data:
            cbar = fgivenx_plot(
                funcs[i], x, samples[i], ax, weights=weights[i],
                logzs=logzs[i], y=x, **kwargs)
        elif i >= 2:
            print(i, len(funcs))
            cbar = fgivenx_plot(
                funcs[i - 2], x, samples[i - 2], ax, weights=weights[i - 2],
                logzs=logzs[i - 2], y=x, **kwargs)
        elif i == 0:
            y_true = np.zeros(x.shape)
            for nf in range(data['nfuncs']):
                comp = data['func'](x, *data['args'][nf::data['nfuncs']])
                if data['nfuncs'] != 1:
                    ax.plot(x, comp, color=data_color, linestyle=':')
                y_true += comp
            ax.plot(x, y_true, color=data_color)
        elif i == 1:
            ax.errorbar(data['x1'], data['y'], yerr=data['y_error_sigma'],
                        xerr=data['x_error_sigma'], fmt='none',
                        ecolor=data_color)
        if col == 0:
            ax.set_ylabel('$y$')
        if (data is None and col == 0) or (data is not None and col == 2):
            cbar_plot = plt.colorbar(cbar, cax=plt.subplot(gs[row, -1]),
                                     ticks=[1, 2, 3])
            cbar_plot.solids.set_edgecolor('face')
            cbar_plot.ax.set_yticklabels(
                [r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])
            # cbar_plot.ax.set(aspect=colorbar_aspect)
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
        if i != 0:
            ax.set_yticklabels([])
    gs.update(wspace=0.1, hspace=0.2)
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


def plot_prior(likelihood, nsamples):
    """Plots the default prior for the likelihood object's
    parameters."""
    prior = bsr.priors.get_default_prior(
        likelihood.basis_func, likelihood.nfunc, likelihood.adaptive)
    hypercube_samps = np.random.random((nsamples, likelihood.ndim))
    thetas = np.apply_along_axis(prior, 1, hypercube_samps)
    assert likelihood.basis_func.__name__[-2:] in ['1d', '2d']
    if likelihood.basis_func.__name__[-2:] == '1d':
        fig = plot_1d_grid(likelihood.fit_fgivenx, thetas, None)
    elif likelihood.basis_func.__name__[-2:] == '2d':
        y = likelihood.fit_mean(
            thetas, likelihood.data['x1'], likelihood.data['x2'])
        print('ymax={} ymean={}'.format(y.max(), np.mean(y)))
        fig = plot_colormap(y, likelihood.data['x1'], likelihood.data['x2'])
    return fig