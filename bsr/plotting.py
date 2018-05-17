#!/usr/bin/env python
"""Functions for plotting the results."""
import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import fgivenx
import fgivenx.plot


def plot_1d(funcs, samples, weights, **kwargs):
    """Plot a bunch of fgivenx subplots"""
    if not isinstance(funcs, list):
        funcs = [funcs]
    if not isinstance(samples, list):
        samples = [samples]
    if not isinstance(weights, list):
        weights = [weights]
    assert len(funcs) == len(samples) == len(weights)
    logzs = kwargs.pop('logzs', [None] * len(funcs))
    data = kwargs.pop('data', None)
    x = kwargs.pop('x', np.linspace(0.0, 1.0, 100))
    nplots = len(funcs)
    if data is not None:
        nplots += 2
    nrow = kwargs.pop('nrow', 1)
    ncol = int(np.ceil(nplots / nrow))
    titles = kwargs.pop('titles', None)
    figsize = kwargs.pop('figsize', (2.1 * ncol + 1, 2.4 * nrow))
    colorbar_aspect = kwargs.pop('colorbar_aspect', 40)
    # Make figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        nrow, ncol + 1,
        width_ratios=[colorbar_aspect] * ncol + [1],
        height_ratios=[1] * nrow)
    for i in range(nplots):
        col = i % ncol
        row = i // ncol
        ax = plt.subplot(gs[row, col])
        if data is None:
            cbar = fgivenx_plot(
                funcs[i], x, samples[i], ax, weights=weights[i],
                logzs=logzs[i], y=x, **kwargs)
        elif i >= 2:
            cbar = fgivenx_plot(
                funcs[i - 2], x, samples[i - 2], ax, weights=weights[i - 2],
                logzs=logzs[i - 2], y=x, **kwargs)
        elif i == 0:
            y_true = np.zeros(x.shape)
            for nf in range(data['nfuncs']):
                y_true += data['func'](x, *data['args'][nf::data['nfuncs']])
            ax.plot(x, y_true, color='darkred')
        elif i == 1:
            ax.errorbar(data['x1'], data['y'], yerr=data['y_error_sigma'],
                        xerr=data['x_error_sigma'], fmt='none',
                        ecolor='darkred')
        if col == ncol - 1:
            ax.set_ylabel('$y$')
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
    y, pmf = fgivenx.compute_pmf(
        func, x, thetas, logZ=logzs, weights=weights, parallel=parallel,
        ntrim=ntrim, ny=ny, y=y, cache=cache, tqdm_kwargs=tqdm_kwargs)
    cbar = fgivenx.plot.plot(
        x, y, pmf, ax, colors=colormap, smooth=smooth,
        rasterize_contours=rasterize_contours)
    return cbar


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
    gs.update(wspace=0.1, hspace=0.2)
    return fig
