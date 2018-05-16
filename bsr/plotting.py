#!/usr/bin/env python
"""Functions for plotting the results."""
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import fgivenx
import fgivenx.plot


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
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    y, pmf = fgivenx.compute_pmf(
        func, x, thetas, logZ=logzs, weights=weights, parallel=parallel,
        ntrim=None, ny=ny, y=y, cache=cache, tqdm_kwargs=tqdm_kwargs)
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
    gs = gridspec.GridSpec(
        nrow, ncol + 1,
        width_ratios=[colorbar_aspect] * ncol + [1],
        height_ratios=[1] * nrow)
    for i, y in enumerate(y_list):
        col = i % ncol
        row = i // ncol
        ax = plt.subplot(gs[row, col])
        im = ax.pcolor(x1, x2, y, vmin=0, vmax=1, linewidth=0, rasterized=True)
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
