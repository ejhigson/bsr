#!/usr/bin/env python
"""Functions for plotting the results."""
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


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
