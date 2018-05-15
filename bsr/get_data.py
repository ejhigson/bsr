#!/usr/bin/env python
"""Generate data for fitting."""
import copy
import inspect
import numpy as np
from PIL import Image


def generate_data(data_func, data_type, y_error_sigma, x_error_sigma=None,
                  npoints=32, x1min=0, x1max=1, x2min=0, x2max=1):
    """Get data dictionary."""
    data = {}
    data['x1min'] = x1min
    data['x1max'] = x1max
    if isinstance(data_func, str):
        data['func_name'] = data_func
    else:
        data['func_name'] = data_func.__name__
    # data['data_name'] = utils.data_name(data_func, data_type, npoints,
    #                                     y_error_sigma, x_error_sigma)
    if data['func_name'][-2:] == '1d':
        data['x1'] = (np.random.random(npoints) * (x1max - x1min)) + x1min
        data['x2'] = None
        data['x_error_sigma'] = x_error_sigma
    elif data['func_name'][-2:] == '2d' or data['func_name'] == 'get_image':
        data['x1'], data['x2'] = make_grid(npoints, x2_points=npoints,
                                           x1min=x1min, x2min=x2min,
                                           x1max=x1max, x2max=x2max)
        data['x2min'] = x2min
        data['x2max'] = x2max
        data['x_error_sigma'] = None  # always None in 2d
    if data['func_name'] == 'get_image':
        data['y'], _, _ = get_image(data_type, npoints)
        data['image_file'] = data_type
    else:
        data_func_args = get_data_args(data_func, data_type)
        data['args'] = data_func_args
        data['nfuncs'] = data_type
        data['y'] = 0
        for i in range(data_type):
            if data['x2'] is None:
                data['y'] += data_func(data['x1'],
                                       *data_func_args[i::data_type])
            else:
                data['y'] += data_func(data['x1'], data['x2'],
                                       *data_func_args[i::data_type])
    data['y_error_sigma'] = y_error_sigma
    # fix the random seed
    data['random_seed'] = 0
    np.random.seed(data['random_seed'])
    # add noise
    data['y_no_noise'] = copy.deepcopy(data['y'])
    data['y'] += data['y_error_sigma'] * np.random.normal(size=data['y'].shape)
    if data['x_error_sigma'] is not None:
        data['x1_no_noise'] = copy.deepcopy(data['x1'])
        data['x1'] += (data['x_error_sigma'] *
                       np.random.normal(size=data['x1'].shape))
    return data


def make_grid(x1_points, x2_points=None, x1min=0.0, x2min=0.0, x1max=1.0,
              x2max=1.0):
    """Returns grid of x1 and x2 coordinates"""
    if x2_points is None:
        x2_points = x1_points
    x1_setup = np.linspace(x1min, x1max, num=x1_points)
    # flip x2 order to have y increacing on plots' verticle axis
    x2_setup = np.linspace(x2min, x2max, num=x2_points)[::-1]
    x1_grid, x2_grid = np.meshgrid(x1_setup, x2_setup)
    return x1_grid, x2_grid


def get_image(filename, side_size, file_dir='images/'):
    image_filename = filename + '_' + str(side_size) + 'x' + str(side_size)
    # open image and resize it
    size = (side_size, side_size)
    im_fullsize = Image.open(file_dir + filename)
    # convert to greyscale
    im_fullsize = im_fullsize.convert('L')
    pixels_fullsize = np.zeros(im_fullsize.size)
    for (x, y), _ in np.ndenumerate(pixels_fullsize):
        pixels_fullsize[x, y] = im_fullsize.getpixel((x, y))
    pixels_fullsize *= 1.0 / 256
    im = im_fullsize.resize(size, Image.ANTIALIAS)
    pixels = np.zeros(size)
    for (x, y), _ in np.ndenumerate(pixels):
        pixels[x, y] = im.getpixel((x, y))
    pixels *= 1.0 / 256
    return pixels, pixels_fullsize, image_filename


# Set up data
# -----------
def get_data_args(data_func, nfuncs):
    assert data_func.__name__ in ['gg_1d', 'gg_2d'], (
        'no data args found! func={} nfuncs={}'.format(
            data_func.__name__, nfuncs))
    if data_func.__name__ == 'gg_1d':
        # the order is (with first arg sorted):
        if nfuncs == 1:
            data_args = [{'a': 0.3, 'mu': 0.4, 'sigma': 0.2, 'beta': 2.0}]
        elif nfuncs == 2:
            data_args = [{'a': 0.15, 'mu': 0.75, 'sigma': 0.1, 'beta': 2.0},
                         {'a': 0.30, 'mu': 0.35, 'sigma': 0.2, 'beta': 4.0}]
    elif data_func.__name__ == 'gg_2d':
        # the order is (with first arg sorted):
        # [a_1, mu1_1, mu2_1, s1_1, s2_1, b1_1, b2_1, rot angle]
        if nfuncs == 1:
            # data_args = [0.1, 0.7, 0.6, 0.1, 0.1, 4, .5, 0.15 * np.pi]
            data_args = [{'a': 0.1,
                          'mu1': 0.7, 'mu2': 0.6,
                          'sigma1': 0.1, 'sigma2': 0.1,
                          'beta1': 4, 'beta2': 0.5,
                          'omega': 0.15 * np.pi}]
        elif nfuncs == 2:
            # data_args = [0.1, 0.3, 0.35, 0.25, 0.85, 0.0, 0.1, 0.3, 0.5, 0.3,
            #              4.0, 2, 4, 2, np.pi * 0.1, 0.0]
            data_args = [{'a': 0.1,
                          'mu1': 0.35, 'mu2': 0.85,
                          'sigma1': 0.1, 'sigma2': 0.5,
                          'beta1': 4, 'beta2': 4,
                          'omega': 0.1 * np.pi},
                         {'a': 0.3,
                          'mu1': 0.25, 'mu2': 0.0,
                          'sigma1': 0.3, 'sigma2': 0.3,
                          'beta1': 2, 'beta2': 2,
                          'omega': 0}]
    data_args_list = []
    for name in inspect.signature(data_func).parameters:
        if name not in ['x', 'x1', 'x2']:
            data_args_list += [d[name] for d in data_args]
    return data_args_list
