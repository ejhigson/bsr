#!/usr/bin/env python
"""Generate data for fitting."""
import copy
import numpy as np
from PIL import Image
import bsr.basis_functions as bf


def generate_data(data_func, data_type, y_error_sigma, x_error_sigma=None,
                  **kwargs):
    """Get data dictionary."""
    npoints = kwargs.pop('npoints', None)
    x1min = kwargs.pop('x1min', 0.0)
    x1max = kwargs.pop('x1max', 1.0)
    x2min = kwargs.pop('x2min', 0.0)
    x2max = kwargs.pop('x2max', 1.0)
    seed = kwargs.pop('seed', 0)
    file_dir = kwargs.pop('file_dir', 'images/')
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    assert callable(data_func)
    if x_error_sigma == 0:
        x_error_sigma = None  # prevents needlessly doing x error integrals
    state = np.random.get_state()  # Save random state before seeding
    np.random.seed(seed)
    data = {}
    data['random_seed'] = seed
    data['x1min'] = x1min
    data['x1max'] = x1max
    data['func'] = data_func
    if data_func.__name__[-2:] == '1d':
        if npoints is None:
            npoints = 100
        data['x1'] = (np.random.random(npoints) * (x1max - x1min)) + x1min
        data['x2'] = None
        data['x2min'] = None
        data['x2max'] = None
        data['x_error_sigma'] = x_error_sigma
    elif data_func.__name__[-2:] == '2d' or data_func.__name__ == 'get_image':
        if npoints is None:
            npoints = 32
        data['x1'], data['x2'] = make_grid(
            npoints, x2_points=npoints, x1min=x1min, x2min=x2min,
            x1max=x1max, x2max=x2max)
        data['x2min'] = x2min
        data['x2max'] = x2max
        assert x_error_sigma is None
        data['x_error_sigma'] = None  # always None in 2d
    if data_func.__name__ == 'get_image':
        data['y'] = get_image(data_type, npoints, file_dir=file_dir)
        data['args'] = None
        data['data_type'] = data_type
    else:
        data_func_args = get_data_args(data_func, data_type)
        data['args'] = copy.deepcopy(data_func_args)
        data['data_type'] = data_type
        data['y'] = bf.sum_basis_funcs(
            data_func, data_func_args, data_type, data['x1'],
            x2=data['x2'], adaptive=False)
    data['y_error_sigma'] = y_error_sigma
    data['data_name'] = get_data_name(
        data_func, data_type, npoints, y_error_sigma, x_error_sigma)
    # Add Noise
    # ---------
    data['y_no_noise'] = copy.deepcopy(data['y'])
    data['y'] += data['y_error_sigma'] * np.random.normal(size=data['y'].shape)
    if data['x_error_sigma'] is not None:
        data['x1_no_noise'] = copy.deepcopy(data['x1'])
        data['x1'] += (data['x_error_sigma'] *
                       np.random.normal(size=data['x1'].shape))
    else:
        data['x1_no_noise'] = copy.deepcopy(data['x1'])
    np.random.set_state(state)  # Reset random state
    return data


def make_grid(x1_points, **kwargs):
    """Returns grid of x1 and x2 coordinates"""
    x2_points = kwargs.pop('x2_points', x1_points)
    x1min = kwargs.pop('x1min', 0.0)
    x1max = kwargs.pop('x1max', 1.0)
    x2min = kwargs.pop('x2min', 0.0)
    x2max = kwargs.pop('x2max', 1.0)
    if kwargs:
        raise TypeError('Unexpected **kwargs: {0}'.format(kwargs))
    x1_setup = np.linspace(x1min, x1max, num=x1_points)
    # flip x2 order to have y increacing on plots' verticle axis
    x2_setup = np.linspace(x2min, x2max, num=x2_points)[::-1]
    x1_grid, x2_grid = np.meshgrid(x1_setup, x2_setup)
    return x1_grid, x2_grid


def get_image(data_type, side_size, file_dir='images/'):
    """Load image from file into array format."""
    assert isinstance(data_type, int)
    filename = 'xdf_crop_{}.jpeg'.format(data_type)
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
    return pixels


def get_data_name(data_func, data_type, npoints, y_error_sigma, x_error_sigma):
    """Standard string describing data for save names."""
    data_name = '{}_{}'.format(data_func.__name__, data_type)
    if data_func.__name__ != 'get_image':
        data_name += 'funcs'
    data_name += '_{}pts_{}ye'.format(npoints, y_error_sigma)
    if x_error_sigma is not None:
        data_name += '_{}xe'.format(x_error_sigma)
    return data_name.replace('.', '_')


# Set up data
# -----------
def get_data_args(data_func, nfuncs):
    """Returns default arguments for generating data."""
    if data_func.__name__ == 'gg_1d' and nfuncs in [1, 2, 3]:
        # first arg is sorted
        if nfuncs == 1:
            data_args = [{'a': 0.75, 'mu': 0.4, 'sigma': 0.3, 'beta': 2.0}]
        elif nfuncs == 2:
            data_args = [{'a': 0.2, 'mu': 0.4, 'sigma': 0.6, 'beta': 5.0},
                         {'a': 0.55, 'mu': 0.4, 'sigma': 0.2, 'beta': 4.0}]
        elif nfuncs == 3:
            data_args = [{'a': 0.2, 'mu': 0.4, 'sigma': 0.6, 'beta': 5.0},
                         {'a': 0.35, 'mu': 0.6, 'sigma': 0.07, 'beta': 2.0},
                         {'a': 0.55, 'mu': 0.32, 'sigma': 0.14, 'beta': 6.0}]
    elif data_func.__name__ == 'ta_1d' and nfuncs in [1, 2, 3]:
        # first arg is sorted
        if nfuncs == 1:
            data_args = [{'a': 0.6, 'w_0': 0, 'w_1': 3}]
        elif nfuncs == 2:
            data_args = [{'a': 0.7, 'w_0': -1, 'w_1': 3},
                         {'a': 0.9, 'w_0': 2, 'w_1': -3}]
        elif nfuncs == 3:
            data_args = [
                {'a': 0.6, 'w_0': -7, 'w_1': 8},
                {'a': 1, 'w_0': -1, 'w_1': 3},
                {'a': 1.4, 'w_0': 2, 'w_1': -3}]
    elif data_func.__name__ == 'gg_2d' and nfuncs in [1, 2, 3]:
        # the order is (with first arg sorted):
        # [a_1, mu1_1, mu2_1, s1_1, s2_1, b1_1, b2_1, rot angle]
        if nfuncs == 1:
            data_args = [{'a': 0.8,
                          'mu1': 0.6, 'mu2': 0.6,
                          # 'sigma1': 0.1, 'sigma2': 0.1,
                          # 'beta1': 4, 'beta2': 0.5,
                          'sigma1': 0.1, 'sigma2': 0.2,
                          'beta1': 2, 'beta2': 2,
                          'omega': 0.1 * np.pi}]
        elif nfuncs == 2:
            data_args = [{'a': 0.5,
                          'mu1': 0.5, 'mu2': 0.4,
                          'sigma1': 0.4, 'sigma2': 0.2,
                          'beta1': 2, 'beta2': 2,
                          'omega': 0},
                         {'a': 0.8,
                          'mu1': 0.5, 'mu2': 0.6,
                          'sigma1': 0.1, 'sigma2': 0.1,
                          'beta1': 2, 'beta2': 2,
                          'omega': 0}]
        elif nfuncs == 3:
            data_args = [{'a': 0.5,
                          'mu1': 0.3, 'mu2': 0.7,
                          'sigma1': 0.2, 'sigma2': 0.2,
                          'beta1': 2, 'beta2': 2,
                          'omega': 0},
                         {'a': 0.7,
                          'mu1': 0.7, 'mu2': 0.6,
                          'sigma1': 0.15, 'sigma2': 0.15,
                          'beta1': 2, 'beta2': 2,
                          'omega': 0},
                         {'a': 0.9,
                          'mu1': 0.4, 'mu2': 0.3,
                          'sigma1': 0.1, 'sigma2': 0.1,
                          'beta1': 2, 'beta2': 2,
                          'omega': 0}]
    else:
        raise AssertionError('no data args found! func={} nfuncs={}'.format(
            data_func.__name__, nfuncs))
    data_args_list = []
    for name in bf.get_bf_param_names(data_func):
        data_args_list += [d[name] for d in data_args]
    # if data_func.__name__[:2] == 'ta':
    #     data_args_list.append(const)
    return data_args_list
