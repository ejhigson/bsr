#!/usr/bin/env python
"""Basis functions for fitting."""

import numpy as np
import scipy.special


def gg_1d(x, a, mu, sigma, beta):
    """1d generalised gaussian"""
    const = a * beta / (2 * sigma * scipy.special.gamma(1.0 / beta))
    return const * np.exp((-1.0) * (np.absolute(x - mu) / sigma) ** beta)


def gg_2d(self, x1, x2, a, mu1, mu2, sigma1, sigma2, beta1, beta2, omega):
    """2d generalised gaussian"""
    # Rotate gen gaussian around the mean
    assert omega < 0.25 * np.pi and omega > -0.25 * np.pi, \
        "Angle=" + str(omega) + "must be in range +-pi/4=" + str(np.pi / 4)
    x1_new = x1 - mu1
    x2_new = x2 - mu2
    x1_new = np.cos(omega) * x1_new - np.sin(omega) * x2_new
    x2_new = np.sin(omega) * x1_new + np.cos(omega) * x2_new
    # NB we do not include means as x1_new and x2_new are relative to means
    return (a * self.GG1d(x1_new, 1.0, 0, sigma1, beta1)
            * self.GG1d(x2_new, 1.0, 0, sigma2, beta2))


def nn_1d(x, a, w_0, w_1):
    """1d neural network tanh."""
    return a * np.tanh(w_0 + (w_1 * x))


def nn_2d(x1, x2, a, w_0, w_1, w_2):
    """2d neural network tanh."""
    return a * np.tanh((w_1 * x1) + (w_2 * x2) + w_0)
