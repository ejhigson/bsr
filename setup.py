#!/usr/bin/env python
"""bsr setup module."""
import setuptools


setuptools.setup(name='bsr',
                 version='0.0.0',
                 description=(
                     'Code for the Bayesian sparse reconstruction paper.'),
                 url='https://github.com/ejhigson/bsr',
                 author='Edward Higson',
                 author_email='e.higson@mrao.cam.ac.uk',
                 license='MIT',
                 packages=['bsr'],
                 install_requires=['numpy>=1.13',
                                   'scipy>=1.0.0',
                                   'matplotlib',
                                   'mpi4py',
                                   'Pillow',
                                   'nestcheck>=0.1.8',
                                   'dyPolyChord>=0.0.3'])
