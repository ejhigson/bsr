# Bayesian sparse reconstruction paper code

[![arXiv](http://img.shields.io/badge/arXiv-1704.03459-B31B1B.svg)](https://arxiv.org/abs/1704.03459)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ejhigson/dns/blob/master/LICENSE)

This repository contains the code used for making the results and plots in "Bayesian sparse reconstruction: a brute-force approach to astronomical imaging and machine learning" ([Higson et. al, 2018](https://arxiv.org/abs/1704.03459)). If the code is useful for your research then please cite the paper - the BibTeX is:

```latex
@article{Higson2018bayesian,
author={Higson, Edward and Handley, Will and Hobson, Mike and Lasenby, Anthony},
title={Bayesian sparse reconstruction: a brute-force approach to astronomical imaging and machine learning},
journal={arXiv preprint arXiv:1704.03459},
url={https://arxiv.org/abs/1704.03459},
year={2018}}
```

If you have any questions then feel free to email e.higson@mrao.cam.ac.uk. However, note that is research code and is not actively maintained.

# Requirements

Generating the results in the paper requires ``PolyChord`` v1.15, plus the requirements listed in ``setup.py``. For the paper we used Python 3.6.

# Set up

The code for computation using Python likelihoods and for all data processing and plotting is contained in the ``bsr`` Python module. This can be installed, along with its dependencies, by running the following command from within this repo:

```
pip install . --user
```

You can check your installation is working using the test suite (this requires `nose`) by running (again from within this repo):

```
nosetests
```

# Generating and plotting results

Nested sampling runs can be generated and the results plotted using ``compute_results.py``; see its documentation for more details. This also contains instructions for using the C++ version of the likelihood contained in ``CC_ini_likelihood.cpp``.

After nested sampling runs have been generated, results can also be plotted in the ``results_testing.ipynb`` notebook. ``paper_diagrams.ipynb`` contains the code for making some of the explanatory figures in the paper.