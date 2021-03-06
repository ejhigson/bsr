# Bayesian sparse reconstruction

[![DOI](http://img.shields.io/badge/DOI-10.1093/mnras/sty3307-darkblue.svg)](https://doi.org/10.1093/mnras/sty3307)
[![arXiv](http://img.shields.io/badge/arXiv-1809.04598-B31B1B.svg)](https://arxiv.org/abs/1809.04598)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ejhigson/dns/blob/master/LICENSE)

This repository contains the code used for making the results and plots in "Bayesian sparse reconstruction: a brute-force approach to astronomical imaging and machine learning" ([Higson et. al, 2019](http://dx.doi.org/10.1093/mnras/sty3307)).

If you have any questions then feel free to email <e.higson@mrao.cam.ac.uk>. However, note that this is research code and is not actively maintained.

### Requirements

Generating the results in the paper requires ``PolyChord`` v1.15, plus the requirements listed in ``setup.py``. Results in the paper were run using Python 3.6.

### Set up

The code for computation using Python likelihoods and for all data processing and plotting is contained in the ``bsr`` Python module. This can be installed, along with its dependencies, by running the following command from within this repo:

```
pip install . --user
```
You can check your installation is working using the test suite by running
```
nosetests
```
from within this repo. This requires `nose`.

### Generating and plotting results from the paper

Nested sampling runs can be generated and the results plotted using ``compute_results.py``; see its documentation for more details. This also contains instructions for using the C++ version of the likelihood contained in ``CC_ini_likelihood.cpp``.

After nested sampling runs have been generated, results can also be examined in more detail in the ``results_testing.ipynb`` notebook (this also creates the results tables and plots). ``paper_diagrams.ipynb`` contains the code for making some of the explanatory figures in the paper.

### Attribution

If the code is useful for your research then please cite the Bayesian sparse reconstruction paper. The BibTeX is:

```latex
@article{Higson2019bayesian,
author={Higson, Edward and Handley, Will and Hobson, Michael and Lasenby, Anthony},
title={Bayesian sparse reconstruction: a brute-force approach to astronomical imaging and machine learning},
journal={Monthly Notices of the Royal Astronomical Society},
volume={483},
number={4},
pages={4828-4846},
year={2019},
doi={10.1093/mnras/sty3307},
url={https://doi.org/10.1093/mnras/sty3307},
archivePrefix={arXiv},
arxivId={1809.04598}}
```

Note that some of ``bsr``'s dependencies have additional attribution requirements and associated papers.
