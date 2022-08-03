# AutoML Two-Sample Test

[![Checked with MyPy](https://img.shields.io/badge/mypy-checked-blue)](https://github.com/python/mypy)
[![Code style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Tests](https://github.com/jmkuebler/auto-tst/actions/workflows/tests.yml/badge.svg)](https://github.com/jmkuebler/auto-tst/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/badge/PyPI-1.2-blue)](https://pypi.org/project/autotst/)
[![Downloads](https://static.pepy.tech/personalized-badge/autotst?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/autotst)
[![arXiv](https://img.shields.io/badge/arXiv-2206.08843-b31b1b.svg)](https://arxiv.org/abs/2206.08843) 

`autotst` is a Python package for easy-to-use two-sample testing and distribution shift detection.

Given two datasets `sample_P` and `sample_Q` drawn from distributions $P$ and $Q$, the 
goal is to estimate a $p$-value for the null hypothesis $P=Q$.
`autotst` achieves this by learning a witness function and taking its mean discrepancy as a test statistic
(see References).

The package provides functionalities to prepare the data, an interface to train an ML model, and methods
to evaluate $p$-values and interpret results.

By default, autotst uses the Tabular Predictor of [AutoGluon](https://auto.gluon.ai/), but it is easy 
to wrap and use your own favorite ML framework (see below).

The full documentation of the package can be found [here](https://jmkuebler.github.io/auto-tst/).

## Installation
Requires at least Python 3.7. Since the installation also installs AutoGluon, it can take a few moments.
```
pip install autotst
```

## How to use `autotst`
We provide worked out examples in the 'Example' directory. In the following assume that
`sample_P` and `sample_Q` are two `numpy` arrays containing samples from $P$ and $Q$. 

### Default Usage:

The easiest way to compute a $p$-value is to use the default settings
```python
import autotst
tst = autotst.AutoTST(sample_P, sample_Q)
p_value = tst.p_value()
```
You would then reject the null hypothesis if `p_value` is smaller or equal to your significance level.

### Customizing the testing pipeline
We highly recommend to use the pipeline step by step, which would look like this:
```python
import autotst
from autotst.model import AutoGluonTabularPredictor

tst = autotst.AutoTST(sample_P, sample_Q, split_ratio=0.5, model=AutoGluonTabularPredictor)
tst.split_data()
tst.fit_witness(time_limit=60)  # time limit adjustable to your needs (in seconds)
p_value = tst.p_value_evaluate(permutations=10000)  # control number of permutations in the estimation
```
This allows you to change the time limit for fitting the witness function and you can also pass other 
arguments to the fit model (see [AutoGluon](https://auto.gluon.ai/) documentation).

Since the permutations are very cheap, the default number of permutations is relatively high and should work for most
use-cases. If your significance level is, say, smaller than 1/1000, consider increasing it further.

### Customizing the machine learning model
If you have good domain knowledge about your problem and believe that a specific ML framework will work well,
it is easy to wrap your model. 
Therefore, simply inherit from the class `Model` and wrap the methods
(see our implementation in [`model.py`](autotst/model.py)).

You can then run the test simply by importing your model and initializing the test accordingly.

```python
import autotst

tst = autotst.AutoTST(sample_P, sample_Q, model=YourCustomModel)
...
... etc.
```

We also provide a wrapper for `AutoGluonImagePredictor`. However, it seems that this should not be used 
with small datasets and small training times.

## References
If you use this package, please cite this paper:

Jonas M. Kübler, Vincent Stimper, Simon Buchholz, Krikamol Muandet, Bernhard Schölkopf: "AutoML Two-Sample Test", [arXiv 2206.08843](https://arxiv.org/abs/2206.08843) (2022)

Bibtex:
```
@misc{kubler2022autotst,
  doi = {10.48550/ARXIV.2206.08843},
  url = {https://arxiv.org/abs/2206.08843},
  author = {Kübler, Jonas M. and Stimper, Vincent and Buchholz, Simon and Muandet, Krikamol and Schölkopf, Bernhard},  
  title = {AutoML Two-Sample Test},
  publisher = {arXiv},
  year = {2022},
}
```
