# About

## Overview

Given two datasets `sample_P` and `sample_Q` drawn from distributions $P$ and $Q$, the 
goal is to estimate a $p$ value for the null hypothesis $P=Q$.
`autotst` achieves this by learning a witness function and taking its mean discrepancy as a test statistic
(see References).

The package provides functionalities to prepare the data, an interface to train an ML model, and methods
to evaluate p values and interpret results.

By default, autotst uses the Tabular Predictor of [AutoGluon](https://auto.gluon.ai/), but it is easy 
to wrap and use your own favorite ML framework.

## Source code and usage examples

The source code and usage examples can be found on [github](https://github.com/jmkuebler/auto-tst)


## Reference

Jonas M. Kübler, Vincent Stimper, Simon Buchholz, Krikamol Muandet, Bernhard Schölkopf: "AutoML Two-Sample Test" (2022).