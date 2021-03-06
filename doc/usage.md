## How to use `autotst`
We provide worked out examples in the 'Example' directory. In the following assume that
`sample_P` and `sample_Q` are two `numpy` arrays containing samples from P and Q. 

### Default Usage:

The easiest way to compute a p-value is to use the default settings
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
