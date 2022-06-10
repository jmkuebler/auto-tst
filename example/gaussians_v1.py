import numpy as np
import autotst
from autotst.model import AutoGluonTabularPredictor

# You can play around with the datasets here
# # 2-dimensional Gaussians
# X = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=40)
# Y = np.random.multivariate_normal([1, 0], [[1, 0], [0, 1]], size=20)
#  # 1-dimensional Gaussians
X = np.random.normal(0, 1, size=40)
Y = np.random.normal(1, 1, size=20)

print("----- running test step by step with custom time limit -----")
tst = autotst.AutoTST(X, Y, model=AutoGluonTabularPredictor)

splitted_sets : autotst.SplittedSets = tst.split_data()
tst.fit_witness(time_limit=5)
print("P-Value", tst.p_value_evaluate())
print("Most significant examples from P and Q: ", tst.interpret(1))


print("----- running test with default settings (1 minute default time limit) -----")
tst_new = autotst.AutoTST(X, Y)
print("P-value with default settings (time limit of 1 minutes): ", tst_new.p_value())


