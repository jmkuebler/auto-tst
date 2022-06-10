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

model = AutoGluonTabularPredictor()

splitted_sets  = autotst.SplittedSets.from_samples(X,Y)

autotst.fit_witness(
    splitted_sets.training_set,
    splitted_sets.training_labels,
    model,
    time_limit=5
)

p_value = autotst.p_value_evaluate(
    model,
    splitted_sets.test_set
)

print("P-Value:",p_value)

#print("----- running test with default settings (1 minute default time limit) -----")

#model = AutoGluonTabularPredictor()
#p_value = autotst.p_value(X,Y,model)

#print("P-Value:",p_value)
