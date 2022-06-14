import numpy as np
import autotst
from autotst.model import AutoGluonTabularPredictor


# You can play around with the datasets here

# # 2-dimensional Gaussians
X = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=40)
Y = np.random.multivariate_normal([1, 0], [[1, 0], [0, 1]], size=20)

# # 1-dimensional Gaussians
# X = np.random.normal(0, 1, size=40)
# Y = np.random.normal(1, 1, size=20)

##


print("\n----- running the full pipeline in one line, with custom time limit -----")
p = autotst.p_value(X, Y, time_limit=5)
print("P-Value", p)
print()

print("\n----- same, but with explicit pipeline ----")
model = AutoGluonTabularPredictor()

splitted_sets = autotst.SplittedSets.from_samples(X, Y)

autotst.fit_witness(
    splitted_sets.training_set, splitted_sets.training_labels, model, time_limit=5
)

predictions, p_value = autotst.p_value_evaluate(
    model,
    splitted_sets.test_set,
    splitted_sets.test_labels)

most_significants = autotst.interpret(splitted_sets.test_set, predictions, k=1)

print("P-Value:", p_value)
print("Most significant examples from P and Q: ", most_significants)
