import numpy as np
import autotst
from autotst.model import AutoGluonTabularPredictor
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil

"""
We use data from the same distribution (2-d Gaussian) to asses correct estimation of p-values.
We run the experiment 200 times and plot the distribtution of the p-values.
If the method runs correctly, the p-values should be uniformly randomly distributed, or,
shifted towards larger values (which happens when the algorithm makes conservative decisions).
"""
p_values = []
for i in tqdm(range(100)):
    X = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=40)
    Y = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=20)

    tst = autotst.AutoTST(
        X, Y, model=AutoGluonTabularPredictor, path=f"AutogluonModels_type_I/type_I_{i}"
    )

    splitted_sets: autotst.SplittedSets = tst.split_data()
    tst.fit_witness(time_limit=2)  # using 2 seconds limit to make it relatively fast.
    p_values.append(tst.p_value_evaluate())

plt.hist(p_values)
plt.xlim((0, 1))
plt.show()

shutil.rmtree("AutogluonModels_type_I")  # remove all stored models
