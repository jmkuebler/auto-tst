import numpy as np
import autotst


X = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=40)
Y = np.random.multivariate_normal([1, 0], [[1, 0], [0, 1]], size=20)


tst = autotst.AutoTST(X, Y)

data_train, data_test, label_train, label_test = tst.split_data()
tst.fit_witness(time_limit=5)
print(tst.p_value_evaluate())
print(tst.interpret(1))


tst_new = autotst.AutoTST(X, Y)
print(tst_new.p_value())
