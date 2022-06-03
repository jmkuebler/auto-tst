# Import packages
import numpy as np
from .model import AutoGluonTabularPredictor


def permutations_p_value(predictions, labels, permutations=10000):
    """
    Compute p value of the witness mean discrepancy test statistic via permutations
    :param predictions: one-dimensional array with the witness predictions of the test data
    :param labels: one-dimensional array with labels 1 and 0 indicating data coming from P or Q
    :param int permutations: Number of permutations
    :return: p value
    """
    p_samp = predictions[labels == 1]
    q_samp = predictions[labels == 0]
    tau = np.mean(p_samp) - np.mean(q_samp)  # value on original partition
    p = 0.
    for i in range(0, permutations):
        np.random.shuffle(predictions)
        p_samp = predictions[labels == 1]
        q_samp = predictions[labels == 0]
        tau_sim = np.mean(p_samp) - np.mean(q_samp)

        if tau <= tau_sim:
            p += np.float(1 / permutations)
    return p


class AutoTST:
    """
    AutoML Two-Sample Test

    Documentation with example of the class goes here
    """
    def __init__(self, sample_p, sample_q, split_ratio=0.5, model=AutoGluonTabularPredictor, **model_kwargs):
        """
        Constructor
        :param sample_p: Sample drawn from P
        :param sample_q: Sample drawn from Q
        :param split_ratio: Ratio that defines how much data is used for training the witness
        :param model: Model used to learn the witness function
        :param **model_kwargs: Keyword arguments to initialize the model
        :return: None
        """
        self.X = sample_p
        self.Y = sample_q
        self.model = model(**model_kwargs)
        self.split_ratio = split_ratio
        self.size_ratio = len(sample_p) / (len(sample_p) + len(sample_q))
        self.data_train = None
        self.data_test = None
        self.label_train = None
        self.label_test = None
        self.prediction_test = None

    def split_data(self):
        """
        Split & label data using the instances splitting ratio. The splits are stored as attributes but also returned.
        :return: tuple, length=4. Tuple containing training/test data and train/test labels.
        """
        n = len(self.X)
        n_train = int(n * self.split_ratio)
        m = len(self.Y)
        m_train = int(m * self.split_ratio)
        X_train, X_test = self.X[:n_train], self.X[n_train:]
        Y_train, Y_test = self.Y[:m_train], self.Y[m_train:]
        self.data_train = np.concatenate((X_train, Y_train))
        self.data_test = np.concatenate((X_test, Y_test))
        self.label_train = np.array([1] * n_train + [0] * m_train)
        self.label_test = np.array([1] * (n - n_train) + [0] * (m - m_train))
        return self.data_train, self.data_test, self.label_train, self.label_test

    def fit_witness(self, **kwargs):
        """
        Fit witness
        :param kwargs: Keyword arguments to be passed to fit method of model
        :return: None
        """
        # specify weights (only relevant for imbalanced samples)
        weights = [1 - self.size_ratio if label == 1 else self.size_ratio for label in self.label_train]
        self.model.fit(self.data_train, self.label_train, weights, **kwargs)

    def p_value_evaluate(self, permutations=10000):
        """
        Evaluate p value
        :param permutations: number of permutations when estimating the p-value
        :return: p value
        """
        self.prediction_test = self.model.predict(self.data_test)
        pval = permutations_p_value(np.array(self.prediction_test), self.label_test, permutations=permutations)
        return pval

    def p_value(self):
        """
        Run the complete pipeline and return p value with default settings.
        :return: p-value
        """
        self.split_data()
        self.fit_witness()
        pval = self.p_value_evaluate()
        return pval

    def interpret(self, k=1):
        """
        Return the k most typical examples from P and Q.
        :return: Tuple: (k most significant examples from P, k most significant examples from Q)
        """
        if self.prediction_test is None:
            raise RuntimeError('Interpretation can only be done after the p-value was computed.')
        most_typical = np.argsort(self.prediction_test)
        return self.data_test[most_typical[-k:]], self.data_test[most_typical[:k]]
