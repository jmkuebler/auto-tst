import typing
import numpy as np
from .autotst_types import Samples, Dataset, Labels


class SplittedSets:
    def __init__(
        self,
        training_set: Dataset,
        test_set: Dataset,
        training_labels: Labels,
        test_labels: Labels,
    ):
        self.training_set = training_set
        self.test_set = test_set
        self.training_labels = training_labels
        self.test_labels = test_labels

    def training_split(self) -> typing.Tuple[int, int]:
        p = len(np.where(self.training_labels == 1)[0])
        q = len(self.training_set) - p
        return p, q

    def test_split(self) -> typing.Tuple[int, int]:
        p = len(np.where(self.test_labels == 1)[0])
        q = len(self.test_set) - p
        return p, q

    @staticmethod
    def split(
        X: Samples, Y: Samples, split_ratio: float
    ) -> typing.Tuple[Dataset, Dataset, Labels, Labels]:

        if X.shape[1:] != Y.shape[1:]:
            raise ValueError("X and Y should be samples of items of same dimension")

        if split_ratio < 0 or split_ratio > 1:
            raise ValueError("split ratio should be between 0 and 1")

        n = len(X)
        n_train = int(n * split_ratio)
        m = len(Y)
        m_train = int(m * split_ratio)
        X_train, X_test = X[:n_train], X[n_train:]
        Y_train, Y_test = Y[:m_train], Y[m_train:]
        data_train = np.concatenate((X_train, Y_train))
        data_test = np.concatenate((X_test, Y_test))
        label_train = np.array([1] * n_train + [0] * m_train)
        label_test = np.array([1] * (n - n_train) + [0] * (m - m_train))
        return data_train, data_test, label_train, label_test

    @classmethod
    def from_samples(
        cls, sample_p: Samples, sample_q: Samples, split_ratio: float = 0.5
    ) -> object:
        return cls(*cls.split(sample_p, sample_q, split_ratio))
