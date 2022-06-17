import typing
import numpy as np
from .autotst_types import Samples, Dataset, Predictions
from .splitted_sets import SplittedSets
from .model import AutoGluonTabularPredictor, Model
from . import functions


class AutoTST:
    """
    AutoML Two-Sample Test

    Documentation with example of the class goes here
    """

    def __init__(
        self,
        sample_p: Samples,
        sample_q: Samples,
        split_ratio: float = 0.5,
        model: typing.Type[Model] = AutoGluonTabularPredictor,
        **model_kwargs
    ) -> None:
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
        self.splitted_sets: typing.Optional[SplittedSets] = None
        self.prediction_test: typing.Optional[Predictions] = None
        self._fitted = False

    def split_data(self) -> SplittedSets:
        """
        Split & label data using the instances splitting ratio. The splits are stored as attributes but also returned.
        """
        self.splitted_sets = typing.cast(
            SplittedSets, SplittedSets.from_samples(self.X, self.Y, self.split_ratio)
        )
        return self.splitted_sets

    def fit_witness(self, **kwargs) -> None:
        """
        Fit witness

        :param kwargs: Keyword arguments to be passed to fit method of model
        :return: None
        """
        if self.splitted_sets is None:
            raise ValueError("split_data should be called first")
        data_train = self.splitted_sets.training_set
        label_train = self.splitted_sets.training_labels
        functions.fit_witness(data_train, label_train, self.model, **kwargs)
        self._fitted = True

    def p_value_evaluate(self, permutations: int = 10000) -> float:
        """
        Evaluate p value.

        :param permutations: number of permutations when estimating the p-value
        :return: p value
        """
        if permutations < 0:
            raise ValueError("permutations should be positive")
        if not self._fitted:
            raise ValueError("the model should be trained first")
        if not self.splitted_sets:
            raise ValueError("split_data should be called first")
        data_test = self.splitted_sets.test_set
        label_test = self.splitted_sets.test_labels
        self.prediction_test = np.array(self.model.predict(data_test))
        return functions.permutations_p_value(
            self.prediction_test, label_test, permutations=permutations
        )

    def p_value(self, permutations: int = 10000, **fit_kwargs):
        """
        Run the complete pipeline and return p value with default settings.

        :return: p-value
        """
        self.split_data()
        self.fit_witness(**fit_kwargs)
        pval = self.p_value_evaluate(permutations=permutations)
        return pval

    def interpret(self, k=1):
        """
        Return the k most typical examples from P and Q.

        :return: Tuple: (k most significant examples from P, k most significant examples from Q)
        """
        if self.prediction_test is None:
            raise RuntimeError(
                "Interpretation can only be done after the p-value was computed."
            )
        p, q = self.splitted_sets.training_split()
        if k > p or k > q:
            raise ValueError("k should be between {} and {}".format(p, q))
        return functions.interpret(self.splitted_sets.test_set, self.prediction_test, k)
