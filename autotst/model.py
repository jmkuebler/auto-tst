# Import packages
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.vision import ImagePredictor, ImageDataset
import pandas as pd
import numpy as np
import warnings
from .autotst_types import Dataset, Weights, Predictions


class Model:
    """
    Generic model class for two-sample tests
    """

    def __init__(self, **kwargs):
        raise NotImplementedError()

    def fit(self, data_train, label_train, weights, **kwargs):
        raise NotImplementedError()

    def predict(self, data_test):
        raise NotImplementedError()


class AutoGluonTabularPredictor(Model):
    """
    Wrapper model for the Tabular Predictor of the AutoGluon
    package
    """

    def __init__(self, **kwargs) -> None:
        self.model = TabularPredictor(
            label="label", sample_weight="weights", problem_type="regression", **kwargs
        )

    def fit(
        self,
        data_train: Dataset,
        label_train: Dataset,
        weights: Weights,
        presets: str = "best_quality",
        time_limit: int = 60,
        verbosity: int = 0,
        **kwargs
    ) -> None:
        """
        Wrapper around fit routine.
        :param data_train: training data
        :param label_train: training labels
        :param weights: weights for the loss
        :param presets: Autogluon preset
        :param time_limit: time limit for train (seconds)
        :param verbosity: control output of Autogluon
        :param kwargs: other arguments to be passed to AutoGluon's fit routine.
        :return:
        """
        df_train = pd.DataFrame(data_train)
        df_train["label"] = label_train
        df_train["weights"] = weights
        df_train = TabularDataset(df_train)
        self.model.fit(
            df_train,
            presets=presets,
            time_limit=time_limit,
            verbosity=verbosity,
            **kwargs
        )

    def predict(self, data_test: Dataset) -> Predictions:
        df_test = TabularDataset(pd.DataFrame(data_test))
        return self.model.predict(df_test)


class AutoGluonImagePredictor(Model):
    """
    Wrapper model for the Image Classifier of the AutoGluon
    package.
    The objective is classification, and the witness function uses the predicted probabilities.
    """

    def __init__(self, **kwargs) -> None:
        self.model = ImagePredictor(label="label", verbosity=0, **kwargs)

    def fit(
        self,
        data_train: Dataset,
        label_train: Dataset,
        weights: Weights,
        presets: str = "best_quality",
        time_limit: int = 60,
        **kwargs
    ) -> None:
        """
        Wrapper around fit routine.
        :param data_train: training data - provided as a list of image paths!
        :param label_train: training labels
        :param weights: weights for the loss - will be ignored here!!!
        :param presets: Autogluon preset
        :param time_limit: time limit for train (seconds)
        :param kwargs: other arguments to be passed to AutoGluon's fit routine.
        :return:
        """
        if weights[0] != 0.5:
            warnings.warn(
                "AutoGluonImagePredictor ignores the weights! consider oversampling or using another model."
            )
        df_train = pd.DataFrame({"image": data_train, "label": label_train})
        self.model.fit(df_train, presets=presets, time_limit=time_limit, **kwargs)

    def predict(self, data_test: Dataset) -> Predictions:
        df_test = pd.DataFrame({"image": data_test})
        predictions = np.array(self.model.predict_proba(df_test))
        predictions = predictions[:, 1]  # return probability of class '1'
        return predictions
