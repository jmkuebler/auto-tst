import typing
import numpy as np
from .autotst_types import Weights, Predictions, Labels, Samples, Dataset
from .model import Model, AutoGluonTabularPredictor
from .splitted_sets import SplittedSets


def permutations_p_value(
    predictions: Predictions, labels: Labels, permutations: int = 10000
) -> float:
    """
    Compute p value of the witness mean discrepancy test statistic via permutations

    :param predictions: one-dimensional array with the witness predictions of the test data
    :param labels: one-dimensional array with labels 1 and 0 indicating data coming from P or Q
    :param int permutations: Number of permutations
    :return: p value
    """

    if len(predictions.shape) > 1:
        raise ValueError("predictions should be one dimentional")

    if len(labels.shape) > 1:
        raise ValueError("labels should be one dimentional")

    if len(predictions) != len(labels):
        raise ValueError("predictions and labels should be of the same length")

    p_samp = predictions[labels == 1]
    q_samp = predictions[labels == 0]
    tau = np.mean(p_samp) - np.mean(q_samp)  # value on original partition
    p = 1 / (permutations + 1)
    for i in range(0, permutations):
        np.random.shuffle(predictions)
        p_samp = predictions[labels == 1]
        q_samp = predictions[labels == 0]
        tau_sim = np.mean(p_samp) - np.mean(q_samp)

        if tau <= tau_sim:
            p += 1.0 / (permutations + 1)

    return p


def get_weights(label_train: Labels) -> Weights:
    """
    Labels being a one-dimensional array with labels 1 and 0, returns an array of
    weights that gives higher values to indexes corresponding to the less represented
    label.
    """

    n1 = len(np.where(label_train)[0])
    n2 = len(label_train) - n1

    ratio = n1 / (n1 + n2)
    return np.array([1.0 - ratio] * n1 + [ratio] * n2)


def fit_witness(
    data_train: Dataset, label_train: Dataset, model: Model, **kwargs
) -> None:
    """
    Calls the fit function of the model on the provided dataset, weighted to account
    for the difference of representation of the two labels.
    :param predictions: one-dimensional array with the witness predictions of the test data
    :param labels: one-dimensional array with labels 1 and 0 indicating data coming from one sample or the other
    :param model: the model on which the fit function is applied.
    """

    if len(data_train) != len(label_train):
        raise ValueError("data_train and label_train should be of the same length")

    weights = get_weights(label_train)
    model.fit(data_train, label_train, weights, **kwargs)


def p_value_evaluate(
    model: Model, data_test: Dataset, labels_test: Labels, permutations: int = 10000
) -> typing.Tuple[Dataset, float]:

    """
    Apply the model to generate predictions, and uses these predictions to evaluate the  p value.
    :param model: the model used for prediction, assumed to have been fitted
    :param dataset: dataset
    :param labels: one-dimensional array with labels 1 and 0 indicating data coming from one sample or the other
    :param permutations: number of permutations when estimating the p-value
    :return: the predictions and the p value
    """

    prediction_test = np.array(model.predict(data_test))
    return (
        prediction_test,
        permutations_p_value(prediction_test, labels_test, permutations),
    )


def get_default_model() -> Model:
    """
    Returns an instance of the AutoGluonTabularPredictor, with default parameters
    """
    return AutoGluonTabularPredictor()


def p_value(
    sample_p: Samples,
    sample_q: Samples,
    model: typing.Optional[Model] = None,
    split_ratio: float = 0.5,
    permutations: int = 10000,
    **fit_kwargs
) -> float:

    """
    Split the datasets unto a training and a test set, fit the model using the training set
    and uses the test set to compute the p-value.
    :param sample_p: samples drawn from a first distribution
    :param sample_q: samples drawn from a second distribution
    :param model: instance of model for fitting and prediction. If None (the default): an AutoGluonTabularPredictor will be used
    :param split_ratio: for splitting into learning and testing sets
    :param permutations: number of permutations used to estimate the p value
    :param fit_kwargs: parameters to the model's fit function
    :return: p value
    """

    if model is None:
        model = get_default_model()

    splitted_sets = typing.cast(
        SplittedSets, SplittedSets.from_samples(sample_p, sample_q, split_ratio)
    )

    fit_witness(
        splitted_sets.training_set, splitted_sets.training_labels, model, **fit_kwargs
    )

    return p_value_evaluate(
        model,
        splitted_sets.test_set,
        splitted_sets.test_labels,
        permutations=permutations,
    )[1]


def interpret(
    data_test: Dataset, predictions: Predictions, k: int = 1
) -> typing.Tuple[Dataset, Dataset]:

    """
    Returns the k most typical examples from the two distributions
    :param data_test: dataset with the first items corresponding to the first distribution and the last items to the second distributions
    :param predictions: label prediction corresponding to the dataset
    :param k: number of items to extract from the dataset, for each distribution
    :return: the k most typical examples from the two distributions
    """

    if len(data_test) != len(predictions):
        raise ValueError("data_test and predictions should be of the same length")

    most_typical = np.argsort(predictions)
    p_typical = data_test[most_typical[:+k]]
    q_typical = data_test[most_typical[-k:]]
    return p_typical, q_typical
