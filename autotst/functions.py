import typing
import numpy as np
from .autotst_types import Weights, Predictions, Labels, Samples, Dataset
from .model import Model
from .splitted_sets import SplittedSets

def permutations_p_value(
        predictions: Predictions,
        labels: Labels,
        permutations: int =10000
)->float:
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
            p += float(1 / permutations)
    return p

def get_weights(
        label_train: Labels
)->Weights:

    n1 = len(np.where(label_train))
    n2 = len(label_train)-n1
    ratio = n1 / (n1+n2)
    return np.array([1.-ratio]*n1+[1.]*n2)

def fit_witness(
        data_train: Dataset,
        label_train: Dataset,
        model: Model,
        **kwargs
)->None:

    if len(data_train)!=len(label_train):
        raise ValueError("data_train and label_train should be of the same length")

    weights = get_weights(label_train)
    model.fit(data_train,label_train,weights,**kwargs)

def p_value_evaluate(
        model: Model,
        data_test: Dataset,
        permutations: int = 10000
)->float:

    prediction_test = model.predict(data_test)
    return permutations_p_value(np.array(prediction_test),permutations)
    
def p_value(
        sample_p: Samples,
        sample_q: Samples,
        model: Model,
        split_ratio: float=0.5,
        permutations: int = 10000,
        **fit_kwargs
)->float:
    splitted_sets = typing.cast(
        SplittedSets,
        SplittedSets.from_samples(
            sample_p,
            sample_q,
            split_ratio
        )
    )

    fit_witness(
        splitted_sets.training_set,
        splitted_sets.training_labels,
        model,
        **fit_kwargs
    )

    return p_value_evaluate(
        model,
        splitted_sets.test_set,
        permutations=permutations
    )

def interpret(
        self,
        data_test: Dataset,
        predictions: Predictions,
        k: int = 1
)->typing.Tuple[Dataset,Dataset]:

    most_typical=np.argsort(predictions)
    p_typical = data_test[most_typical[-k:]]
    q_typical = data_test[most_typical[:+k]]
    return p_typical,q_typical
