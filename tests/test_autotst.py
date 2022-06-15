import pytest
import numpy as np
import autotst


def test_permutations_p_value():

    # perfect predictions
    preds = np.array([1.0 for _ in range(100)] + [0.0 for _ in range(200)])
    labels = np.array([1 for _ in range(100)] + [0 for _ in range(200)])
    p = autotst.permutations_p_value(preds, labels)
    assert pytest.approx(p, 1e-3) == 0.0

    # totally undecided prediction
    preds = np.array([0.5 for _ in range(300)])
    labels = np.array([1 for _ in range(100)] + [0 for _ in range(200)])
    p = autotst.permutations_p_value(preds, labels)
    assert pytest.approx(p, 1e-3) == 1.0

    # decent predictions
    preds1 = np.random.normal(0.75, 0.2, size=100)
    preds2 = np.random.normal(0.25, 0.2, size=200)
    preds = np.concatenate((preds1, preds2))
    labels = np.array([1 for _ in range(100)] + [0 for _ in range(200)])
    p_decent = autotst.permutations_p_value(preds, labels)

    # somehow undecided predictions
    preds1 = np.random.normal(0.6, 0.2, size=100)
    preds2 = np.random.normal(0.4, 0.2, size=200)
    preds = np.concatenate((preds1, preds2))
    labels = np.array([1 for _ in range(100)] + [0 for _ in range(200)])
    p_somehow_undecided = autotst.permutations_p_value(preds, labels)

    # undecided predictions
    preds1 = np.random.normal(0.55, 0.2, size=100)
    preds2 = np.random.normal(0.45, 0.2, size=200)
    preds = np.concatenate((preds1, preds2))
    labels = np.array([1 for _ in range(100)] + [0 for _ in range(200)])
    p_undecided = autotst.permutations_p_value(preds, labels)

    # very undecided predictions
    preds1 = np.random.normal(0.51, 0.2, size=100)
    preds2 = np.random.normal(0.49, 0.2, size=200)
    preds = np.concatenate((preds1, preds2))
    labels = np.array([1 for _ in range(100)] + [0 for _ in range(200)])
    p_very_undecided = autotst.permutations_p_value(preds, labels)

    assert p_decent <= p_somehow_undecided
    assert p_somehow_undecided <= p_undecided
    assert p_undecided <= p_very_undecided

    # only 1d array authorized
    with pytest.raises(ValueError):
        predictions_2d = np.random.normal([1, 3], [0.5, 1.5], size=(500, 2))
        labels_2d = np.array([1.0] * 500 + [0.0] * 500)
        autotst.permutations_p_value(predictions_2d, labels_2d)

    # preds and labels should be of same length
    with pytest.raises(ValueError):
        predictionsE = np.array([1.0] * 20)
        labelsE = np.array([1.0] * 10 + [0.0] * 11)
        autotst.permutations_p_value(predictionsE, labelsE)


def test_get_weights():

    n1 = 10
    n2 = 20
    labels = np.array([1 for _ in range(n1)] + [0 for _ in range(n2)])
    weights = autotst.get_weights(labels)

    assert pytest.approx(sum(weights[:n1])) == sum(weights[n1:])


def test_split_sets():

    n1 = 100
    n2 = 200

    p = np.random.normal(0.75, 0.2, size=n1)
    q = np.random.normal(0.25, 0.2, size=n2)

    for ratio in (0.25, 0.5, 0.75):

        pq = autotst.SplittedSets.from_samples(p, q, ratio)

        assert len(pq.training_set) + len(pq.test_set) == n1 + n2
        assert len(pq.training_labels) + len(pq.test_labels) == n1 + n2

        assert len(pq.training_set) == len(pq.training_labels)
        assert len(pq.test_set) == len(pq.test_labels)

        assert (
            pytest.approx(
                len(pq.training_set) / (len(pq.test_set) + len(pq.training_set))
            )
            == ratio
        )

        l1 = len(np.where(pq.training_labels == 1)[0])
        l2 = len(np.where(pq.training_labels == 0)[0])

        assert pytest.approx(l1 / l2) == n1 / n2


def test_pipelines():

    n1 = 50
    n2 = 100

    p = np.random.normal(0.75, 0.2, size=n1)
    q = np.random.normal(0.25, 0.2, size=n2)

    p1 = autotst.p_value(p, q, time_limit=1)

    at = autotst.AutoTST(p, q)
    p2 = at.p_value(time_limit=1)

    assert pytest.approx(p1) == p2


def test_global():

    n1 = 400
    n2 = 200

    p = np.random.normal(1, 1, size=n1)
    q = np.random.normal(1, 1, size=n2)
    p_value = autotst.p_value(p, q, time_limit=1)

    assert p_value > 0.1

    p = np.random.normal(0, 1, size=n1)
    q = np.random.normal(1, 1, size=n2)
    p_value = autotst.p_value(p, q, time_limit=1)
    assert p_value < 0.05

