import numpy as np


def test_nearest_neighbors_excludes_the_query(evaluate):
    x = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]])
    result = evaluate.nearest_neighbors(x, ["A", "B", "C"], 0, 2)
    assert list(result["DTXSID"]) == ["B", "C"]


def test_binary_metrics_perfect_ranking(evaluate):
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = evaluate.binary_metrics(y, p, threshold=0.9)
    assert metrics["auroc"] == 1.0 and metrics["sensitivity"] == 1.0 and metrics["specificity"] == 1.0
