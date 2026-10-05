import numpy as np


def test_scaffold_split_never_shares_a_scaffold(transfer):
    scaffolds = ["a"] * 5 + ["b"] * 3 + ["c"] * 2 + ["d"] + ["e"] + [""] * 4
    train, test = transfer.scaffold_holdout(scaffolds, test_fraction=0.25)
    assert set(train).isdisjoint(test) and len(train) + len(test) == len(scaffolds)
    assert {scaffolds[i] for i in train}.isdisjoint({scaffolds[i] for i in test})


def test_scaffold_split_is_deterministic(transfer):
    scaffolds = list("aabbbccdefgh")
    first = transfer.scaffold_holdout(scaffolds, 0.25)
    second = transfer.scaffold_holdout(scaffolds, 0.25)
    assert all((a == b).all() for a, b in zip(first, second))


def test_training_subsets_are_nested_within_a_seed(transfer):
    pool = np.arange(100)
    small = set(transfer.training_subset(pool, seed=3, fraction=0.25))
    large = set(transfer.training_subset(pool, seed=3, fraction=0.5))
    assert small < large and len(small) == 25 and len(large) == 50


def test_training_subsets_differ_between_seeds(transfer):
    pool = np.arange(100)
    assert set(transfer.training_subset(pool, 0, 0.25)) != set(transfer.training_subset(pool, 1, 0.25))


def test_regression_metrics_on_a_perfect_prediction(transfer):
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = transfer.regression_metrics(y, y)
    assert metrics["rmse"] == 0.0 and metrics["r2"] == 1.0 and metrics["spearman"] > 0.99


def test_ac50_quality_checks_keep_only_reliable_hits(transfer):
    import pandas as pd
    metrics = pd.DataFrame({
        "hit_code": [1, 2, 0, 1, 1, 1, 0.9993, 2],
        "logac50": [0.5, -0.5, 0.2, 1.176, -2.0, -1.5, 0.3, np.nan],
    })
    kept, steps = transfer.quality_hits(metrics)
    assert list(kept["logac50"]) == [0.5, -0.5]          # not: no-hit, placeholders, out of range, fractional code, missing
    assert steps[-1][1] == 2 and [n for _, n in steps] == [8, 6, 5, 3, 2]
