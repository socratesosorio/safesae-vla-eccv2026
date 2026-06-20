import numpy as np

from src.utils.metrics import bootstrap_rate_ci, cost_weighted_f1, pareto_frontier, rate


def test_rate_basic():
    assert rate([1, 0, 1, 1]) == 0.75


def test_bootstrap_ci_bounds():
    ci = bootstrap_rate_ci([1, 0, 1, 1, 0, 1], n_boot=200)
    assert 0.0 <= ci.lo <= ci.mean <= ci.hi <= 1.0


def test_cost_weighted_f1_nonnegative():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    score = cost_weighted_f1(y_true, y_pred, fn_weight=10.0)
    assert score >= 0.0


def test_pareto_frontier_monotonic():
    pts = [(0.8, -0.3), (0.7, -0.1), (0.6, -0.4), (0.75, -0.2)]
    frontier = pareto_frontier(pts)
    assert len(frontier) >= 1
