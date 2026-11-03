import numpy as np

from dc.curve import ece, operating_point, pick_threshold
from dc.policy import Calibrator, decide


def test_operating_point_counts_coverage_and_errors():
    conf = np.array([0.9, 0.8, 0.3, 0.1])
    correct = np.array([1.0, 0.0, 1.0, 0.0])
    coverage, risk = operating_point(conf, correct, 0.5)
    assert coverage == 0.5  # 2 of 4 auto-resolved
    assert risk == 0.5  # 1 of those 2 was wrong


def test_pick_threshold_holds_the_budget():
    conf = np.array([0.95, 0.9, 0.8, 0.7, 0.6])
    correct = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    assert pick_threshold(conf, correct, 0.0) == 0.8


def test_no_threshold_meets_an_impossible_budget():
    conf = np.array([0.9, 0.8])
    correct = np.array([0.0, 0.0])
    assert pick_threshold(conf, correct, 0.0) is None
    assert decide(0.99, None) == "escalate"


def test_perfectly_calibrated_scores_have_zero_ece():
    conf = np.array([0.0, 0.0, 1.0, 1.0])
    correct = np.array([0.0, 0.0, 1.0, 1.0])
    assert ece(conf, correct) == 0.0


def test_calibrator_learns_a_clear_signal():
    X = [[1.0], [1.0], [1.0], [0.0], [0.0], [0.0]]
    y = [1, 1, 1, 0, 0, 0]
    p = Calibrator().fit(X, y).predict(X)
    assert p[0] > 0.7
    assert p[-1] < 0.3


def test_three_outcomes():
    assert decide(0.9, 0.8) == "auto_resolve"
    assert decide(0.7, 0.8) == "escape_hatch"
    assert decide(0.3, 0.8) == "escalate"
