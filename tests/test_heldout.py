import numpy as np
import pytest

from pdeview.data.audit import ic_signature
from pdeview.heldout import (
    HALF_CELL,
    Case,
    _sha1,
    heldout_cases,
    initial_condition,
    seen_cases,
)
from pdeview.solver.swe import cell_centres, dam_break_ic


def test_training_distribution_has_81_initial_conditions():
    seen = seen_cases()
    assert len(seen) == 81
    assert sum(c.weight for c in seen) == pytest.approx(1.0)
    sigs = {ic_signature(c.radius) for c in seen}
    assert len(sigs) == 81  # one distinct IC per radius interval
    radii = np.random.default_rng(0).uniform(0.3, 0.7, 2000)
    assert {ic_signature(r) for r in radii} <= sigs  # PDEBench's generator can't leave the set


def test_heldout_initial_conditions_are_new_and_in_spec():
    cases, _ = heldout_cases(n=20, seed=1)
    x = cell_centres()
    train = {_sha1(initial_condition(c, x)) for c in seen_cases()}
    sigs = [_sha1(initial_condition(c, x)) for c in cases]
    assert not set(sigs) & train
    assert len(set(sigs)) == len(cases)
    assert {c.family for c in cases} == {"subcell", "cellavg"}
    for c in cases:
        assert 0.3 <= c.radius <= 0.7
        assert abs(c.cx) <= HALF_CELL and abs(c.cy) <= HALF_CELL
    assert heldout_cases(n=20, seed=1) == heldout_cases(n=20, seed=1)


def test_cell_average_initial_condition():
    x = cell_centres()
    dx = x[1] - x[0]
    h0 = initial_condition(Case("t", "cellavg", 0.5, ic="cell_average"), x)
    assert h0.min() == 1.0 and h0.max() == 2.0
    assert ((h0 > 1) & (h0 < 2)).sum() > 50  # the cells cut by the dam wall
    assert (h0 - 1).sum() * dx**2 == pytest.approx(np.pi * 0.5**2, rel=2e-3)
    np.testing.assert_array_equal(h0, h0.T)


def test_point_initial_condition_matches_pdebench_rule():
    x = cell_centres()
    c = Case("t", "subcell", 0.45, 0.01, -0.005)
    np.testing.assert_array_equal(initial_condition(c, x),
                                  dam_break_ic(x, x, 0.45, (0.01, -0.005)))
