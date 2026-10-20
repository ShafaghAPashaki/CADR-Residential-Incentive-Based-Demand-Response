"""Numerical contract for two intentionally overlapping rebound categories."""
import numpy as np


def run():
    cap = 7.0
    eps = 1e-9
    # 1: only incentive-shift; 2: only conflict shift; 3: both sufficient
    # individually; 4: baseline is already above cap and cannot be counted.
    baseline = np.array([6., 6., 6., 8.])
    after = np.array([8., 8., 8., 9.])
    incentive_in = np.array([2.5, 0., 2., 2.])
    conflict_in = np.array([0., 2.5, 2., 2.])
    incentive_rebound = (
        (incentive_in > eps) & (baseline <= cap + eps)
        & (after > cap + eps) & (after-incentive_in <= cap + eps)
    )
    conflict_rebound = (
        (conflict_in > eps) & (baseline <= cap + eps)
        & (after > cap + eps) & (after-conflict_in <= cap + eps)
    )
    assert incentive_rebound.tolist() == [True, False, True, False]
    assert conflict_rebound.tolist() == [False, True, True, False]
    assert np.any(incentive_rebound & conflict_rebound), (
        'Categories must be recorded separately and are not disjoint')
    print('PASS: incentive-only, conflict-only, overlapping and baseline-overload cases')

if __name__ == '__main__':
    run()
