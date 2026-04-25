import math

import numpy as np

from bandit import Bandit


def _make_bandit(n=3, T=100, sample_seed=0):
    return Bandit(
        n_machines=n,
        n_pulls=T,
        a=[0.0] * n,
        b=[1.0] * n,
        c=[2 * math.pi * (k + 2) / T for k in range(n)],
        phi=[0.0] * n,
        d=[0.0] * n,
        sigma_a=[0.5] * n,
        sigma_d=[0.0] * n,
        sample_seed=sample_seed,
    )


def test_sample_seed_determinism():
    a = _make_bandit(sample_seed=1)
    b = _make_bandit(sample_seed=1)
    samples_a = [a.sample(i % 3, i) for i in range(50)]
    samples_b = [b.sample(i % 3, i) for i in range(50)]
    assert samples_a == samples_b


def test_drift_changes_mu():
    bandit = Bandit(
        n_machines=3, n_pulls=100,
        a=[0.0, 0.0, 0.0],
        b=[1.0, 1.0, 1.0],
        c=[2 * math.pi * 2 / 100] * 3,
        phi=[0.0, 0.0, 0.0],
        d=[0.5, -0.5, 0.0],
        sigma_a=[0.1, 0.1, 0.1],
        sigma_d=[0.0, 0.0, 0.0],
    )
    assert [bandit.mu(i, 0) for i in range(3)] != [bandit.mu(i, 99) for i in range(3)]


def test_best_arm_rotates():
    bandit = _make_bandit(n=5, T=800)
    best_per_step = [int(np.argmax([bandit.mu(i, t) for i in range(5)])) for t in range(0, 800, 50)]
    rotations = sum(1 for x, y in zip(best_per_step, best_per_step[1:]) if x != y)
    assert rotations >= 3, f"best arm only rotated {rotations} times across 800 pulls"


def test_sigma_drift():
    bandit = Bandit(
        n_machines=2, n_pulls=100,
        a=[0.0, 0.0], b=[0.0, 0.0], c=[0.0, 0.0], phi=[0.0, 0.0], d=[0.0, 0.0],
        sigma_a=[0.5, 0.5], sigma_d=[1.0, 0.0],
    )
    assert math.isclose(bandit.sigma(0, 0), 0.5)
    assert math.isclose(bandit.sigma(0, 100), 1.5)
    assert math.isclose(bandit.sigma(1, 100), 0.5)
