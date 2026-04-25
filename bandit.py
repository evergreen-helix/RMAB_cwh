"""Bandit dynamics with explicit per-machine coefficients.

μ_i(t) = a_i + b_i * sin(c_i * t + φ_i) + d_i * (t / n_pulls)
σ_i(t) = sigma_a_i + sigma_d_i * (t / n_pulls)
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Bandit:
    n_machines: int
    n_pulls: int
    a: list[float]
    b: list[float]
    c: list[float]
    phi: list[float]
    d: list[float]
    sigma_a: list[float]
    sigma_d: list[float]
    sample_seed: int = 0

    def __post_init__(self) -> None:
        for name in ("a", "b", "c", "phi", "d", "sigma_a", "sigma_d"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.shape != (self.n_machines,):
                raise ValueError(f"{name} must have length {self.n_machines}, got {arr.shape}")
            setattr(self, name, arr)
        self.sample_rng = np.random.default_rng(self.sample_seed)

    def mu(self, i: int, t: int) -> float:
        return float(
            self.a[i]
            + self.b[i] * np.sin(self.c[i] * t + self.phi[i])
            + self.d[i] * (t / self.n_pulls)
        )

    def sigma(self, i: int, t: int) -> float:
        return float(self.sigma_a[i] + self.sigma_d[i] * (t / self.n_pulls))

    def sample(self, i: int, t: int) -> float:
        return float(self.sample_rng.normal(self.mu(i, t), max(self.sigma(i, t), 1e-9)))
