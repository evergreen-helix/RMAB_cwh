import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bandit import Bandit
from task import TASK_SPEC

LOG_DIR = Path(__file__).parent / "logs"


class RandomPolicy:
    def __init__(self, n, seed=0):
        self.n = n
        self.rng = np.random.default_rng(seed)

    def select(self, t):
        return int(self.rng.integers(0, self.n))

    def update(self, machine, sample, t):
        pass


class RoundRobin:
    def __init__(self, n):
        self.n = n

    def select(self, t):
        return t % self.n

    def update(self, machine, sample, t):
        pass


class Greedy:
    def __init__(self, n, warmup=2):
        self.n = n
        self.warmup = warmup
        self.totals = np.zeros(n)
        self.counts = np.zeros(n)

    def select(self, t):
        if t < self.warmup * self.n:
            return t % self.n
        with np.errstate(invalid="ignore"):
            means = np.where(self.counts > 0, self.totals / np.maximum(self.counts, 1), -np.inf)
        return int(np.argmax(means))

    def update(self, machine, sample, t):
        self.totals[machine] += sample
        self.counts[machine] += 1


class SlidingUCB:
    def __init__(self, n, window=30, alpha=2.0):
        self.n = n
        self.window = window
        self.alpha = alpha
        self.history = []

    def select(self, t):
        cutoff = t - self.window
        sums = np.zeros(self.n)
        counts = np.zeros(self.n)
        for tt, m, s in self.history:
            if tt > cutoff:
                sums[m] += s
                counts[m] += 1
        for i in range(self.n):
            if counts[i] == 0:
                return i
        means = sums / counts
        n_recent = int(counts.sum())
        ucb = means + self.alpha * np.sqrt(np.log(max(n_recent, 1)) / counts)
        return int(np.argmax(ucb))

    def update(self, machine, sample, t):
        self.history.append((t, machine, sample))


class Oracle:
    def __init__(self, bandit):
        self.bandit = bandit

    def select(self, t):
        return int(np.argmax([self.bandit.mu(i, t) for i in range(self.bandit.n_machines)]))

    def update(self, machine, sample, t):
        pass


def make_bandit():
    return Bandit(
        n_machines=TASK_SPEC["num_machines"],
        n_pulls=TASK_SPEC["num_pulls"],
        a=TASK_SPEC["a"],
        b=TASK_SPEC["b"],
        c=TASK_SPEC["c"],
        phi=TASK_SPEC["phi"],
        d=TASK_SPEC["d"],
        sigma_a=TASK_SPEC["sigma_a"],
        sigma_d=TASK_SPEC["sigma_d"],
        sample_seed=TASK_SPEC["sample_seed"],
    )


def run_policy(name, policy, bandit, n_pulls, stamp):
    csv_path = LOG_DIR / f"pulls-{name}-{stamp}.csv"
    cumulative = 0.0
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pull_number", "pulled_machine", "value_given", "total_value"])
        for t in range(n_pulls):
            m = policy.select(t)
            s = bandit.sample(m, t)
            cumulative += s
            policy.update(m, s, t)
            w.writerow([t + 1, m, f"{s:.6f}", f"{cumulative:.6f}"])
    print(f"{name:12s} cumulative={cumulative:9.4f}  -> {csv_path.name}")
    return cumulative


def main():
    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    n = TASK_SPEC["num_machines"]
    n_pulls = TASK_SPEC["num_pulls"]

    factories = [
        ("random",      lambda b: RandomPolicy(n, seed=0)),
        ("roundrobin",  lambda b: RoundRobin(n)),
        ("greedy",      lambda b: Greedy(n, warmup=2)),
        ("sliding_ucb", lambda b: SlidingUCB(n, window=30, alpha=2.0)),
        ("oracle",      lambda b: Oracle(b)),
    ]

    print(f"task: {n} machines, {n_pulls} pulls, seed={TASK_SPEC['sample_seed']}")
    print("-" * 60)
    for name, factory in factories:
        bandit = make_bandit()
        run_policy(name, factory(bandit), bandit, n_pulls, stamp)


if __name__ == "__main__":
    main()
