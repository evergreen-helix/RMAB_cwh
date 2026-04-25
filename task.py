"""Single source of truth for the task spec used by test_agent.py and run_heuristics.py."""
import math

T = 200

TASK_SPEC = {
    "task_id": "smoke-1",
    "num_machines": 5,
    "num_pulls": T,
    "a":       [0.5, 0.7, 0.9, 1.1, 1.3],
    "b":       [1.5, 1.5, 1.5, 1.5, 1.5],
    "c":       [2 * math.pi * k / T for k in (2, 2.3, 2.6, 2.9, 3)],
    "phi":     [0.0, 0.7, 1.4, 2.1, 2.8],
    "d":       [0.0, 0.0, 0.0, 0.0, 0.0],
    "sigma_a": [1.5, 1.5, 1.5, 1.5, 1.5],
    "sigma_d": [0.0, 0.0, 0.0, 0.0, 0.0],
    "sample_seed": 1,
}
