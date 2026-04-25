# RMAB_cwh

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/evergreen/RMAB_cwh)

LLM-as-policy on a Restless Multi-Armed Bandit with hidden drifting reward distributions.

## Goal — capability isolation

- Isolate long-horizon capability: **sample-budgeted estimation of a non-stationary process from sparse single-arm observations**
- No scenario re-skinning, no roleplay, no narrative — just the bare capability
- OR gives a clean substrate: known optimal, known hardness, seed-reproducible noise, mechanical failure modes

## What's being tested

- **Change detection without prompt** — world drifts; agent isn't told
- **Belief updating from sparse single-arm samples** — one pull = one observation of one of N drifting curves
- **Costly investigation under hard budget** — pulls are the entire bankroll
- **Memory + compression at horizon** — too long to hold raw samples in context
- **Tool-use as cognition** — sandbox provided; quality matters, not quantity

## Why "LLM-as-policy"

- Most LLM + bandit prior work places the LLM as a *reward designer* (e.g. ARMMAN/DLM)
- This env puts the LLM in the **policy seat** — every action comes directly from the model
- No RL agent wrapping it, no planner sitting on top, no classical algorithm in the loop

## Mechanics

- Each machine = a noisy Gaussian whose mean drifts deterministically:
  - `μ_i(t) = a_i + b_i·sin(c_i·t + φ_i) + d_i·(t/T)`
  - `σ_i(t) = σ_a_i + σ_d_i·(t/T)`
- Coefficients drawn per-task from a seed → fully reproducible
- *Exogenous-global-process* RMAB subclass (Gafni & Cohen, arXiv 2202.13665)

## Tasks

- `train` — 3 machines, 50 pulls (smoke / development)
- `test` — 5 machines, 800 pulls (full evaluation)

## Reward

- Per-pull reward = the sampled value
- Cumulative emitted on terminal pull (`finished=True`)
- Programmatic verification — no LLM grader

## Tools

- `pull(machine_id: int)` — pull a machine, return a sample
- Python sandbox via `ClaudeCodeToolset` — write code, fit models, persist files across calls

## Compute

- Image: `generalreasoning/python-ds:3.12-tools`
- No GPU; modest CPU / memory
- Network blocked

## Time horizon

- Multi-turn: 50–800 tool calls per episode

## Safety

- Sandbox is network-blocked
- Abstract bandit domain; no dual-use risk
- Minimal prompt — no jailbreak surface

## License

- TBD

## Citation

```bibtex
@dataset{rmab_cwh_2026,
  title     = {RMAB_cwh — Restless Multi-Armed Bandit eval for LLM-as-policy},
  year      = {2026},
  publisher = {Jamie Norton}
}
```
