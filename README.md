# RMAB_cwh

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/evergreen/RMAB_cwh)

An environment for evaluating LLMs as policies on a restless multi-armed bandit with hidden drifting rewards.

## What this is for

The point of the env is to test one thing. Can an LLM run a sample budget against a non-stationary process when its only signal is its own pull history? Five levers, noisy rewards, a fixed budget, no story attached.

OR is useful here because the optimal is computable and the hardness is parametric. When the agent fails, you can usually point at why.

## Capabilities under test

- Change detection. The world drifts. The prompt doesn't mention it.
- Belief updating from sparse, single-arm samples. Each pull tells you about one curve at one moment.
- Costly investigation. Pulls are the whole bankroll.
- Memory and compression. 200+ turns is too much to hold in context as raw samples.
- Tool-use as cognition. There's a sandbox. Using it well is part of the test.

## LLM-as-policy

Most "LLMs + bandits" work has the model designing or shaping a reward function — ARMMAN/DLM is the canonical example. Here the LLM is the thing picking the action every turn. No wrapper, no planner, no classical algorithm in the loop. If it does well, the model did the work.

## Mechanics

Each machine is a noisy Gaussian whose mean drifts:

```
μ_i(t) = a_i + b_i · sin(c_i · t + φ_i) + d_i · (t / T)
σ_i(t) = σ_a_i + σ_d_i · (t / T)
```

Coefficients are drawn from the task seed, so episodes are reproducible to the sample. The dynamics fit the exogenous-global-process subclass of RMAB (Gafni & Cohen, arXiv 2202.13665).

## Splits

- `train`: 3 machines, 50 pulls. Used for development.
- `test`: 5 machines, 800 pulls.

## Reward

Each pull returns the sampled value. The cumulative is emitted on the terminal pull. Verification is programmatic, no judge model.

## Tools

- `pull(machine_id: int)` returns a sample.
- A Python sandbox via `ClaudeCodeToolset`. The agent can write code, fit models, and persist files across calls.

## Compute

`generalreasoning/python-ds:3.12-tools`. No GPU. Network blocked in production.

## Time horizon

50 to 800 tool calls per episode.

## Safety

The sandbox has no network. The domain is abstract. The prompt is short and doesn't ask for anything outside the env.

## License

TBD.

## Citation

```bibtex
@dataset{rmab_cwh_2026,
  title     = {RMAB_cwh — Restless Multi-Armed Bandit eval for LLM-as-policy},
  year      = {2026},
  publisher = {Jamie Norton}
}
```
