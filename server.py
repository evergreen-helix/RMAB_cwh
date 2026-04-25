from openreward import AsyncOpenReward, SandboxSettings
from openreward.environments import (
    Environment, JSONObject, Server, Split,
    TextBlock, ToolOutput, tool,
)
from openreward.toolsets import ClaudeCodeToolset
from pydantic import BaseModel

from bandit import Bandit


class RMABTaskSpec(BaseModel):
    task_id: str
    num_machines: int
    num_pulls: int
    a: list[float]
    b: list[float]
    c: list[float]
    phi: list[float]
    d: list[float]
    sigma_a: list[float]
    sigma_d: list[float]
    sample_seed: int = 0


class PullParams(BaseModel, extra="forbid"):
    machine_id: int


class RMAB(Environment):
    toolsets = [ClaudeCodeToolset]

    def __init__(self, task_spec: JSONObject = {}, secrets: dict[str, str] = {}):
        super().__init__(task_spec)
        self.config = RMABTaskSpec.model_validate(task_spec)
        self.bandit = Bandit(
            n_machines=self.config.num_machines,
            n_pulls=self.config.num_pulls,
            a=self.config.a,
            b=self.config.b,
            c=self.config.c,
            phi=self.config.phi,
            d=self.config.d,
            sigma_a=self.config.sigma_a,
            sigma_d=self.config.sigma_d,
            sample_seed=self.config.sample_seed,
        )
        self.t = 0
        self.cumulative_reward = 0.0

        if not secrets.get("api_key"):
            raise ValueError("OpenReward API key required (pass via secrets={'api_key': ...})")
        self.sandbox_settings = SandboxSettings(
            environment="evergreen/RMAB_cwh",
            image="generalreasoning/python-ds:3.12-tools",
            machine_size="1:2",
            block_network=True,
        )
        or_client = AsyncOpenReward(api_key=secrets.get("api_key"))
        self.sandbox = or_client.sandbox(self.sandbox_settings)

    async def setup(self) -> None:
        await self.sandbox.start()

    async def teardown(self) -> None:
        await self.sandbox.stop()

    def get_prompt(self) -> list[TextBlock]:
        n = self.config.num_machines
        return [TextBlock(text=(
            f"You are facing {n} machines, indexed 0..{n - 1}.\n"
            f"On each turn, call `pull(machine_id=...)` to get a value. "
            f"You want to maximise the value you collect.\n"
            f"You have {self.config.num_pulls} many pulls in total you must complete.\n"
            f"You also have a sandbox shell available for your own analysis. "
            f"Files you write persist across calls, but each shell invocation starts in a fresh cwd — use absolute paths."
        ))]

    @tool
    async def pull(self, params: PullParams) -> ToolOutput:
        """Pull machine `machine_id` (0..N-1). Returns a sample value which is your reward."""
        i = params.machine_id
        n = self.config.num_machines
        if not (0 <= i < n):
            return ToolOutput(
                blocks=[TextBlock(text=f"machine_id must be an integer in [0, {n - 1}]")],
                reward=0.0,
                finished=False,
            )

        sample = self.bandit.sample(i, self.t)
        self.t += 1
        self.cumulative_reward += sample
        terminal = self.t >= self.config.num_pulls

        text = f"machine {i} → {sample:.4f}"
        if terminal:
            text += f"\nepisode complete. cumulative reward: {self.cumulative_reward:.4f}"
            return ToolOutput(
                blocks=[TextBlock(text=text)],
                reward=self.cumulative_reward,
                finished=True,
            )

        return ToolOutput(
            blocks=[TextBlock(text=text)],
            reward=sample,
            finished=False,
        )

    @classmethod
    def list_splits(cls):
        return [Split(name="train", type="train"), Split(name="test", type="test")]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        import math
        T_train, T_test = 50, 800
        c_train = [2 * math.pi * k / T_train for k in (2, 2.5, 3)]
        c_test = [2 * math.pi * k / T_test for k in (2, 2.3, 2.6, 2.9, 3)]

        if split == "train":
            return [{
                "task_id": "train-0",
                "num_machines": 3,
                "num_pulls": T_train,
                "a":       [0.0, 0.5, 1.0],
                "b":       [1.0, 0.8, 0.5],
                "c":       c_train,
                "phi":     [0.0, math.pi / 2, math.pi],
                "d":       [0.5, -0.3, 0.0],
                "sigma_a": [0.5, 0.5, 0.5],
                "sigma_d": [0.0, 0.0, 0.0],
                "sample_seed": 1,
            }]
        if split == "test":
            return [{
                "task_id": "test-0",
                "num_machines": 5,
                "num_pulls": T_test,
                "a":       [0.0, 0.3, 0.6, 0.9, 1.2],
                "b":       [1.0, 1.2, 0.8, 1.0, 0.6],
                "c":       c_test,
                "phi":     [0.0, 0.7, 1.4, 2.1, 2.8],
                "d":       [0.3, -0.2, 0.1, -0.1, 0.2],
                "sigma_a": [0.3, 0.4, 0.3, 0.5, 0.4],
                "sigma_d": [0.0, 0.0, 0.0, 0.0, 0.0],
                "sample_seed": 42,
            }]
        raise ValueError(f"Unknown split: {split}")


if __name__ == "__main__":
    Server([RMAB]).run()
