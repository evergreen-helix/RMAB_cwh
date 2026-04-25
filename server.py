
from openreward import AsyncOpenReward, SandboxSettings
from openreward.environments import (
    Environment, JSONObject, Server, Split,
    TextBlock, ToolOutput, tool,
)
from openreward.toolsets import ClaudeCodeToolset
import numpy as np
from pydantic import BaseModel


class RMAB_model:
    def __init__(self, model_params={}):
        self.num_machines = np.array(model_params["num_machines"])
        self.num_pulls = np.array(model_params["num_pulls"])
        self.means_base = np.array(model_params["means"])
        self.sds_base = np.array(model_params["sds"])
        self.means = np.array(model_params["means"])
        self.sds = np.array(model_params["sds"])
        self.mean_drift_rates = np.array(model_params["mean_drift_rates"])
        self.sd_drift_rates = np.array(model_params["sd_drift_rates"])
        self.done_pulls = 0

    def get_sample(self, machine_index: int):
        sample = np.random.normal(
            loc=self.means[machine_index],
            scale=self.sds[machine_index],
            size=1,
        )
        self._update_machines()
        return float(sample[0])

    def _update_machines(self):
        self.done_pulls += 1
        self.means = self.means_base + (self.done_pulls * self.mean_drift_rates)
        self.sds = self.sds_base + (self.done_pulls * self.sd_drift_rates)
        return None

    def _get_means(self):
        return self.means

    def _get_sds(self):
        return self.sds


class RMABTaskSpec(BaseModel):
    task_id: str
    num_machines: int
    num_pulls: int
    means: list[float]
    sds: list[float]
    mean_drift_rates: list[float]
    sd_drift_rates: list[float]


class PullParams(BaseModel, extra="forbid"):
    machine_id: int


class RMAB(Environment):
    toolsets = [ClaudeCodeToolset]

    def __init__(self, task_spec: JSONObject = {}, secrets: dict[str, str] = {}):
        super().__init__(task_spec)
        self.config = RMABTaskSpec.model_validate(task_spec)
        self.model = RMAB_model({
            "num_machines": self.config.num_machines,
            "num_pulls": self.config.num_pulls,
            "means": self.config.means,
            "sds": self.config.sds,
            "mean_drift_rates": self.config.mean_drift_rates,
            "sd_drift_rates": self.config.sd_drift_rates,
        })
        self.cumulative_reward = 0.0

        if not secrets.get("api_key"):
            raise ValueError("OpenReward API key required (pass via secrets={'api_key': ...})")
        self.sandbox_settings = SandboxSettings(
            environment="evergreen/RMAB",
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
            f"You also have a sandbox shell available for your own analysis."
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

        sample = self.model.get_sample(i)
        self.cumulative_reward += sample
        terminal = self.model.done_pulls >= self.config.num_pulls

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
        if split == "train":
            return [{
                "task_id": "train-0",
                "num_machines": 3,
                "num_pulls": 50,
                "means": [0.0, 1.0, 2.0],
                "sds": [1.0, 1.0, 1.0],
                "mean_drift_rates": [0.1, -0.1, 0.3],
                "sd_drift_rates": [0.1, 0.5, 1.0],
            }]
        if split == "test":
            return [{
                "task_id": "test-0",
                "num_machines": 5,
                "num_pulls": 800,
                "means": [0.0, 0.5, 1.0, 1.5, 2.0],
                "sds": [0.5, 0.5, 0.5, 0.5, 0.5],
                "mean_drift_rates": [0.001, 0.002, -0.001, 0.003, -0.002],
                "sd_drift_rates": [0.0, 0.0, 0.0, 0.0, 0.0],
            }]
        raise ValueError(f"Unknown split: {split}")


if __name__ == "__main__":
    Server([RMAB]).run()
