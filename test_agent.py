import asyncio
import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from server import RMAB, PullParams
from task import TASK_SPEC

load_dotenv()

LOG_DIR = Path(__file__).parent / "logs"


class Logger:
    def __init__(self):
        LOG_DIR.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = LOG_DIR / f"episode-{stamp}.jsonl"
        self.csv_path = LOG_DIR / f"pulls-{stamp}.csv"
        self.f = self.path.open("w", encoding="utf-8")
        self.csv_f = self.csv_path.open("w", encoding="utf-8", newline="")
        self.csv_writer = csv.writer(self.csv_f)
        self.csv_writer.writerow(["pull_number", "pulled_machine", "value_given", "total_value"])

    def emit(self, event: str, **fields):
        rec = {"ts": time.time(), "event": event, **fields}
        self.f.write(json.dumps(rec, default=str) + "\n")
        self.f.flush()

    def emit_pull(self, pull_number: int, machine_id: int, value: float, total: float):
        self.csv_writer.writerow([pull_number, machine_id, f"{value:.6f}", f"{total:.6f}"])
        self.csv_f.flush()

    def close(self):
        self.f.close()
        self.csv_f.close()

MODEL = "gpt-5.5"

PULL_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "pull",
        "description": "Pull machine `machine_id` (0..N-1). Returns a sample value which is your reward.",
        "parameters": {
            "type": "object",
            "properties": {"machine_id": {"type": "integer"}},
            "required": ["machine_id"],
            "additionalProperties": False,
        },
    },
}

BASH_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command in the sandbox for your own analysis (e.g. write a Python script to a file and run it). Returns stdout/stderr and exit code.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

TOOLS = [PULL_TOOL_SPEC, BASH_TOOL_SPEC]


async def run():
    log = Logger()
    print(f"logging to {log.path}")
    env = RMAB(task_spec=TASK_SPEC, secrets={"api_key": os.environ["OPENREWARD_API_KEY"]})
    await env.setup()
    try:
        client = OpenAI()
        prompt_text = "\n".join(b.text for b in env.get_prompt())
        messages = [{"role": "user", "content": prompt_text}]
        log.emit("episode_start", task_spec=TASK_SPEC, model=MODEL, prompt=prompt_text)

        end_reason = "cap"
        prev_cum = 0.0
        for step in range(TASK_SPEC["num_pulls"] * 3):
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="required",
            )
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_unset=True))

            log.emit(
                "assistant",
                step=step,
                content=msg.content,
                tool_calls=[tc.model_dump() for tc in (msg.tool_calls or [])],
            )
            print(f"[step {step}] Info: {msg}")

            if not msg.tool_calls:
                print(f"[step {step}] No tool was called: {msg.content}")
                end_reason = "no_tool_call"
                break

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                if tc.function.name == "pull":
                    result = await env.pull(PullParams(**args))
                    sample = env.cumulative_reward - prev_cum
                    prev_cum = env.cumulative_reward
                    log.emit_pull(env.t, args["machine_id"], sample, env.cumulative_reward)
                    log.emit(
                        "tool_result",
                        step=step,
                        tool="pull",
                        tool_call_id=tc.id,
                        machine_id=args["machine_id"],
                        sample=sample,
                        reward=result.reward,
                        cumulative=env.cumulative_reward,
                        t_env=env.t,
                        finished=result.finished,
                        text=result.blocks[0].text,
                    )
                    print(f"[step {step}] pull(machine_id={args['machine_id']}) → reward={result.reward:.5f} finished={result.finished}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.blocks[0].text,
                    })
                    if result.finished:
                        print(f"\ncumulative: {env.cumulative_reward:.4f}")
                        end_reason = "terminal"
                        log.emit("episode_end", cumulative=env.cumulative_reward, t_env=env.t, reason=end_reason)
                        return messages
                elif tc.function.name == "bash":
                    output, code = await env.sandbox.run(args["command"])
                    log.emit(
                        "tool_result",
                        step=step,
                        tool="bash",
                        tool_call_id=tc.id,
                        command=args["command"],
                        stdout=output,
                        exit_code=code,
                    )
                    print(f"[step {step}] bash({args['command']!r}) → exit {code}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"{output}\n\n(exit {code})",
                    })
                else:
                    log.emit("tool_result", step=step, tool=tc.function.name, tool_call_id=tc.id, error="unknown_tool")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"unknown tool: {tc.function.name}",
                    })
        log.emit("episode_end", cumulative=env.cumulative_reward, t_env=env.t, reason=end_reason)
        return messages
    finally:
        await env.teardown()
        log.close()


if __name__ == "__main__":
    messages = asyncio.run(run())
    print(messages)
