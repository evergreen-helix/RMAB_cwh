"""Local smoke test: drive the RMAB env in-process with an OpenAI tool-calling loop."""
import asyncio
import json
import math
import os

from dotenv import load_dotenv
from openai import OpenAI

from server import RMAB, PullParams

load_dotenv()

MODEL = "gpt-4o-mini"

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
    "sample_seed": 42,
}

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
    env = RMAB(task_spec=TASK_SPEC, secrets={"api_key": os.environ["OPENREWARD_API_KEY"]})
    await env.setup()
    try:
        client = OpenAI()
        prompt_text = "\n".join(b.text for b in env.get_prompt())
        messages = [{"role": "user", "content": prompt_text}]

        for step in range(TASK_SPEC["num_pulls"] * 3):
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="required",
            )
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_unset=True))

            print(f"[step {step}] Info: {msg}")

            if not msg.tool_calls:
                print(f"[step {step}] No tool was called: {msg.content}")
                break

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                if tc.function.name == "pull":
                    result = await env.pull(PullParams(**args))
                    print(f"[step {step}] pull(machine_id={args['machine_id']}) → reward={result.reward:.5f} finished={result.finished}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.blocks[0].text,
                    })
                    if result.finished:
                        print(f"\ncumulative: {env.cumulative_reward:.4f}")
                        return messages
                elif tc.function.name == "bash":
                    output, code = await env.sandbox.run(args["command"])
                    print(f"[step {step}] bash({args['command']!r}) → exit {code}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"{output}\n\n(exit {code})",
                    })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"unknown tool: {tc.function.name}",
                    })
        return messages
    finally:
        await env.teardown()


if __name__ == "__main__":
    messages = asyncio.run(run())
    print(messages)
