from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from .tools import MLWorkflowTools


SYSTEM_PROMPT = """
You are an ML workflow orchestration agent. Your job is to complete tabular
classification workflows using only the available tools.

Supported datasets: iris, penguins.
Supported models: logistic regression, decision tree, knn.

Available tools:
- model_memory
- dataset_loader
- dataset_preprocessing
- train_model
- evaluate_model
- visualize_results

For each turn, return exactly one JSON object and no markdown.
To call a tool:
{"action": "tool_name", "input": "tool input"}

When the request is complete:
{"final": "concise completion summary"}

Use canonical tool inputs. Examples:
- dataset_loader -> "iris" or "penguins"
- dataset_preprocessing -> "iris" or "penguins"
- train_model -> "logistic regression", "decision tree", or "knn"
- evaluate_model -> ""
- visualize_results -> ""

Do not invent tool results. Use one tool action per turn and wait for its
observation before choosing the next action.
""".strip()


@dataclass
class AgentRun:
    final: str
    trace: list[dict[str, str]] = field(default_factory=list)


class GroqMLAgent:
    """LLM-driven controller around deterministic ML workflow tools."""

    def __init__(
        self,
        tools: MLWorkflowTools,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add a valid key."
            )

        self.client = Groq(api_key=key)
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.tools = tools
        self.registry = tools.tool_registry()

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start : end + 1])
            raise ValueError(f"Agent returned invalid JSON: {text}")

    def run(self, question: str, max_steps: int = 10) -> AgentRun:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        trace: list[dict[str, str]] = []

        for _ in range(max_steps):
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=300,
            )
            raw = completion.choices[0].message.content or ""
            response = self._parse_json(raw)

            if "final" in response:
                return AgentRun(final=str(response["final"]), trace=trace)

            action = str(response.get("action", "")).strip()
            action_input = str(response.get("input", "")).strip()
            if action not in self.registry:
                observation = f"Unknown tool '{action}'. Available: {', '.join(self.registry)}."
            else:
                try:
                    observation = str(self.registry[action](action_input))
                except Exception as exc:
                    observation = f"Tool error: {type(exc).__name__}: {exc}"

            trace.append(
                {"action": action, "input": action_input, "observation": observation}
            )
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": f"Tool observation:\n{observation}\nContinue with one JSON action or final response.",
                }
            )

        return AgentRun(
            final=f"Stopped after {max_steps} steps before receiving a final response.",
            trace=trace,
        )
