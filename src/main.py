from __future__ import annotations

import argparse

from dotenv import load_dotenv

from .agent import GroqMLAgent
from .tools import MLWorkflowTools


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an LLM-driven machine-learning workflow agent."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="Train a logistic regression model on the Iris dataset, evaluate it, and visualize the results.",
    )
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()

    load_dotenv()
    tools = MLWorkflowTools(output_dir=args.output_dir)
    agent = GroqMLAgent(tools)
    result = agent.run(args.question, max_steps=args.max_steps)

    print("\nAgent trace")
    print("=" * 60)
    for i, step in enumerate(result.trace, start=1):
        print(f"{i}. {step['action']}({step['input']!r})")
        print(step["observation"])
        print("-" * 60)

    print("\nFinal")
    print("=" * 60)
    print(result.final)


if __name__ == "__main__":
    main()
