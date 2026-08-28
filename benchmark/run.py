"""Benchmark CLI skeleton.

# TODO(reconcile): wire to eda_agent.planner once available. Stage 2 builds
# datasets/planting/scoring only; there is no agent yet to run this against.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="Run planted-defect benchmarks against the EDA agent (not yet wired up).",
    )
    parser.add_argument("--model", type=str, default=None, help="LLM model identifier to benchmark")
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()
    raise NotImplementedError(
        "benchmark.run is a skeleton; eda_agent.planner does not exist yet in this build wave"
    )


if __name__ == "__main__":
    main()
