#!/usr/bin/env python
"""Summarise one or two Harbor jobs: rewards, phase timings, tokens, cost.

    python run/summarize.py <job-dir> [<job-dir> ...]
    python run/summarize.py --compare <job-a> <job-b>

Per-trial numbers come from each trial's ``result.json``, so this works on a job
that is still running (finished trials only) as well as a finished one. With
``--compare`` the second job's rewards are joined onto the first by task name,
which is how the AgentCore and Docker runs of the same slice are lined up.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def _seconds(phase: dict | None) -> float | None:
    if not phase or not phase.get("started_at") or not phase.get("finished_at"):
        return None
    start = datetime.fromisoformat(phase["started_at"])
    end = datetime.fromisoformat(phase["finished_at"])
    return (end - start).total_seconds()


def load(job_dir: Path) -> list[dict]:
    trials = []
    for result_path in sorted(job_dir.glob("*/result.json")):
        result = json.loads(result_path.read_text())
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        agent = result.get("agent_result") or {}
        exception = result.get("exception_info") or None
        trials.append(
            {
                "task": result.get("task_name"),
                "reward": rewards.get("reward"),
                "exception": (exception or {}).get("exception_type"),
                "message": (exception or {}).get("exception_message", "")[:120],
                "env_sec": _seconds(result.get("environment_setup")),
                "setup_sec": _seconds(result.get("agent_setup")),
                "agent_sec": _seconds(result.get("agent_execution")),
                "verify_sec": _seconds(result.get("verifier")),
                "total_sec": _seconds(result),
                "input_tokens": agent.get("n_input_tokens"),
                "cache_tokens": agent.get("n_cache_tokens"),
                "output_tokens": agent.get("n_output_tokens"),
                "cost_usd": agent.get("cost_usd"),
            }
        )
    return trials


def _fmt(value, width=7, digits=1) -> str:
    if value is None:
        return "-".rjust(width)
    if isinstance(value, float):
        return f"{value:>{width}.{digits}f}"
    return f"{value:>{width}}"


def report(job_dir: Path, trials: list[dict], other: dict[str, float | None] | None) -> None:
    print(f"\n=== {job_dir.name}  ({len(trials)} trials)")
    header = f"{'task':38} {'reward':>7} {'env':>7} {'setup':>7} {'agent':>7} {'verify':>7} {'total':>7} {'cost$':>7}"
    if other is not None:
        header += f" {'other':>7}"
    print(header)
    for trial in sorted(trials, key=lambda t: t["task"] or ""):
        line = (
            f"{(trial['task'] or '?'):38} {_fmt(trial['reward'], 7, 2)} "
            f"{_fmt(trial['env_sec'], 7, 0)} {_fmt(trial['setup_sec'], 7, 0)} "
            f"{_fmt(trial['agent_sec'], 7, 0)} {_fmt(trial['verify_sec'], 7, 0)} "
            f"{_fmt(trial['total_sec'], 7, 0)} {_fmt(trial['cost_usd'], 7, 2)}"
        )
        if other is not None:
            line += f" {_fmt(other.get(trial['task']), 7, 2)}"
        if trial["exception"]:
            line += f"  !{trial['exception']}: {trial['message']}"
        print(line)

    scored = [t for t in trials if t["reward"] is not None]
    solved = [t for t in scored if t["reward"] and t["reward"] > 0]
    errored = [t for t in trials if t["exception"]]
    cost = sum(t["cost_usd"] or 0 for t in trials)
    tokens = (
        sum(t["input_tokens"] or 0 for t in trials),
        sum(t["cache_tokens"] or 0 for t in trials),
        sum(t["output_tokens"] or 0 for t in trials),
    )
    print(
        f"\n  scored {len(scored)}/{len(trials)}   solved {len(solved)}/{len(trials)}"
        f" = {len(solved) / max(1, len(trials)):.1%}   errored {len(errored)}"
    )
    print(f"  cost ${cost:.2f}   tokens in={tokens[0]:,} cached={tokens[1]:,} out={tokens[2]:,}")
    for phase in ("env_sec", "setup_sec", "agent_sec", "verify_sec", "total_sec"):
        values = [t[phase] for t in trials if t[phase] is not None]
        if values:
            values.sort()
            print(
                f"  {phase:10} median {values[len(values) // 2]:7.0f}s"
                f"  max {values[-1]:7.0f}s  sum {sum(values) / 60:7.1f}min"
            )
    if errored:
        print("  exceptions:")
        for trial in errored:
            print(f"    {trial['task']}: {trial['exception']}: {trial['message']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("jobs", nargs="+", type=Path)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="join the second job's rewards onto the first as an 'other' column",
    )
    args = parser.parse_args()

    loaded = [(job, load(job)) for job in args.jobs]
    other = None
    if args.compare and len(loaded) > 1:
        other = {t["task"]: t["reward"] for t in loaded[1][1]}
    for index, (job, trials) in enumerate(loaded):
        report(job, trials, other if index == 0 else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
