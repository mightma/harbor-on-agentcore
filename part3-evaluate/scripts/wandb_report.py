#!/usr/bin/env python
"""Push a finished (or running) Harbor job's per-trial results to Weights & Biases.

    python run/wandb_report.py <job-dir> --run-name eval-Qwen3.5-4B-acr
    python run/wandb_report.py <job-dir> --project my-project --tag baseline

Harbor writes rewards and phase timings per trial; SkyRL logs its own training
metrics to wandb directly. This closes the gap for the evaluation half, so a
baseline eval and a post-RL eval land in the same workspace as the training run
and can be put on one axis.

Reads WANDB_API_KEY from the environment or from rl/.env. Parsing is deliberately
the same shape as run/summarize.py so the two never disagree about what a trial's
reward was.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from summarize import load  # same directory; keeps one parser for trial results

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_api_key() -> None:
    if os.environ.get("WANDB_API_KEY"):
        return
    env_file = REPO_ROOT / "rl" / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("WANDB_API_KEY="):
            os.environ["WANDB_API_KEY"] = line.split("=", 1)[1].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--project", default="harbor-acr-swebench")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--tag", action="append", default=[], dest="tags")
    parser.add_argument(
        "--notes", default=None, help="Free-text note stored on the wandb run"
    )
    args = parser.parse_args()

    _load_api_key()
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("No WANDB_API_KEY in the environment or rl/.env")

    trials = load(args.job_dir)
    if not trials:
        raise SystemExit(f"No trial results under {args.job_dir}")

    import wandb

    scored = [t for t in trials if t["reward"] is not None]
    solved = [t for t in scored if t["reward"] and t["reward"] > 0]
    # A trial that raised is not the same as a trial that scored 0: the first is a
    # broken rollout, the second is a real "did not solve". Reporting only the
    # mean over scored trials hides how many never got that far, so both go up.
    errored = [t for t in trials if t["exception"]]

    def _mean(values: list[float]) -> float | None:
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    summary = {
        "n_trials": len(trials),
        "n_scored": len(scored),
        "n_errored": len(errored),
        "n_solved": len(solved),
        "resolve_rate": (len(solved) / len(scored)) if scored else None,
        "mean_reward": _mean([t["reward"] for t in scored]),
        "mean_agent_sec": _mean([t["agent_sec"] for t in trials]),
        "mean_env_sec": _mean([t["env_sec"] for t in trials]),
        "mean_verify_sec": _mean([t["verify_sec"] for t in trials]),
        "mean_total_sec": _mean([t["total_sec"] for t in trials]),
        "total_output_tokens": sum(t["output_tokens"] or 0 for t in trials),
        "total_input_tokens": sum(t["input_tokens"] or 0 for t in trials),
    }

    run = wandb.init(
        project=args.project,
        name=args.run_name or args.job_dir.name,
        tags=args.tags,
        notes=args.notes,
        config={"job_dir": str(args.job_dir), "job_name": args.job_dir.name},
    )

    columns = [
        "task",
        "reward",
        "exception",
        "env_sec",
        "setup_sec",
        "agent_sec",
        "verify_sec",
        "total_sec",
        "input_tokens",
        "output_tokens",
    ]
    table = wandb.Table(columns=columns)
    for trial in sorted(trials, key=lambda t: t["task"] or ""):
        table.add_data(*[trial.get(column) for column in columns])

    exception_counts: dict[str, int] = {}
    for trial in errored:
        exception_counts[trial["exception"]] = (
            exception_counts.get(trial["exception"], 0) + 1
        )

    run.log({"trials": table, **summary})
    for name, count in exception_counts.items():
        run.summary[f"exception/{name}"] = count
    run.summary.update(summary)
    print(f"logged {len(trials)} trials to {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
