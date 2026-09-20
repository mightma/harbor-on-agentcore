#!/usr/bin/env python
"""Compare two Harbor jobs over the tasks they have in common.

    compare_jobs.py "$HARBOR_JOBS/swebv-bedrock-..." "$HARBOR_JOBS/swebv-docker-..."

Built for the one comparison this kit keeps needing: the same scaffold on two sandbox
providers. Aggregate solve rates are the least interesting output -- two runs of the
same 40 tasks can land on the same percentage while disagreeing about which ones
passed, and that disagreement is the only thing that says the sandbox changed a
result rather than the sampling.

So it reports, over the intersection only:

  per-task agreement      how many tasks got the same reward, and every task that
                          did not, with both rewards
  stage medians           environment start, agent setup, agent, verifier, total
  throughput              measured over each job in *full*, never over the
                          intersection: a subset sampled from a larger job is spread
                          across that job's whole window without its members ever
                          running together, so wall clock over the intersection
                          measures the spread of the sample and nothing else. Doing
                          that produced "effective concurrency 1.7x" for a run that
                          actually achieved 14.9x of a configured 16.
  cost                    model tokens, which is all Harbor records. Sandbox cost
                          is not in here: ACR session time bills separately and
                          docker bills as the instance you are already paying for.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from datetime import datetime
from pathlib import Path


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def _stage(result: dict, stage: str) -> float | None:
    block = result.get(stage) or {}
    start, finish = _dt(block.get("started_at")), _dt(block.get("finished_at"))
    return (finish - start).total_seconds() if start and finish else None


def load(job: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(job.glob("*/result.json")):
        d = json.loads(path.read_text())
        name = d["task_name"].split("verified__", 1)[-1]
        agent = d.get("agent_result") or {}
        out[name] = {
            "reward": ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
            "exception": (d.get("exception_info") or {}).get("exception_type"),
            "cost": agent.get("cost_usd") or 0.0,
            "in": agent.get("n_input_tokens") or 0,
            "out": agent.get("n_output_tokens") or 0,
            "started": _dt(d.get("started_at")),
            "finished": _dt(d.get("finished_at")),
            **{s: _stage(d, s) for s in
               ("environment_setup", "agent_setup", "agent_execution", "verifier")},
        }
    return out


def median(values: list[float | None]) -> float:
    real = [v for v in values if v is not None]
    return st.median(real) if real else float("nan")


def summarise(label: str, rows: list[dict]) -> dict:
    starts = [r["started"] for r in rows if r["started"]]
    ends = [r["finished"] for r in rows if r["finished"]]
    wall = (max(ends) - min(starts)).total_seconds() if starts and ends else float("nan")
    trial_sum = sum(
        (r["environment_setup"] or 0) + (r["agent_setup"] or 0)
        + (r["agent_execution"] or 0) + (r["verifier"] or 0)
        for r in rows
    )
    return {
        "label": label,
        "n": len(rows),
        "solved": sum(1 for r in rows if r["reward"] == 1.0),
        "errored": sum(1 for r in rows if r["exception"]),
        "env": median([r["environment_setup"] for r in rows]),
        "setup": median([r["agent_setup"] for r in rows]),
        "agent": median([r["agent_execution"] for r in rows]),
        "verify": median([r["verifier"] for r in rows]),
        "wall_min": wall / 60,
        "trial_sum_min": trial_sum / 60,
        "effective_conc": trial_sum / wall if wall and wall == wall else float("nan"),
        "cost": sum(r["cost"] for r in rows),
        "in": sum(r["in"] for r in rows),
        "out": sum(r["out"] for r in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_a", type=Path)
    parser.add_argument("job_b", type=Path)
    args = parser.parse_args()

    a, b = load(args.job_a), load(args.job_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        print("no tasks in common")
        return 2
    print(f"{args.job_a.name}: {len(a)} trials")
    print(f"{args.job_b.name}: {len(b)} trials")
    print(f"comparing the {len(shared)} in common\n")

    # Rewards and per-stage medians compare over the intersection; throughput has to
    # come from the whole job, for the reason in the docstring.
    sa = summarise(args.job_a.name, [a[t] for t in shared])
    sb = summarise(args.job_b.name, [b[t] for t in shared])
    fa = summarise(args.job_a.name, list(a.values()))
    fb = summarise(args.job_b.name, list(b.values()))

    rows = [
        ("solved", f"{sa['solved']}/{sa['n']} = {100*sa['solved']/sa['n']:.1f}%",
         f"{sb['solved']}/{sb['n']} = {100*sb['solved']/sb['n']:.1f}%"),
        ("errored", sa["errored"], sb["errored"]),
        ("env start, median", f"{sa['env']:.0f}s", f"{sb['env']:.0f}s"),
        ("agent setup, median", f"{sa['setup']:.0f}s", f"{sb['setup']:.0f}s"),
        ("agent, median", f"{sa['agent']:.0f}s", f"{sb['agent']:.0f}s"),
        ("verifier, median", f"{sa['verify']:.0f}s", f"{sb['verify']:.0f}s"),
        ("trial time, summed", f"{sa['trial_sum_min']:.1f} min", f"{sb['trial_sum_min']:.1f} min"),
        (f"wall clock, whole job", f"{fa['wall_min']:.1f} min ({fa['n']})",
         f"{fb['wall_min']:.1f} min ({fb['n']})"),
        ("achieved concurrency, whole job",
         f"{fa['effective_conc']:.1f}x", f"{fb['effective_conc']:.1f}x"),
        ("model cost", f"${sa['cost']:.2f}", f"${sb['cost']:.2f}"),
        ("tokens in / out", f"{sa['in']:,} / {sa['out']:,}", f"{sb['in']:,} / {sb['out']:,}"),
    ]
    width = max(len(r[0]) for r in rows)
    print(f"{'':{width}}  {'A':>22}  {'B':>22}")
    for name, va, vb in rows:
        print(f"{name:{width}}  {str(va):>22}  {str(vb):>22}")

    disagree = [(t, a[t]["reward"], b[t]["reward"]) for t in shared
                if a[t]["reward"] != b[t]["reward"]]
    print(f"\nper-task agreement: {len(shared) - len(disagree)}/{len(shared)}")
    for task, ra, rb in disagree:
        print(f"  {task:44} A={ra}  B={rb}")
    if not disagree:
        print("  the sandbox changed no outcome")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
