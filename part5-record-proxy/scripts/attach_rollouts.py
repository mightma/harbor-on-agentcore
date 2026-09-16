#!/usr/bin/env python
"""Attach proxy-recorded rollouts to the trials they came from.

    scripts/attach_rollouts.py "$HARBOR_JOBS/<job>" --record "$HARBOR_JOBS/rollouts.jsonl"
    scripts/attach_rollouts.py "$HARBOR_JOBS/<job>" --record ... --dry-run

This is the seam the design called the real unknown. Harbor populates
``agent_result.rollout_details`` from its *own* LLM layer, so for an installed
harness the field is ``null`` -- the trial ran, the tokens existed, and nothing
wrote them down. ``record_proxy.py`` writes them down; this puts them where a
consumer (SkyRL, or ``summarize.py``) already looks.

Doing it here rather than inside Harbor is deliberate for a prototype: it needs no
Harbor change, and it makes the matching *auditable* -- every trial is reported as
matched or not, with the reason. The upstream version of this would set the field
during the trial instead, from a proxy handle the agent knows about; that is an RFC,
not a kit feature (see DESIGN-generalization.md).

Matching, in order
------------------
1. **Session id.** If the trial's ``config.json`` sets ``agent.kwargs.session_id``
   and the proxy saw that id in a header, they belong together. This is the SkyRL
   case, where each rollout gets its own id, and it is exact.
2. **First user message.** Otherwise, compare the trial's own first user turn
   (``agent/trajectory.json``, the ATIF trajectory every installed harness writes)
   with the first user message the proxy hashed. For SWE tasks the instruction
   contains the problem statement, so this is unique per instance in practice --
   but it is a heuristic, and a job with duplicate instructions (``n_attempts > 1``)
   will produce ambiguous matches, which are reported and skipped rather than
   guessed.

Anything unmatched in either direction is printed. A silent partial attach would be
the worst possible outcome: it trains on a trajectory belonging to another task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def load_turns(record_path: Path) -> dict[str, list[dict]]:
    """Group recorded turns by rollout, ordered by turn number."""
    by_rollout: dict[str, list[dict]] = defaultdict(list)
    for line in record_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        by_rollout[entry["rollout_id"]].append(entry)
    for turns in by_rollout.values():
        turns.sort(key=lambda t: t["turn"])
    return dict(by_rollout)


def rollout_detail(turns: list[dict]) -> dict:
    """Turns -> one RolloutDetail (harbor/models/agent/rollout_detail.py).

    Every field is a list indexed by turn: prompt token ids for that turn including
    the history, the completion's token ids, and the logprobs of those completion
    tokens. Turns whose token ids are missing are dropped, and the count is
    reported -- a trajectory with a hole is not a trajectory.
    """
    usable = [
        turn
        for turn in turns
        if turn.get("prompt_token_ids") and turn.get("completion_token_ids")
    ]
    return {
        "prompt_token_ids": [t["prompt_token_ids"] for t in usable],
        "completion_token_ids": [t["completion_token_ids"] for t in usable],
        "logprobs": [t["logprobs"] for t in usable],
        "extra": {
            "proxy_rollout_id": [t["rollout_id"] for t in usable],
            "proxy_turn": [t["turn"] for t in usable],
        },
    }


def trial_dirs(job_dir: Path) -> list[Path]:
    return sorted(p for p in job_dir.iterdir() if p.is_dir() and (p / "result.json").exists())


def trial_session_id(trial: Path) -> str | None:
    config_path = trial / "config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    kwargs = ((config.get("agent") or {}).get("kwargs")) or {}
    value = kwargs.get("session_id")
    return str(value) if value else None


def trial_first_user_sha(trial: Path) -> str | None:
    """sha256 of the trial's own first user message, from the ATIF trajectory."""
    trajectory_path = trial / "agent" / "trajectory.json"
    if not trajectory_path.exists():
        return None
    try:
        trajectory = json.loads(trajectory_path.read_text())
    except json.JSONDecodeError:
        return None
    for step in trajectory.get("steps") or []:
        if step.get("source") == "user":
            message = step.get("message") or ""
            return hashlib.sha256(message.encode("utf-8", "replace")).hexdigest()
    return None


def match(trials: list[Path], rollouts: dict[str, list[dict]]) -> tuple[dict, list, list]:
    """Return (trial -> rollout_id, unmatched trials, unmatched rollout ids)."""
    by_session: dict[str, str] = {}
    by_first_user: dict[str, list[str]] = defaultdict(list)
    for rollout_id, turns in rollouts.items():
        # A forked rollout shares its first user message with the trial's main
        # conversation, so including it would make every forked trial "ambiguous".
        # Forks are reported by the caller and deliberately left unattached: they
        # are not linear trajectories, which is the whole point of flagging them.
        if any(turn.get("forked") for turn in turns):
            continue
        session_id = turns[0].get("session_id")
        if session_id and rollout_id == session_id:
            by_session[session_id] = rollout_id
        by_first_user[turns[0]["first_user_sha256"]].append(rollout_id)

    pairs: dict[Path, str] = {}
    unmatched_trials: list[tuple[Path, str]] = []
    for trial in trials:
        session_id = trial_session_id(trial)
        if session_id and session_id in by_session:
            pairs[trial] = by_session[session_id]
            continue
        sha = trial_first_user_sha(trial)
        if sha is None:
            unmatched_trials.append((trial, "no ATIF trajectory and no session id"))
            continue
        candidates = [r for r in by_first_user.get(sha, []) if r not in pairs.values()]
        if len(candidates) == 1:
            pairs[trial] = candidates[0]
        elif not candidates:
            unmatched_trials.append((trial, "no rollout with this first user message"))
        else:
            unmatched_trials.append(
                (trial, f"ambiguous: {len(candidates)} rollouts share this instruction")
            )
    unmatched_rollouts = [r for r in rollouts if r not in pairs.values()]
    return pairs, unmatched_trials, unmatched_rollouts


def attach(trial: Path, detail: dict, backup: bool) -> int:
    """Write rollout_details into a trial's result.json. Returns the turn count."""
    result_path = trial / "result.json"
    result = json.loads(result_path.read_text())
    agent_result = result.setdefault("agent_result", {})
    agent_result["rollout_details"] = [detail]
    if backup:
        original = trial / "result.json.orig"
        if not original.exists():
            original.write_text(result_path.read_text())
    result_path.write_text(json.dumps(result, indent=1) + "\n")
    return len(detail["prompt_token_ids"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not keep result.json.orig (kept by default: this rewrites a "
        "measured artifact)",
    )
    args = parser.parse_args()

    rollouts = load_turns(args.record)
    trials = trial_dirs(args.job_dir)
    print(f"{len(trials)} trial(s), {len(rollouts)} recorded rollout(s)")

    forked = [r for r, turns in rollouts.items() if any(t.get("forked") for t in turns)]
    if forked:
        print(
            f"  {len(forked)} rollout(s) contain a forked turn (subagent, or the "
            f"client rewrote its history): {forked[:3]}"
        )
        print("  those are NOT linear trajectories; do not train on them unsegmented")

    pairs, unmatched_trials, unmatched_rollouts = match(trials, rollouts)
    attached = 0
    for trial, rollout_id in sorted(pairs.items()):
        detail = rollout_detail(rollouts[rollout_id])
        dropped = len(rollouts[rollout_id]) - len(detail["prompt_token_ids"])
        note = f", {dropped} turn(s) dropped for missing token ids" if dropped else ""
        if args.dry_run:
            print(f"  would attach {rollout_id} -> {trial.name}"
                  f" ({len(detail['prompt_token_ids'])} turns{note})")
            continue
        turns = attach(trial, detail, backup=not args.no_backup)
        attached += 1
        print(f"  attached {rollout_id} -> {trial.name} ({turns} turns{note})")

    for trial, reason in unmatched_trials:
        print(f"  UNMATCHED trial {trial.name}: {reason}")
    for rollout_id in unmatched_rollouts:
        head = rollouts[rollout_id][0]["first_user_head"][:60].replace("\n", " ")
        print(f"  UNMATCHED rollout {rollout_id}: {head!r}")

    if not args.dry_run:
        print(f"\n{attached}/{len(trials)} trial(s) now carry rollout_details")
    raise SystemExit(1 if (unmatched_trials or unmatched_rollouts) else 0)


if __name__ == "__main__":
    main()
