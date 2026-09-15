#!/usr/bin/env python
"""Tally an oracle gate job and classify its failures by root cause.

    gate_report.py <job-dir>... [--allowlist-out DIR] [--task-root DIR]

Multiple job dirs merge left to right, so a re-gate supersedes the pass it
corrects: `gate_report.py jobs/gate-1 jobs/regate --allowlist-out state/`.

Every oracle failure looks identical at the top level -- `test_patch_resolved`
asserting that not all FAIL_TO_PASS tests passed -- so `verifier/test-stdout.txt`
tells you nothing about *why*. The discriminating evidence is one level down:

    agent/oracle.txt      what `git apply` said about the golden patch
    agent/exit-code.txt   written only when the oracle command failed

Reading those splits a pile of "broken repositories" into causes that need
completely different responses:

    corrupt patch at line N   the patch was truncated in transit. A tooling bug,
                              not a repository problem. Fix and regenerate.
    No such file or directory the patch parsed but its target is not in the tree.
                              Inspect the checkout.
    Applied patch ... cleanly the patch applied and the tests still failed. The
                              only class that might be a genuinely dead reward.
    <no oracle.txt>           the trial never got that far; read trial.log.

With --allowlist-out it also writes the passing images and their task names, which
is what parts 3 and 4 restrict themselves to.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

SIGNATURES = [
    ("corrupt patch", "truncated patch (tooling bug -- regenerate tasks)"),
    ("No such file or directory", "patch target missing from the tree"),
    ("does not apply", "patch does not apply to this commit"),
    ("Applied patch", "patch applied cleanly, tests still failed"),
]


def classify(oracle_text: str | None) -> str:
    if oracle_text is None:
        return "no oracle.txt (trial failed earlier -- read trial.log)"
    for needle, label in SIGNATURES:
        if needle in oracle_text:
            return label
    first = oracle_text.strip().splitlines()
    return f"other: {first[0][:60]}" if first else "empty oracle.txt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_dir", type=Path, nargs="+",
                    help="One or more gate jobs; later ones override earlier per instance")
    ap.add_argument("--task-root", type=Path, default=None,
                    help="Task set the job ran over; needed to map images to task names")
    ap.add_argument("--allowlist-out", type=Path, default=None,
                    help="Directory to write gate_passing_{images,tasks}.txt into")
    args = ap.parse_args()

    # Merge left to right so a later re-gate wins for any instance it re-ran.
    latest: dict[str, Path] = {}
    for job in args.job_dir:
        found = sorted(job.glob("*/result.json"))
        if not found:
            print(f"no */result.json under {job}")
            return 2
        for result_path in found:
            config = json.loads((result_path.parent / "config.json").read_text())
            latest[Path(config["task"]["path"]).name] = result_path

    passed: list[str] = []
    failed: list[tuple[str, str, object]] = []
    passing_instances: list[str] = []

    for result_path in latest.values():
        trial = result_path.parent
        payload = json.loads(result_path.read_text())
        reward = ((payload.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        config = json.loads((trial / "config.json").read_text())
        instance = Path(config["task"]["path"]).name

        if reward == 1.0:
            passed.append(instance)
            passing_instances.append(instance)
            continue

        oracle_path = trial / "agent" / "oracle.txt"
        oracle = oracle_path.read_text(errors="replace") if oracle_path.exists() else None
        exception = payload.get("exception_info")
        label = (
            f"{exception.get('exception_type')} (infra -- retry before believing it)"
            if exception
            else classify(oracle)
        )
        failed.append((instance, label, reward))

    print(f"instances     : {len(latest)}")
    print(f"reward 1.0    : {len(passed)}")
    print(f"not 1.0       : {len(failed)}\n")

    if failed:
        buckets = collections.Counter(label for _, label, _ in failed)
        print("failures by cause:")
        for label, count in buckets.most_common():
            print(f"  {count:4}  {label}")
        print("\ndetail:")
        for instance, label, reward in sorted(failed):
            print(f"  {str(reward):5}  {instance}\n           {label}")

    if args.allowlist_out:
        if not args.task_root:
            print("\n--allowlist-out needs --task-root")
            return 2
        image_of: dict[str, str] = {}
        for toml_path in args.task_root.glob("*/task.toml"):
            match = re.search(r'^\s*docker_image\s*=\s*"([^"]+)"', toml_path.read_text(), re.M)
            if match:
                image_of[toml_path.parent.name] = match.group(1)

        passing_images = {image_of[i] for i in passing_instances if i in image_of}
        names = sorted(n for n, image in image_of.items() if image in passing_images)

        args.allowlist_out.mkdir(parents=True, exist_ok=True)
        (args.allowlist_out / "gate_passing_images.txt").write_text("\n".join(sorted(passing_images)) + "\n")
        (args.allowlist_out / "gate_passing_tasks.txt").write_text("\n".join(names) + "\n")
        print(f"\nwrote {len(passing_images)} images / {len(names)} task names to {args.allowlist_out}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
