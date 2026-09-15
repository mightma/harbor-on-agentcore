#!/usr/bin/env python
"""Copy the task dirs whose arm64 base image is already in the local Docker store.

    python scripts/select_pulled_tasks.py \
        --src $HARBOR_DATASETS/swebv-arm64/train \
        --dst $HARBOR_DATASETS/swebv-arm64/train-ready

Training can start before scripts/prepull_arm64.sh has worked through the whole list —
prepull is paced by Docker Hub's 100/hour, so the full 208 takes hours — but a
task whose base is not local resolves its FROM against the registry and dies as
an ImageBuildError. In RL that is worse than in an eval: the trial is a rollout,
and a whole GRPO step waits on it.

So the training set is the intersection of "task generated" and "base image
pulled". Task dirs are a few KB, so they are copied rather than symlinked, which
keeps them independent of whatever the source directory does next.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def local_base_images() -> set[str]:
    out = subprocess.run(
        ["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def base_image_of(task_dir: Path) -> str | None:
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        return None
    for line in dockerfile.read_text().splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("FROM "):
            return stripped.split(None, 1)[1].strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument(
        "--limit", type=int, default=None, help="Keep at most N tasks (sorted by id)"
    )
    parser.add_argument("--clean", action="store_true", help="Empty --dst first")
    args = parser.parse_args()

    have = local_base_images()
    if args.clean and args.dst.exists():
        shutil.rmtree(args.dst)
    args.dst.mkdir(parents=True, exist_ok=True)

    ready, missing = [], []
    for task_dir in sorted(p for p in args.src.iterdir() if p.is_dir()):
        image = base_image_of(task_dir)
        if image and image in have:
            ready.append(task_dir)
        else:
            missing.append(task_dir.name)

    if args.limit:
        ready = ready[: args.limit]

    for task_dir in ready:
        target = args.dst / task_dir.name
        if target.exists():
            continue
        shutil.copytree(task_dir, target)

    print(f"{len(ready)} tasks ready in {args.dst}")
    print(f"{len(missing)} still waiting on a base image pull")
    if missing[:5]:
        print("  e.g.", ", ".join(missing[:5]))


if __name__ == "__main__":
    main()
