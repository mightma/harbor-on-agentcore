#!/usr/bin/env python
"""Derive every list in data/ from something you can observe.

    make_lists.py --selfbuilt            # ask ECR which instance images exist
    make_lists.py --from-gate <job>...   # read oracle gate jobs
    make_lists.py --runnable             # combine the above with the probe output

Why this exists: a committed list of instance ids is a claim, and a claim needs a
command that reproduces it. `swebv-arm64-instances.txt` has one
(`probe_arm64_images.py --out`) and so does the build (`build_images.py --push`), but
the derived lists used to be produced by hand, which makes them unauditable the
moment anything changes. Each flag below owns exactly one file:

    --selfbuilt    swebv-arm64-selfbuilt.txt     from ECR: which images were built
    --from-gate    swebv-arm64-gated.txt         from the gate: reward 1.0
                   swebv-arm64-undeployable.txt  from the gate: maxImageSizeMb refusal
                   swebv-arm64-selfbuilt-gated.txt  the intersection, for reproducing
                                                    a number measured on it
    --runnable     swebv-arm64-runnable.txt      published ∪ selfbuilt − undeployable

`--runnable` is a *claim about deployability*, and it is only measured for the part it
can measure. An instance built here and refused by CreateAgentRuntime is known
undeployable. An instance with a published image is *assumed* deployable, because
nothing has checked every published image against ACR's 2048 MB ceiling -- and the
matplotlib images built here proved that ceiling is reachable. `--check-published`
verifies that assumption for the published half, one registry request per instance.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TAG_PREFIX = "sweb.eval.arm64."
ACR_IMAGE_LIMIT_MB = 2048


def write(name: str, ids: set[str], note: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / name).write_text("\n".join(sorted(ids)) + "\n")
    print(f"  {name:36} {len(ids):4}  {note}")


def read(name: str) -> set[str]:
    path = DATA / name
    if not path.exists():
        raise SystemExit(
            f"{path} not found. It comes from:\n"
            "  swebv-arm64-instances.txt    probe_arm64_images.py --out\n"
            "  swebv-arm64-selfbuilt.txt    make_lists.py --selfbuilt\n"
            "  swebv-arm64-undeployable.txt make_lists.py --from-gate <job>..."
        )
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def ecr_instance_ids(repository: str, region: str) -> set[str]:
    """Instance ids whose image is in ECR, read from the tags themselves."""
    ids: set[str] = set()
    token = None
    while True:
        cmd = [
            "aws", "ecr", "describe-images", "--repository-name", repository,
            "--region", region, "--max-items", "100", "--output", "json",
        ]
        if token:
            cmd += ["--starting-token", token]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            # Before the first push the repository does not exist yet, which is a
            # normal state for a from-scratch run rather than an error: the answer to
            # "which images were built" is legitimately "none".
            if "RepositoryNotFoundException" in out.stderr:
                print(f"  (ECR repository {repository} does not exist yet)")
                return set()
            raise SystemExit(f"aws ecr describe-images failed: {out.stderr.strip()}")
        payload = json.loads(out.stdout or "{}")
        for image in payload.get("imageDetails", []):
            for tag in image.get("imageTags", []):
                if tag.startswith(TAG_PREFIX):
                    ids.add(tag[len(TAG_PREFIX):].removesuffix("-latest"))
        token = payload.get("NextToken")
        if not token:
            break
    return ids


def from_gate(job_dirs: list[Path]) -> tuple[set[str], set[str]]:
    """(gated, undeployable) from oracle gate jobs, later jobs winning.

    A re-gate after a fix supersedes the pass it corrects, which is why the jobs are
    applied in the order given rather than merged as sets.
    """
    reward: dict[str, float | None] = {}
    undeployable: set[str] = set()
    for job in job_dirs:
        for result_path in sorted(job.glob("*/result.json")):
            result = json.loads(result_path.read_text())
            instance = result["task_name"].split("verified__", 1)[-1]
            rewards = (result.get("verifier_result") or {}).get("rewards") or {}
            reward[instance] = rewards.get("reward")
            info = result.get("exception_info") or {}
            message = info.get("exception_message") or ""
            if "maxImageSizeMb" in message:
                undeployable.add(instance)
            elif instance in undeployable:
                # A later job got a session for it, so the refusal was fixed (a
                # slimmed image) rather than permanent.
                undeployable.discard(instance)
    return {i for i, r in reward.items() if r == 1.0}, undeployable


def check_published(ids: set[str], limit_mb: int) -> set[str]:
    """Which published images exceed ACR's compressed-size ceiling.

    Costs one registry manifest request per instance and Docker Hub counts those
    against the anonymous limit of 100/hour/IP, so log in first.
    """
    sys.path.insert(0, str(HERE))
    from probe_arm64_images import probe  # noqa: PLC0415

    over: set[str] = set()
    for n, instance in enumerate(sorted(ids), 1):
        _, exists, size = probe(instance, want_size=True)
        if exists and size and size / 1e6 > limit_mb:
            over.add(instance)
            print(f"    over limit: {instance} {size / 1e6:.0f} MB")
        if n % 50 == 0:
            print(f"    ...{n}/{len(ids)}")
    return over


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--selfbuilt", action="store_true")
    parser.add_argument("--from-gate", nargs="+", type=Path, default=None, metavar="JOB")
    parser.add_argument("--runnable", action="store_true")
    parser.add_argument(
        "--check-published",
        action="store_true",
        help="Also probe every published image's size and treat anything over the "
        "ceiling as undeployable. One registry request per instance.",
    )
    parser.add_argument(
        "--ecr-repository",
        default=os.environ.get("SWEBENCH_ECR_REPOSITORY", "swebv-arm64"),
    )
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    parser.add_argument("--limit-mb", type=int, default=ACR_IMAGE_LIMIT_MB)
    args = parser.parse_args()

    if not (args.selfbuilt or args.from_gate or args.runnable):
        parser.error("pick at least one of --selfbuilt / --from-gate / --runnable")

    if args.selfbuilt:
        ids = ecr_instance_ids(args.ecr_repository, args.region)
        write("swebv-arm64-selfbuilt.txt", ids,
              f"images tagged {TAG_PREFIX}* in ECR {args.ecr_repository}")

    if args.from_gate:
        gated, undeployable = from_gate(args.from_gate)
        write("swebv-arm64-gated.txt", gated, "oracle reward 1.0")
        write("swebv-arm64-undeployable.txt", undeployable,
              f"CreateAgentRuntime refused: over {args.limit_mb} MB")
        selfbuilt_path = DATA / "swebv-arm64-selfbuilt.txt"
        if selfbuilt_path.exists():
            write("swebv-arm64-selfbuilt-gated.txt", gated & read("swebv-arm64-selfbuilt.txt"),
                  "gated ∩ selfbuilt")

    if args.runnable:
        published = read("swebv-arm64-instances.txt")
        selfbuilt = read("swebv-arm64-selfbuilt.txt") if (DATA / "swebv-arm64-selfbuilt.txt").exists() else set()
        undeployable = (
            read("swebv-arm64-undeployable.txt")
            if (DATA / "swebv-arm64-undeployable.txt").exists()
            else set()
        )
        if args.check_published:
            print(f"  probing {len(published)} published images for size...")
            over = check_published(published, args.limit_mb)
            undeployable |= over
            write("swebv-arm64-undeployable.txt", undeployable,
                  "gate refusals + published images over the ceiling")
        runnable = (published | selfbuilt) - undeployable
        note = "published ∪ selfbuilt − undeployable"
        if not args.check_published:
            note += "  (published half assumed deployable, not measured)"
        write("swebv-arm64-runnable.txt", runnable, note)


if __name__ == "__main__":
    main()
