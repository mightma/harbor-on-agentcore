#!/usr/bin/env python
"""Wrap each built arm64 SWE-smith repo image into a task-ready shared image.

The upstream SWE-smith Harbor adapter puts the per-task work in the *task's*
``environment/Dockerfile``::

    FROM {docker_image}
    RUN apt-get update && apt-get install -y git
    RUN curl -LsSf https://astral.sh/uv/.../install.sh | sh
    RUN mkdir -p /logs
    RUN git fetch && git checkout {instance_id}      # <-- per task

Because ``{instance_id}`` is in the Dockerfile, every one of the ~52k tasks gets
its own environment content hash, so every task pushes its own image and (on
AgentCore Runtime) deploys its own agent runtime. That is 52k runtimes against a
1,000-per-account quota, for a set of images that differ only by which git branch
is checked out.

This script moves everything that is *common* into one image per repository:

    apt: git, curl        (the adapter's Dockerfile installed these per task)
    uv                    (same)
    /logs                 (same)
    git fetch --all       (bake every task branch in, so the per-task checkout
                           needs no network at rollout time)
    OPENBLAS_CORETYPE     (AgentCore's microVM reports Neoverse-V1 but masks SVE;
                           OpenBLAS picks its kernel from the MIDR and `import
                           numpy` dies with SIGILL. See the part 4 README section 3.)

What is left per task -- ``git checkout <instance_id>`` -- is then carried by
task.toml, which Harbor does *not* include in the environment content hash, so
all tasks of one repository converge on one image and one runtime.

    scripts/prepare_swesmith_images.py --list
    scripts/prepare_swesmith_images.py --concurrency 8 --push
"""

from __future__ import annotations

import argparse
import os
import concurrent.futures
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PREPARED_SUFFIX = "prepared-arm64"
UV_VERSION = "0.7.13"

DOCKERFILE = f"""\
FROM {{base}}

# The adapter's per-task Dockerfile installed these; they are identical for every
# task of a repository, so they belong in the shared image.
RUN apt-get update \\
 && apt-get install -y --no-install-recommends git curl ca-certificates \\
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/{UV_VERSION}/install.sh | sh
RUN mkdir -p /logs

WORKDIR /testbed

# Bake every task branch into the image so the per-task checkout is a local
# operation at rollout time: no git remote, no network, no Docker Hub quota.
RUN git fetch --all --tags --prune && git branch -r | wc -l

# AgentCore Runtime's vCPU advertises Neoverse-V1 (CPU part 0xd40) but masks SVE
# out of /proc/cpuinfo Features. OpenBLAS selects its kernel from the MIDR, picks
# an SVE kernel, and `import numpy` exits with SIGILL -- which surfaces as a
# reward of 0 on every numpy-using task, even with the golden patch applied.
ENV OPENBLAS_CORETYPE=ARMV8
"""


def sh(args: list[str], timeout: int | None = None) -> tuple[int, str]:
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def built_images() -> dict[str, str]:
    """Locally built arm64 repo images, keyed by the swesmith profile key."""
    _, out = sh(["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"])
    found: dict[str, str] = {}
    for line in out.splitlines():
        name = line.strip()
        if ".arm64." not in name or "swesmith" not in name:
            continue
        if PREPARED_SUFFIX in name or ".dkr.ecr." in name or "/swesmith-arm64" in name:
            continue
        stem = name.split("/")[-1].split(":")[0]
        if not stem.startswith("swesmith.arm64."):
            continue
        found[stem[len("swesmith.arm64.") :]] = name.split(":")[0]
    return dict(sorted(found.items()))


def ecr_registry(region: str) -> str:
    code, out = sh(
        ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"]
    )
    if code != 0:
        raise SystemExit(f"aws sts get-caller-identity failed: {out}")
    return f"{out.strip()}.dkr.ecr.{region}.amazonaws.com"


def ecr_login(registry: str, repository: str, region: str) -> None:
    if sh(["aws", "ecr", "describe-repositories", "--repository-names", repository,
           "--region", region])[0] != 0:
        sh(["aws", "ecr", "create-repository", "--repository-name", repository,
            "--region", region])
    code, out = sh(["aws", "ecr", "get-login-password", "--region", region])
    if code != 0:
        raise SystemExit(f"ecr get-login-password failed: {out}")
    p = subprocess.run(
        ["docker", "login", "--username", "AWS", "--password-stdin", registry],
        input=out.strip(), text=True, capture_output=True,
    )
    if p.returncode != 0:
        raise SystemExit(f"docker login failed: {p.stderr}")


def prepare_one(key: str, base: str, target: str, timeout: int, push: bool) -> dict:
    started = time.time()
    with tempfile.TemporaryDirectory() as ctx:
        Path(ctx, "Dockerfile").write_text(DOCKERFILE.format(base=base))
        code, out = sh(
            ["docker", "build", "--platform", "linux/arm64", "--provenance=false",
             "-t", target, "-f", str(Path(ctx, "Dockerfile")), ctx],
            timeout=timeout,
        )
    if code != 0:
        return {"key": key, "status": "error", "sec": int(time.time() - started),
                "error": out.strip().splitlines()[-1] if out.strip() else "build failed"}

    code, arch = sh(["docker", "image", "inspect", target, "--format", "{{.Architecture}}"])
    if arch.strip() != "arm64":
        return {"key": key, "status": "error", "sec": int(time.time() - started),
                "error": f"built image is {arch.strip()!r}, not arm64"}

    _, branches = sh(
        ["docker", "run", "--rm", "--platform", "linux/arm64", target,
         "bash", "-lc", "cd /testbed && git branch -r | wc -l"]
    )
    _, size = sh(["docker", "image", "inspect", target, "--format", "{{.Size}}"])

    if push:
        code, out = sh(["docker", "push", target], timeout=timeout)
        if code != 0:
            return {"key": key, "status": "error", "sec": int(time.time() - started),
                    "error": f"push failed: {out.strip().splitlines()[-1]}"}

    return {
        "key": key,
        "status": "ok",
        "image": target,
        "sec": int(time.time() - started),
        "size_mb": int(int(size.strip() or 0) / 1e6),
        "branches": int(re.sub(r"\D", "", branches) or 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    ap.add_argument(
        "--ecr-repository",
        default=os.environ.get("SWESMITH_ECR_REPOSITORY", "swesmith-arm64"),
    )
    ap.add_argument("--repos", default=None, help="Comma-separated profile keys")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(
            os.environ.get("SWESMITH_BUILD_ROOT")
            or Path(os.environ.get("KIT_WORK_DIR", ".")) / "swesmith-arm64"
        )
        / "prepared.json",
    )
    args = ap.parse_args()

    images = built_images()
    if args.repos:
        wanted = {k.strip() for k in args.repos.split(",") if k.strip()}
        missing = wanted - set(images)
        if missing:
            raise SystemExit(f"no local arm64 image for: {sorted(missing)}")
        images = {k: v for k, v in images.items() if k in wanted}

    if args.list:
        print(f"{len(images)} built arm64 repo images:")
        for key, base in images.items():
            print(f"  {key:56s} {base}")
        return
    if not images:
        raise SystemExit("no built arm64 repo images found; run build_swesmith_arm64.sh first")

    registry = ecr_registry(args.region)
    if args.push:
        ecr_login(registry, args.ecr_repository, args.region)

    targets = {
        key: f"{registry}/{args.ecr_repository}:{key.replace('/', '_')}-{PREPARED_SUFFIX}"
        for key in images
    }

    print(f"preparing {len(images)} images, concurrency {args.concurrency}, push={args.push}")
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {
            pool.submit(prepare_one, k, images[k], targets[k], args.timeout, args.push): k
            for k in images
        }
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            results.append(r)
            tail = (f"{r.get('size_mb')} MB, {r.get('branches')} branches"
                    if r["status"] == "ok" else r.get("error", ""))
            print(f"  {r['status']:<7s} {r['key']:<52s} {r['sec']:>5d}s  {tail}", flush=True)

    ok = [r for r in results if r["status"] == "ok"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sorted(results, key=lambda r: r["key"]), indent=2) + "\n")
    print(f"\n{len(ok)}/{len(results)} prepared; mapping written to {args.output}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
