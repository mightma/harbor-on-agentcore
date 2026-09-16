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

Optionally the harness can be baked in too (``--bake-harness``), which removes
Harbor's per-trial agent setup. See HARNESS_LAYERS below for what each recipe
installs, what Harbor then skips, and why terminus-2 -- which installs no agent at
all -- still has something to bake.

    scripts/prepare_swesmith_images.py --bake-harness terminus-2 --render-only
    scripts/prepare_swesmith_images.py --bake-harness claude-code --harness-version 2.1.272
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

# --- optional harness layers (--bake-harness) ----------------------------------
#
# Harbor sets the agent up inside the sandbox on every trial, and that cost is
# per-trial forever: **\[measured\]** on the 70-task SWE-bench Verified pass,
# `agent_setup` was a 12.0 s median for terminus-2 and 51.3 s for claude-code.
# Baking makes Harbor's own idempotence checks short-circuit, so the layer below
# is exchanged for a one-off image cost.
#
# What each harness checks -- these are the contracts the recipes have to satisfy,
# read out of harbor 0.23.0:
#
#   terminus-2      TmuxSession._install_recording_tools() execs `tmux -V` and
#                   (because record_terminal_session defaults to True)
#                   `asciinema --version` as root, and skips the apt-get when both
#                   answer. NOTE terminus-2 installs no *agent*: it runs in the
#                   harbor process and only shell commands cross into the sandbox.
#                   Its 12.0 s is these two packages, plus a tmux session start
#                   that baking does not remove -- so treat 12.0 s as a ceiling.
#   claude-code     _installed_claude_satisfies_version(): with no `version` set it
#                   is `command -v claude` under a PATH including ~/.local/bin;
#                   with one set it compares `claude --version` exactly. On glibc
#                   Harbor itself installs via the bootstrap script, so the recipe
#                   uses the same installer rather than npm.
#   mini-swe-agent  get_version_command() is `uv tool list | grep mini-swe-agent`,
#                   so it must be installed *as a uv tool*, under the same user and
#                   Python (3.12) Harbor would have used. A plain `pip install`
#                   satisfies nothing and the install runs anyway.
#
# These layers are rendered and reviewed but **not yet built** -- see --render-only
# and the caveat in the part 1 README. Validate a recipe with one image before a
# batch: `docker build`, then run the harness's own version command inside it.
NVM_VERSION = "v0.40.2"  # harbor/agents/installed/node_install.py
NODE_MAJOR = 22
MINI_SWE_PYTHON = "3.12"  # harbor/agents/installed/mini_swe_agent.py

HARNESS_LAYERS: dict[str, str] = {
    "terminus-2": """
# Harness layer: terminus-2. Not an agent install -- terminus-2 is host-internal.
# These are the two packages TmuxSession installs on first use; with both present
# _install_recording_tools() logs "Both tmux and asciinema are already installed"
# and execs no package manager.
RUN apt-get update \\
 && apt-get install -y --no-install-recommends tmux asciinema \\
 && rm -rf /var/lib/apt/lists/* \\
 && tmux -V && asciinema --version
""",
    "claude-code": """
# Harness layer: claude-code. Mirrors harbor's own glibc install path
# (claude_code.py: bootstrap.sh, not npm -- npm is its musl/Alpine branch), and
# puts claude on the PATH the version check uses.
RUN apt-get update \\
 && apt-get install -y --no-install-recommends curl bash procps ca-certificates \\
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://downloads.claude.ai/claude-code-releases/bootstrap.sh \\
    | bash -s --{version_arg} \\
 && echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
ENV PATH="/root/.local/bin:${{PATH}}"
RUN claude --version
""",
    "mini-swe-agent": """
# Harness layer: mini-swe-agent. Installed as a uv tool because that is what
# `uv tool list | grep mini-swe-agent` -- harbor's version check -- looks at. The
# --with extras are harbor's own: litellm's `proxy` extra is avoided on purpose.
RUN apt-get update \\
 && apt-get install -y --no-install-recommends curl bash git build-essential \\
 && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:${{PATH}}"
RUN uv python install {python_version} \\
 && uv tool install --python {python_version} mini-swe-agent{version_spec} \\
      --with litellm --with orjson --with fastapi \\
 && mini-swe-agent --help >/dev/null \\
 && uv tool list | grep mini-swe-agent
""",
}

# Node is only needed on musl images, where claude-code falls back to npm. Kept
# here so a customer adapting the recipe has the same snippet harbor uses.
NVM_SNIPPET = f"""
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/{NVM_VERSION}/install.sh \\
    | env -u NODE_VERSION bash \\
 && export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh" \\
 && nvm install {NODE_MAJOR} && nvm alias default {NODE_MAJOR} && npm -v
"""


def harness_layer(harness: str, version: str | None) -> str:
    """Render one harness layer, or raise for an unknown name."""
    try:
        template = HARNESS_LAYERS[harness]
    except KeyError:
        raise SystemExit(
            f"no bake recipe for harness {harness!r}; "
            f"known: {', '.join(sorted(HARNESS_LAYERS))}"
        ) from None
    return template.format(
        version_arg=f" {version}" if version else "",
        version_spec=f"=={version}" if version else "",
        python_version=MINI_SWE_PYTHON,
    )


def dockerfile_for(base: str, harness: str | None, version: str | None) -> str:
    body = DOCKERFILE.format(base=base)
    return body if harness is None else body + harness_layer(harness, version)


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


# --- portable image references -------------------------------------------------
#
# prepared.json is committed to the repository, so it must not contain the ECR
# account id of whoever produced it. The manifest therefore stores
# "<repository>:<tag>" and the registry host is reattached at use time.
#
# The test for "already qualified" is Docker's own: a reference is registry-scoped
# when its first path segment contains a dot or a colon (a hostname or host:port).
# Plain "swesmith-arm64:tag" has no slash at all, so it is unqualified.


def is_registry_qualified(ref: str) -> bool:
    head = ref.split("/", 1)[0]
    return "/" in ref and ("." in head or ":" in head)


def unqualified_image(ref: str) -> str:
    """Drop the registry host from an image reference, if it has one."""
    return ref.split("/", 1)[1] if is_registry_qualified(ref) else ref


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


def prepare_one(
    key: str,
    base: str,
    target: str,
    timeout: int,
    push: bool,
    harness: str | None = None,
    harness_version: str | None = None,
) -> dict:
    started = time.time()
    with tempfile.TemporaryDirectory() as ctx:
        Path(ctx, "Dockerfile").write_text(
            dockerfile_for(base, harness, harness_version)
        )
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
        # Recorded so a consumer can tell a plain prepared image from one carrying
        # a harness: the tags differ, but only this says which harness.
        **({"harness": harness} if harness else {}),
        # Recorded WITHOUT the registry host, so the manifest is portable across
        # accounts and regions -- it is a committed artifact and a fully qualified
        # ECR URI would bake in the account id that produced it. Consumers put the
        # registry back via unqualified_image()/qualify_image().
        "image": unqualified_image(target),
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
        "--bake-harness",
        choices=sorted(HARNESS_LAYERS),
        default=None,
        help="Also install this harness into the image, so Harbor's per-trial agent "
        "setup short-circuits. Changes the tag to <key>-prepared-<harness>-arm64: a "
        "baked image is harness-specific and must not replace the shared base tag.",
    )
    ap.add_argument(
        "--harness-version",
        default=None,
        help="Pin the baked harness version. Must equal the agent's `version` option "
        "in the eval config, or Harbor's check fails and it reinstalls anyway.",
    )
    ap.add_argument(
        "--render-only",
        action="store_true",
        help="Print the Dockerfile that would be built and exit. Needs no docker "
        "daemon, and is how a new harness recipe gets reviewed before a build.",
    )
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

    if args.render_only:
        # No docker, no ECR, no image list: this is a review aid for the recipes.
        print(
            dockerfile_for(
                "<repo image>", args.bake_harness, args.harness_version
            ).rstrip()
        )
        return

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

    # A baked image is not interchangeable with a plain one, so it gets its own
    # tag rather than overwriting the tag parts 2-4 already point at.
    suffix = (
        PREPARED_SUFFIX
        if not args.bake_harness
        else f"prepared-{args.bake_harness}-arm64"
    )
    targets = {
        key: f"{registry}/{args.ecr_repository}:{key.replace('/', '_')}-{suffix}"
        for key in images
    }

    print(
        f"preparing {len(images)} images, concurrency {args.concurrency}, "
        f"push={args.push}, harness={args.bake_harness or 'none'}"
    )
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {
            pool.submit(
                prepare_one,
                k,
                images[k],
                targets[k],
                args.timeout,
                args.push,
                args.bake_harness,
                args.harness_version,
            ): k
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
