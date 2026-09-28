#!/usr/bin/env python
"""Build arm64 SWE-bench Verified instance images -- the 219 that are not published.

    uv run swebench/build_images.py --list
    uv run swebench/build_images.py --dry-run          # render, no docker
    uv run swebench/build_images.py --limit 2 --concurrency 2
    uv run swebench/build_images.py --concurrency 8 --push

Why this is needed
------------------
**\\[measured\\]** 281 of the 500 Verified instances have a published
``swebench/sweb.eval.arm64.*`` image; the other 219 have none, and AgentCore
Runtime only accepts arm64. Unlike SWE-smith this needs no reverse engineering:
``sweb.base.*`` and ``sweb.env.*`` are *meant* to be built locally by the
``swebench`` harness, which is why they were never published. So "build the
missing ones" means driving swebench's own three-stage chain -- base, env,
instance -- with the architecture forced to arm64.

The one thing upstream will not do for you
------------------------------------------
``TestSpec.arch`` is a field, and ``make_test_spec(..., arch="x86_64")`` is its
default. Every public entry point (``build_instance_images``, the evaluation
runner) constructs its specs *without* passing ``arch``, so there is no flag that
makes the harness build arm64 on an x86 host. This script therefore builds the
specs itself with ``arch="arm64"`` and hands them to upstream's
``build_env_images`` / ``build_instance_image``, which are spec-driven and
idempotent (``get_test_specs_from_dataset`` passes TestSpec objects through
untouched, so the arch survives). Everything else -- the Dockerfiles, the
per-version install commands, the eval scripts -- is upstream's, unmodified.

The same substitution the SWE-smith builder needs
-------------------------------------------------
swebench builds with docker-py (``client.api.build``), which uses the legacy
builder. Against a locally built base on a containerd-snapshotter host that fails
with ``parent snapshot ... does not exist``, exactly as it does in
swesmith/build_images.py. So ``docker_build.build_image`` is replaced with
a ``docker build`` CLI call; only the *mechanism* changes, the Dockerfile and setup
scripts are whatever upstream computed.

Cost, honestly
--------------
Three stages per instance, but base and env are shared: **\\[measured\\]** the full 500
resolve to 1 base image and 40 env images, and the 219 missing ones to **1 base and
34 env images** -- so the env stage is 34 builds, not 219. On 8 vCPU under qemu,
**\\[measured\\]** for ``psf__requests-5414``:

    base      ~7 min     573 MB compressed   (ubuntu 22.04 + aarch64 miniconda)
    env       ~9 min     856 MB              (conda env create + the repo's deps)
    instance    65 s     873 MB / 3.28 GB on disk

The instance stage is cheap; the env stage is where the qemu time goes, and it scales
with the repo's dependency tree rather than with the instance count. A repo whose
wheels are published for aarch64 (requests, pytest, pylint) installs binaries; one
that compiles (matplotlib, scikit-learn, xarray) is far slower, so order your batches
cheapest-first and prefer a native arm64 builder (``DOCKER_HOST=ssh://<graviton>``, or
CodeBuild ARM) for the rest.

The tag that makes the build usable
-----------------------------------
swebench tags a local build ``sweb.eval.arm64.<instance_id>``, but a *generated task
dir* names ``swebench/sweb.eval.arm64.<instance_id with __ -> _1776_>`` -- namespace
added, ``__`` respelled (``TestSpec.instance_image_key``). So a freshly built image
does not satisfy the task that needs it, and the mismatch only surfaces as an
``ImageBuildError`` minutes into a trial. Every successful build therefore also gets
that second local tag (``--task-namespace``, default ``swebench``). This was found by
running the end-to-end, not by reading the code.

Verified end to end
-------------------
**\\[measured\\]** ``psf__requests-5414`` built here, then: the image runs
(``aarch64``, python 3.9.25 in the ``testbed`` env, repo at the base commit), Harbor
wrapped and pushed it, an AgentCore runtime was created from it, and the **oracle gate
scored 1.000 in 71 s** -- so a self-built arm64 image is a working task image, reward
path included. The remaining 218 are a matter of qemu hours, not of correctness.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ARM_ARCH = "arm64"
ARM_PLTF = "linux/arm64/v8"
DATASET = "princeton-nlp/SWE-bench_Verified"

HERE = Path(__file__).resolve().parent
PUBLISHED_LIST = HERE / "data" / "swebv-arm64-instances.txt"
# Beside the scripts and not under data/, because data/ holds only what a command
# regenerates and this is the opposite: findings that no command can rediscover
# without a gate run to read.
PINS_FILE = HERE / "env-pins.txt"


def _run(cmd: list[str], log_path: Path, timeout: int) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        try:
            proc = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
            )
            return proc.returncode, ""
        except subprocess.TimeoutExpired:
            return 124, f"timed out after {timeout}s"


# Whether to strip caches inside the build (see SLIM_SETUP_ENV). A module-level
# switch because build_image_cli's signature is fixed by upstream's call sites --
# it is invoked positionally from build_env_images and by keyword from
# build_instance_image, so an extra parameter would break one of them.
SLIM = True

SLIM_SETUP_ENV = """

# Not upstream's. **\\[measured\\]** on a matplotlib env image, /opt/miniconda3/pkgs is
# 4.4 GB uncompressed while the installed environment is 419 MB, which pushed the
# image to 3264 MB compressed -- past ACR's 2048 MB ceiling, so CreateAgentRuntime
# rejected it with "maxImageSizeMb limit exceeded". Removing the package tarballs
# and caches leaves the environment itself untouched.
#
# This has to happen in *this* RUN: a later layer cannot shrink an image, it only
# hides files. That is why it is appended to upstream's script rather than added as
# a Dockerfile step.
conda clean -afy || true
rm -rf /opt/miniconda3/pkgs/* /root/.cache/pip /root/.cache/conda || true
"""

SLIM_SETUP_REPO = """
python -m pip cache purge >/dev/null 2>&1 || true
rm -rf /root/.cache/pip || true
"""


def slimmed(setup_scripts: dict) -> dict:
    """Append cache cleanup to upstream's setup scripts, in-layer."""
    if not SLIM or not setup_scripts:
        return setup_scripts
    out = dict(setup_scripts)
    if "setup_env.sh" in out:
        out["setup_env.sh"] = out["setup_env.sh"] + SLIM_SETUP_ENV
    if "setup_repo.sh" in out:
        out["setup_repo.sh"] = out["setup_repo.sh"] + SLIM_SETUP_REPO
    return out


def build_image_cli(
    image_name: str,
    setup_scripts: dict,
    dockerfile: str,
    platform: str,
    client=None,
    build_dir: Path | None = None,
    nocache: bool = False,
) -> None:
    """Replacement for swebench's docker-py build. Same inputs, BuildKit instead.

    Accepts the arguments both ways because upstream calls it positionally from
    ``build_env_images`` and by keyword from ``build_instance_image``.
    """
    if build_dir is None:
        raise ValueError("build_dir is required")
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    for name, content in slimmed(setup_scripts or {}).items():
        (build_dir / name).write_text(content)
    (build_dir / "Dockerfile").write_text(dockerfile)

    command = [
        "docker",
        "build",
        "--platform",
        platform,
        # Single-platform: an OCI index is rejected by AgentCore Runtime, which is
        # why harbor's provider passes this too.
        "--provenance=false",
        "-t",
        image_name,
        str(build_dir),
    ]
    if nocache:
        command.insert(2, "--no-cache")
    code, note = _run(command, build_dir / "build_image.log", 10800)
    if code != 0:
        raise RuntimeError(
            f"docker build failed for {image_name} ({note or 'rc=' + str(code)}); "
            f"see {build_dir / 'build_image.log'}"
        )


def instance_ids(args) -> list[str]:
    """The instances to build: explicit, from a file, or the unpublished complement."""
    if args.instances:
        return [i.strip() for i in args.instances.split(",") if i.strip()]
    if args.instance_list:
        return [
            line.strip()
            for line in args.instance_list.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    # Default: everything the published-arm64 list does not cover. That list is an
    # *output* of swebench/probe_arm64_images.py --out, so refresh it rather than
    # trusting a stale copy -- one manifest request per instance, and Docker Hub
    # counts them against the anonymous 100/hour limit.
    from swebench.harness.utils import load_swebench_dataset

    everything = [
        inst["instance_id"] for inst in load_swebench_dataset(DATASET, "test")
    ]
    if not args.published.exists():
        raise SystemExit(
            f"{args.published} not found. Either pass --instances/--instance-list, or "
            "produce the coverage list with:\n"
            "  uv run swebench/probe_arm64_images.py --out swebench/data/swebv-arm64-instances.txt"
        )
    published = {
        line.strip() for line in args.published.read_text().splitlines() if line.strip()
    }
    missing = [i for i in everything if i not in published]
    print(
        f"{len(everything)} instances, {len(published)} with a published arm64 image, "
        f"{len(missing)} to build"
    )
    return missing


def arm64_specs(ids: list[str]):
    """Load the dataset rows and turn them into arm64 test specs."""
    from swebench.harness.test_spec.test_spec import make_test_spec
    from swebench.harness.utils import load_swebench_dataset

    dataset = load_swebench_dataset(DATASET, "test", instance_ids=ids)
    specs = [make_test_spec(inst, arch=ARM_ARCH) for inst in dataset]
    wrong = [s.instance_id for s in specs if s.platform != ARM_PLTF]
    if wrong:
        raise SystemExit(f"arch override did not take: {wrong[:3]}")
    return specs


@dataclass
class Pins:
    """pip constraints to add to an env image, keyed either way round."""

    by_repo: dict[str, list[str]] = field(default_factory=dict)
    by_instance: dict[str, list[str]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.by_repo or self.by_instance)

    def merge(self, other: Pins) -> Pins:
        for src, dst in ((other.by_repo, self.by_repo),
                         (other.by_instance, self.by_instance)):
            for key, reqs in src.items():
                dst.setdefault(key, []).extend(reqs)
        return self


def parse_pins(values: list[str] | None, source: str = "--pin") -> Pins:
    """``'sphinx-doc/sphinx=docutils<0.17'`` -> pins keyed by repo.

    ``instance:<id>=<req>`` keys the same thing by instance id instead, which is
    what the pin file uses -- see ``load_pin_file``.
    """
    pins = Pins({}, {})
    for raw in values or []:
        if "=" not in raw:
            raise SystemExit(
                f"{source} wants [instance:]KEY=requirement[,requirement], got {raw!r}"
            )
        key, requirements = raw.split("=", 1)
        parsed = [s.strip() for s in requirements.split(",") if s.strip()]
        key = key.strip()
        if key.startswith("instance:"):
            pins.by_instance.setdefault(key[len("instance:"):], []).extend(parsed)
        else:
            pins.by_repo.setdefault(key.removeprefix("repo:"), []).extend(parsed)
    return pins


def load_pin_file(path: Path) -> Pins:
    """Read the durable pin list: one ``[instance:]KEY=req[,req]`` per line, ``#`` comments.

    Why a file and not just the flag. `--pin` alone makes a *measured* finding live
    in someone's shell history: the six sphinx instances below were diagnosed, pinned
    and re-gated to 1.000, and then a from-scratch rebuild that followed the
    documented command reproduced all six failures, because the documented command
    does not carry the pin. The gate caught it (9 instances at reward 0.0 where the
    previous run had 3), but only for a reader who knew 3 was the number to expect.

    The docstring of ``apply_env_pins`` argues against a built-in pin *table*, and
    that argument still holds: a pin guessed per repo would break the repo versions
    that need the opposite pin. This file is the narrower thing -- entries keyed by
    **instance id**, so each one touches only the env image it was verified against,
    and sphinx 3's ``docutils<0.17`` cannot leak onto sphinx 5.
    """
    if not path.exists():
        return Pins({}, {})
    lines = [
        line.split("#", 1)[0].strip()
        for line in path.read_text().splitlines()
    ]
    return parse_pins([line for line in lines if line], source=str(path))


def pins_for(spec, pins: Pins) -> list[str]:
    """The requirements to add for one instance: its repo's pins, then its own."""
    requirements = list(pins.by_repo.get(spec.repo, ()))
    for req in pins.by_instance.get(spec.instance_id, ()):
        if req not in requirements:
            requirements.append(req)
    return requirements


def apply_instance_pin(image: str, reqs: list[str], build_root: Path) -> None:
    r"""Add a pip-constraint layer on top of one *instance* image.

    Why this exists. **\[measured\]** six of ten self-built sphinx instances gated
    0.000 with every test erroring on ``ModuleNotFoundError: No module named
    'roman'``: the env resolved **docutils 0.23** against **Sphinx 3.1.0**, and
    docutils dropped ``utils.roman`` in 0.18. Pinning ``docutils<0.17`` in the same
    container turned 0 passing into **14 passing**, i.e. the 2 FAIL_TO_PASS and 12
    PASS_TO_PASS tests that had failed.

    That is *temporal drift*, not architecture: an instance's own requirements file
    is years old and leaves its transitive deps unpinned, so rebuilding today
    resolves versions the code never saw. It would happen identically on amd64 --
    which is why the published images work and a fresh build does not.

    Why the *instance* layer and not the env layer, which is where this used to go.
    **\[measured\]** all 44 sphinx instances in Verified -- and all 10 that are
    self-built -- share a single env image (``sweb.env.py.arm64.764c2112...``): the
    env stage creates a conda environment, and the repo install happens in the
    instance stage. So an env-level pin is a pin on *every* instance of the repo,
    including the 4 self-built sphinx instances that gate 1.000 without it and the
    sphinx-5-era ones that need a newer docutils, not an older one. Keying the pin
    file by instance id would have been a false promise at that layer. The instance
    image is per instance, nothing is built on top of it, and the eval script does
    not reinstall dependencies -- so this layer applies exactly to what was measured.

    Retagging ``image`` in place means the push, the task-facing alias and the size
    report all see the pinned image with no further wiring.
    """
    context = build_root / "pins" / image.replace(":", "__").replace("/", "_")
    context.mkdir(parents=True, exist_ok=True)
    requirements = " ".join(f'"{r}"' for r in reqs)
    (context / "Dockerfile").write_text(
        f"FROM {image}\n"
        "RUN . /opt/miniconda3/bin/activate && conda activate testbed && "
        f"pip install --no-input {requirements}\n"
    )
    code, note = _run(
        ["docker", "build", "--platform", ARM_PLTF, "--provenance=false",
         "-t", image, str(context)],
        context / "build.log",
        1800,
    )
    if code != 0:
        raise RuntimeError(
            f"pin layer failed ({note or 'rc=' + str(code)}); "
            f"see {context / 'build.log'}"
        )


def failed_env_keys(env_failed) -> set[str]:
    """Turn build_env_images' failure list into env image keys.

    It returns the failed *payloads*, not names: each one is the
    ``(image_name, setup_scripts, dockerfile, platform, client, build_dir)`` tuple
    that was passed to ``build_image``. So the key is element 0, and a naive
    ``set(env_failed)`` dies with ``unhashable type: 'dict'`` on the setup_scripts
    dict -- a branch only reachable when an env image actually fails, which is why
    a clean batch never hit it.
    """
    keys = set()
    for entry in env_failed or []:
        keys.add(entry[0] if isinstance(entry, (tuple, list)) else str(entry))
    return keys


def scratch_root(env_var: str, name: str) -> Path | None:
    """Where this script writes its build context and logs.

    Returns None rather than falling back to the current directory. That fallback
    dropped a stray build root into whatever directory you happened to run from --
    usually the repo -- and it did it even for `--list`, which needs no root at all.
    The kit's rule is that every path comes from config.env; a missing one is a
    mistake to report, not to paper over.
    """
    explicit = os.environ.get(env_var)
    if explicit:
        return Path(explicit)
    work_dir = os.environ.get("KIT_WORK_DIR")
    return Path(work_dir) / name if work_dir else None


def require_root(root: Path | None, env_var: str) -> Path:
    if root is None:
        raise SystemExit(
            f"no build root: set KIT_WORK_DIR (source config.env) or {env_var}, "
            "or pass --build-root"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def task_facing_tag(instance_id: str, namespace: str) -> str:
    """The reference a *generated task dir* will name, which is not what we build.

    swebench tags a locally built image `sweb.eval.arm64.<instance_id>:latest`, but
    `TestSpec.instance_image_key` rewrites it for a remote registry:

        f"{namespace}/{key}".replace("__", "_1776_")

    so Harbor's swebench adapter emits `FROM swebench/sweb.eval.arm64.psf_1776_
    requests-5414:latest`. Two differences -- the namespace and the `__` spelling --
    and neither is visible until a trial fails with ImageBuildError several minutes
    in. **\\[measured\\]** this bit on the first end-to-end attempt with a self-built
    image, which is why every build also gets this second local tag: with it, the
    generated task resolves its FROM from the local store and needs no editing.
    """
    return f"{namespace}/sweb.eval.{ARM_ARCH}.{instance_id.lower()}:latest".replace(
        "__", "_1776_"
    )


def add_task_facing_tag(image: str, instance_id: str, namespace: str) -> str | None:
    target = task_facing_tag(instance_id, namespace)
    out = subprocess.run(
        ["docker", "tag", image, target], capture_output=True, text=True
    )
    return target if out.returncode == 0 else None


def assert_arm64(image: str) -> None:
    out = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Architecture}}", image],
        capture_output=True,
        text=True,
    )
    got = out.stdout.strip()
    if got != ARM_ARCH:
        raise RuntimeError(f"{image} built as {got!r}, expected {ARM_ARCH!r}")


def dry_run(specs, build_root: Path) -> None:
    """Render every Dockerfile without a docker daemon, and check the arch took.

    The check that matters is the base image: swebench picks the Miniconda
    installer by arch, and an x86 installer inside a --platform arm64 build is the
    silent failure this whole script exists to avoid.
    """
    out_dir = build_root / "dry-run"
    out_dir.mkdir(parents=True, exist_ok=True)
    bases = {s.base_image_key: s.base_dockerfile for s in specs}
    envs = {s.env_image_key: s.env_dockerfile for s in specs}

    for name, text in {**bases, **envs}.items():
        (out_dir / (name.replace(":", "__").replace("/", "_") + ".Dockerfile")).write_text(text)
    for spec in specs:
        (out_dir / (spec.instance_image_key.replace(":", "__") + ".Dockerfile")).write_text(
            spec.instance_dockerfile
        )

    for name, text in bases.items():
        if "aarch64" not in text:
            raise SystemExit(
                f"base dockerfile for {name} has no aarch64 installer -- the arch "
                "override did not reach get_dockerfile_base()"
            )
    print(
        f"{len(specs)} instances -> {len(bases)} base image(s), {len(envs)} env image(s)"
    )
    print(f"every base fetches an aarch64 Miniconda installer; wrote {out_dir}")
    for name in sorted(bases):
        print(f"  base {name}")


def build_one(
    spec,
    client,
    timeout: int,
    task_namespace: str | None = None,
    push_to: tuple[str, str] | None = None,
    prune: bool = False,
    pin_reqs: list[str] | None = None,
    build_root: Path | None = None,
) -> dict:
    from swebench.harness.docker_build import build_instance_image

    start = time.time()
    image = spec.instance_image_key
    try:
        build_instance_image(spec, client, None, False)
        # On top of the instance image, never the shared env image; see
        # apply_instance_pin(). Inside the try so a bad pin is reported as this
        # instance failing rather than killing the batch.
        if pin_reqs:
            apply_instance_pin(image, pin_reqs, build_root or Path.cwd())
    except Exception as err:  # noqa: BLE001 - one bad instance must not stop the batch
        return {
            "instance_id": spec.instance_id,
            "image": image,
            "status": "error",
            "note": f"{type(err).__name__}: {err}",
            "sec": round(time.time() - start),
        }
    elapsed = round(time.time() - start)
    try:
        assert_arm64(image)
    except RuntimeError as err:
        return {
            "instance_id": spec.instance_id,
            "image": image,
            "status": "wrong-arch",
            "note": str(err),
            "sec": elapsed,
        }
    size = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Size}}", image],
        capture_output=True,
        text=True,
    ).stdout.strip()
    alias = (
        add_task_facing_tag(image, spec.instance_id, task_namespace)
        if task_namespace
        else None
    )
    result = {
        "instance_id": spec.instance_id,
        "image": image,
        # What a generated task dir will actually name. Without it the build is
        # useless to Harbor; see task_facing_tag().
        "task_image": alias,
        "status": "ok",
        "sec": elapsed,
        "size_mb": round(int(size) / 1e6) if size.isdigit() else None,
    }

    # Push (and optionally drop) as soon as this image exists rather than after the
    # whole batch: 219 instance images do not fit on a disk that holds the 34 env
    # images they are built from. **\[measured\]** this host had 38 GB free and one
    # instance image is ~3.3 GB on disk.
    if push_to is not None:
        registry, repository = push_to
        try:
            result["ecr_image"] = push_one(image, registry, repository)
        except subprocess.CalledProcessError as err:
            result["status"] = "push-failed"
            result["note"] = (err.stderr or "").strip().splitlines()[-1:] or ["push failed"]
            result["note"] = result["note"][0]
            return result
        if prune:
            drop_local([image, alias, result["ecr_image"]])
            result["pruned"] = True
    return result


def ecr_login(repository: str, region: str) -> str:
    """Ensure the repository exists, log docker in, and return the registry host."""
    account = subprocess.run(
        ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    registry = f"{account}.dkr.ecr.{region}.amazonaws.com"

    if (
        subprocess.run(
            ["aws", "ecr", "describe-repositories", "--repository-names", repository,
             "--region", region],
            capture_output=True,
        ).returncode
        != 0
    ):
        subprocess.run(
            ["aws", "ecr", "create-repository", "--repository-name", repository,
             "--region", region],
            capture_output=True,
            check=True,
        )

    password = subprocess.run(
        ["aws", "ecr", "get-login-password", "--region", region],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    subprocess.run(
        ["docker", "login", "--username", "AWS", "--password-stdin", registry],
        input=password,
        text=True,
        capture_output=True,
        check=True,
    )
    return registry


def push_one(image: str, registry: str, repository: str) -> str:
    """Push one image under a tag derived from its local name."""
    tag = image.split("/")[-1].replace(":", "-")
    target = f"{registry}/{repository}:{tag}"
    subprocess.run(["docker", "tag", image, target], check=True)
    subprocess.run(["docker", "push", target], capture_output=True, text=True, check=True)
    return target


def drop_local(tags: list[str | None]) -> None:
    """Remove local tags so the next build has the disk back.

    Only the instance image's own layers go: the base and env images it was built
    FROM keep their tags and stay cached for the rest of the batch, which is what
    makes the next instance build 65 s instead of 16 minutes.
    """
    for tag in [tag for tag in tags if tag]:
        subprocess.run(["docker", "image", "rm", tag], capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="List what would be built")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render the base/env/instance Dockerfiles and check the arch. No docker.",
    )
    parser.add_argument("--instances", default=None, help="Comma-separated instance ids")
    parser.add_argument("--instance-list", type=Path, default=None)
    parser.add_argument(
        "--published",
        type=Path,
        default=PUBLISHED_LIST,
        help="Instances that already have a published arm64 image; the default "
        "selection is everything NOT in this file",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=10800)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument(
        "--no-slim",
        action="store_true",
        help="Keep the conda package cache in the image. Default is to strip it: it "
        "is ~4.4 GB uncompressed on a scientific env, which puts the image over "
        "ACR's 2048 MB compressed ceiling and makes CreateAgentRuntime refuse it.",
    )
    parser.add_argument(
        "--pin",
        action="append",
        default=None,
        metavar="[instance:]KEY=REQ[,REQ]",
        help="Add pip constraints to an env image before the instance build, for "
        "instances whose oracle fails on dependency drift rather than on anything "
        "architectural. Applied on top of the pin file. Measured example: "
        "--pin 'sphinx-doc/sphinx=docutils<0.17'. See apply_env_pins().",
    )
    parser.add_argument(
        "--pins-file",
        type=Path,
        default=PINS_FILE,
        help=f"Durable pins, one [instance:]KEY=REQ per line (default "
        f"swebench/{PINS_FILE.name}). This is how a diagnosed dependency-drift fix "
        "survives the next from-scratch rebuild.",
    )
    parser.add_argument(
        "--no-pins-file",
        action="store_true",
        help="Ignore the pin file, to reproduce what an unpinned build produces.",
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=scratch_root("SWEBENCH_BUILD_ROOT", "swebv-arm64"),
    )
    parser.add_argument(
        "--task-namespace",
        default="swebench",
        help="Namespace a generated task dir will use in its FROM line. Every built "
        "image gets a second local tag under it, so the task resolves without "
        "editing. Pass '' to skip.",
    )
    parser.add_argument("--push", action="store_true")
    parser.add_argument(
        "--prune-after-push",
        action="store_true",
        help="Delete each instance image locally once it is in ECR. Needs --push. The "
        "base and env images stay cached, so the next instance still builds in ~1 min; "
        "without this, a full 219-instance run needs ~700 GB of local disk.",
    )
    parser.add_argument(
        "--ecr-repository",
        default=os.environ.get("SWEBENCH_ECR_REPOSITORY", "swebv-arm64"),
    )
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    args = parser.parse_args()

    global SLIM
    SLIM = not args.no_slim

    ids = instance_ids(args)
    if args.limit:
        ids = ids[: args.limit]
    if not ids:
        raise SystemExit("nothing selected")

    specs = arm64_specs(ids)

    if args.list:
        print(f"{len(specs)} instances to build (arch={ARM_ARCH}):")
        for spec in specs:
            print(f"  {spec.instance_id:45} {spec.instance_image_key}")
        return

    args.build_root = require_root(args.build_root, "SWEBENCH_BUILD_ROOT")

    if args.dry_run:
        dry_run(specs, args.build_root)
        return

    import docker
    from swebench.harness import docker_build

    # Upstream's build dirs are relative paths (logs/build_images/...), so run from
    # the build root and its logs land with the rest of the kit's scratch.
    os.chdir(args.build_root)
    docker_build.build_image = build_image_cli
    client = docker.from_env()

    if args.force_rebuild:
        # build_env_images removes and rebuilds the *env* images, but the instance
        # stage skips anything already tagged (build_instance_image checks
        # client.images.get first). Without this, --force-rebuild silently rebuilt
        # half of what it promised: a slimmed env image with a stale fat instance
        # image on top of it, pushed under the same tag and therefore invisible.
        for spec in specs:
            for tag in (
                spec.instance_image_key,
                task_facing_tag(spec.instance_id, args.task_namespace)
                if args.task_namespace
                else None,
            ):
                if tag:
                    subprocess.run(
                        ["docker", "image", "rm", "-f", tag], capture_output=True
                    )
        print(f"--force-rebuild: dropped {len(specs)} instance image(s) first")

    print(f"building env images for {len(specs)} instances")
    _, env_failed = docker_build.build_env_images(
        client, specs, args.force_rebuild, args.concurrency
    )
    if env_failed:
        dead = failed_env_keys(env_failed)
        before = len(specs)
        specs = [s for s in specs if s.env_image_key not in dead]
        print(
            f"{len(dead)} env image(s) failed, so {before - len(specs)} instance(s) "
            f"are skipped; see their build_image.log under "
            f"logs/build_images/env/. A PackagesNotFoundError for the python "
            f"version means that version has no linux-aarch64 conda package and the "
            f"instance cannot be built on arm64 at all."
        )
        for key in sorted(dead):
            print(f"  dead env {key}")

    pins = parse_pins(args.pin)
    if not args.no_pins_file:
        pins.merge(load_pin_file(args.pins_file))
    pin_reqs = {s.instance_id: pins_for(s, pins) for s in specs}
    pinned = {i: r for i, r in pin_reqs.items() if r}
    if pinned:
        print(f"{len(pinned)} instance(s) get a pip-constraint layer:")
        for instance, reqs in sorted(pinned.items()):
            print(f"  {instance:45} {' '.join(reqs)}")
    elif pins:
        print("pins are configured but match none of the instances being built")

    if args.prune_after_push and not args.push:
        raise SystemExit("--prune-after-push needs --push, or the image is just deleted")
    push_to = None
    if args.push:
        # Log in once, before any worker needs it: 219 concurrent `docker login`
        # calls against ECR is its own failure mode.
        push_to = (ecr_login(args.ecr_repository, args.region), args.ecr_repository)
        print(f"pushing to {push_to[0]}/{push_to[1]} as each image finishes")

    print(f"building {len(specs)} instance images, concurrency {args.concurrency}")
    print("(qemu emulation: minutes each is normal)")
    results: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                build_one,
                spec,
                client,
                args.timeout,
                args.task_namespace,
                push_to,
                args.prune_after_push,
                pin_reqs.get(spec.instance_id),
                args.build_root,
            )
            for spec in specs
        ]
        for future in cf.as_completed(futures):
            result = future.result()
            results.append(result)
            note = f" {result.get('note', '')}".rstrip()
            print(
                f"  {result['status']:11} {result['instance_id']:45} "
                f"{result.get('sec', 0):>5}s{note}",
                flush=True,
            )
            if result["status"] == "ok" and not result.get("task_image"):
                print(
                    "    WARNING: no task-facing tag; a generated task dir will not "
                    "resolve this image (see task_facing_tag())"
                )

    summary = args.build_root / "summary.json"
    summary.write_text(
        json.dumps(sorted(results, key=lambda r: r["instance_id"]), indent=2) + "\n"
    )
    ok = [r for r in results if r["status"] == "ok"]
    pushed = [r for r in ok if r.get("ecr_image")]
    print(f"\n{len(ok)}/{len(results)} usable; summary written to {summary}")
    if pushed:
        print(f"{len(pushed)} pushed to ECR"
              + (", local copies dropped" if args.prune_after_push else ""))
    if ok and not args.prune_after_push:
        print(
            "task dirs generated by swebench/tasks.sh resolve these from the "
            "local store via their second tag; see task_facing_tag()"
        )

    sys.exit(0 if len(ok) == len(results) else 1)


if __name__ == "__main__":
    main()
