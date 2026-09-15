#!/usr/bin/env python
"""Generate Harbor tasks for SWE-smith instances that reference the arm64 repo images.

    scripts/swesmith_tasks.sh --repos mewwts__addict.75284f95 \
        --output-dir $HARBOR_DATASETS/swesmith-arm64
    scripts/swesmith_tasks.sh --repos a.1234abcd,b.5678efgh --limit 200

The companion to scripts/build_swesmith_images.py: that builds the images, this makes
the tasks point at them.

Harbor's swesmith adapter names each task's base image through the profile
registry::

    # harbor/adapters/swesmith/src/swesmith_adapter/utils.py
    rp = registry.get_from_inst(sample)
    id_to_image[sample["instance_id"]] = rp.image_name

and ``RepoProfile.image_name`` interpolates ``self.arch``, which upstream fixes to
the *build host's* architecture at import time. So the same one-attribute override
the builder uses is enough here too, and Harbor needs no patching at all — which
is the point of doing it from the outside: the adapter stays exactly as upstream
ships it.

``--repos`` filters instances down to the repositories whose images actually
exist, which matters because building all 134 under qemu is not something you do
in one sitting. Without it you get tasks whose FROM cannot resolve, and the
failure surfaces much later as an ImageBuildError inside a trial.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ARM_PLTF = "linux/arm64/v8"
ARM_ARCH = "arm64"

# The adapter is installed as a package here (see pyproject.toml), not read out
# of a Harbor checkout, so there is no source directory to put on sys.path.
# ADAPTER_SRC lets you point at a working tree instead when you are editing the
# adapter itself; unset, the installed package wins.
ADAPTER_SRC = os.environ.get("ADAPTER_SRC")


def patch_registry_to_arm64() -> None:
    """Flip every profile to arm64 so `image_name` names the arm64 tag.

    Set on the instance, not the class: profiles are dataclasses whose arch/pltf
    are fields, so __init__ would reapply its own baked-in x86_64 defaults over a
    class-attribute patch. They are singletons, so the instance patched here is
    the one Harbor's adapter gets back from `registry.get_from_inst`.
    """
    from swesmith.profiles import registry

    for key, entry in registry.items():
        # Mirror aliases are the same profiles under a second name, and the
        # non-Python registries (Go, JS, ...) hold entries that are not callable
        # -- `registry["<a go repo>"]()` raises TypeError. Filter exactly the way
        # scripts/build_swesmith_images.py does, so both scripts see the same set.
        if key.startswith("swesmith/"):
            continue
        if getattr(entry, "__module__", "") != "swesmith.profiles.python":
            continue
        profile = _profile(entry)
        profile.arch = ARM_ARCH
        profile.pltf = ARM_PLTF


def _profile(entry):
    """Return the profile singleton whether the registry holds a class or an instance."""
    return entry() if isinstance(entry, type) else entry


def local_arm64_images() -> set[str]:
    out = subprocess.run(
        ["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in out.stdout.splitlines() if "swesmith.arm64." in line}




def share_tasks_by_repo(output_dir: Path, prepared: dict[str, str]) -> tuple[int, int]:
    """Rewrite generated tasks so one image serves every task of a repository.

    The adapter writes the per-task checkout into the task's own Dockerfile::

        FROM <repo image>
        RUN apt-get install -y git ... && mkdir -p /logs
        RUN git fetch && git checkout <instance_id>

    Harbor's environment identity is ``environment_content_hash(environment_dir,
    docker_image=...)``, so that ``<instance_id>`` gives each of the ~52k tasks a
    distinct hash: each pushes its own image and, on AgentCore Runtime, deploys
    its own runtime. Against a 1,000-runtime account quota that does not run.

    Two changes make them converge, and neither touches Harbor or the adapter:

    1. Delete ``environment/Dockerfile``. With the directory empty, the hash falls
       back to ``sha256(docker_image)`` -- identical for every task of a
       repository, still distinct across repositories. Everything the deleted
       Dockerfile installed now lives in the prepared image
       (scripts/prepare_swesmith_images.py), including the fetched branches.

    2. Carry the per-task checkout in ``task.toml``, which is *not* part of the
       content hash. ``[environment.healthcheck]`` is the hook that runs at the
       right moment: Trial._prepare() calls environment start, then
       ``run_healthcheck()``, then installs and runs the agent. The loop returns
       as soon as the command exits 0, so a successful checkout runs exactly once,
       and a failed one aborts the trial instead of handing the agent a repository
       sitting on the wrong commit.

    Pair with ``--ek share_by_content=true`` on the agentcore environment, which
    drops the task name from the image tag and the runtime name; without it the
    identical hashes still produce one image and one runtime per task.
    """
    import toml

    shared = skipped = 0
    for task_dir in sorted(p for p in output_dir.iterdir() if p.is_dir()):
        toml_path = task_dir / "task.toml"
        if not toml_path.exists():
            continue
        doc = toml.loads(toml_path.read_text())
        instance_id = str(doc.get("metadata", {}).get("instance_id") or task_dir.name)

        dockerfile = task_dir / "environment" / "Dockerfile"
        image = None
        if dockerfile.exists():
            for line in dockerfile.read_text().splitlines():
                if line.strip().upper().startswith("FROM "):
                    image = line.split(None, 1)[1].strip()
                    break
        key = None
        if image:
            stem = image.split("/")[-1].split(":")[0]
            if stem.startswith("swesmith.arm64."):
                key = stem[len("swesmith.arm64.") :]
        if key is None or key not in prepared:
            skipped += 1
            continue

        dockerfile.unlink()
        env = doc.setdefault("environment", {})
        env["docker_image"] = prepared[key]
        hc: dict[str, object] = {}
        # A local checkout: the prepared image already fetched every branch.
        hc["command"] = f"cd /testbed && git checkout {instance_id}"
        hc["timeout_sec"] = 120.0
        hc["retries"] = 2
        hc["interval_sec"] = 2.0
        env["healthcheck"] = hc
        toml_path.write_text(toml.dumps(doc))
        shared += 1

    return shared, skipped

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repos",
        default=None,
        help="Comma-separated profile keys to keep (the ones you have built images for)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--shared-images",
        type=Path,
        default=None,
        help="prepared.json from scripts/prepare_swesmith_images.py. Rewrites the generated "
        "tasks to share one image (and therefore one AgentCore runtime) per "
        "repository instead of one per task. See share_tasks_by_repo().",
    )
    parser.add_argument(
        "--require-local-images",
        action="store_true",
        help="Fail if a selected repo has no local arm64 image, instead of generating "
        "tasks whose FROM will not resolve",
    )
    args = parser.parse_args()

    if ADAPTER_SRC:
        sys.path.insert(0, ADAPTER_SRC)
    patch_registry_to_arm64()

    from swesmith.profiles import registry

    if args.repos:
        wanted = {key.strip() for key in args.repos.split(",") if key.strip()}
        unknown = wanted - set(registry.keys())
        if unknown:
            raise SystemExit(f"unknown profile keys: {sorted(unknown)}")
    else:
        wanted = None

    selected_images = {
        key: _profile(registry[key]).image_name for key in (wanted or ())
    }
    if selected_images:
        sample = next(iter(selected_images.values()))
        if ".arm64." not in sample:
            raise SystemExit(
                f"arch override did not take effect (got {sample}); "
                "swesmith's profile layout may have changed"
            )

    if args.require_local_images and selected_images:
        have = {name.split(":")[0] for name in local_arm64_images()}
        missing = [
            f"{key} -> {image}"
            for key, image in sorted(selected_images.items())
            if image not in have
        ]
        if missing:
            raise SystemExit(
                "no local arm64 image for:\n  "
                + "\n  ".join(missing)
                + "\nbuild them with scripts/build_swesmith.sh --repos ..."
            )

    from swesmith_adapter.adapter import SWESmithAdapter  # noqa: PLC0415

    # The adapter's own CLI has no repo filter, so it is driven directly. Note the
    # keyword is task_dir, not output_dir.
    adapter = SWESmithAdapter(
        task_dir=args.output_dir,
        limit=0 if wanted else args.limit,
        overwrite=args.overwrite,
    )

    if wanted:
        # Filter through the adapter's own instance -> image mapping rather than
        # recomputing profile keys, so the tasks generated are exactly those whose
        # FROM is one of the images asked for.
        targets = set(selected_images.values())
        selected = [
            instance_id
            for instance_id, image in adapter.id_to_docker_image.items()
            if image in targets
        ]
        selected.sort()
        print(f"{len(selected)} instances across {len(wanted)} repo(s)")
        adapter.task_ids = selected[: args.limit] if args.limit > 0 else selected

    adapter.run()
    generated = sorted(p.name for p in args.output_dir.iterdir() if p.is_dir())
    print(f"{len(generated)} task dirs in {args.output_dir}")

    if args.shared_images:
        prepared_records = json.loads(args.shared_images.read_text())
        prepared = {
            r["key"]: r["image"]
            for r in prepared_records
            if r.get("status") == "ok" and r.get("image")
        }
        shared, skipped = share_tasks_by_repo(args.output_dir, prepared)
        print(
            f"{shared} tasks rewritten to share {len(set(prepared.values()))} "
            f"repo image(s); {skipped} left per-task"
        )
        if skipped:
            print(
                "  (skipped tasks have no prepared image for their repo; they will "
                "still build their own image and runtime)"
            )


if __name__ == "__main__":
    main()
