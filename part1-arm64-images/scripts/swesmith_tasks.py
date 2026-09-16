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

from task_sharing import ShareResult, TaskDir, report, share_tasks

ARM_PLTF = "linux/arm64/v8"
ARM_ARCH = "arm64"

# The adapter is installed as a package here (see pyproject.toml), not read out
# of a Harbor checkout, so there is no source directory to put on sys.path.
# ADAPTER_SRC lets you point at a working tree instead when you are editing the
# adapter itself; unset, the installed package wins.
ADAPTER_SRC = os.environ.get("ADAPTER_SRC")


# --- portable image references -------------------------------------------------
#
# prepared.json is a committed artifact, so it stores "<repository>:<tag>" with no
# registry host -- otherwise it would bake in the ECR account id of whoever
# produced it. The registry goes back on here.
#
# Fully qualified references are passed through unchanged, so a manifest produced
# before this change still works.


def is_registry_qualified(ref: str) -> bool:
    head = ref.split("/", 1)[0]
    return "/" in ref and ("." in head or ":" in head)


def ecr_registry() -> str:
    """The caller's ECR registry host, from $ECR_REGISTRY or the current identity."""
    override = os.environ.get("ECR_REGISTRY")
    if override:
        return override.rstrip("/")
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise SystemExit(
            "cannot qualify image references: set ECR_REGISTRY, or AWS_REGION so the "
            "registry can be derived from the current identity (source config.env)"
        )
    out = subprocess.run(
        ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise SystemExit(f"aws sts get-caller-identity failed: {out.stderr.strip()}")
    return f"{out.stdout.strip()}.dkr.ecr.{region}.amazonaws.com"


def qualify_image(ref: str, registry: str | None = None) -> str:
    if is_registry_qualified(ref):
        return ref
    return f"{registry or ecr_registry()}/{ref}"


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


def share_swesmith_tasks(output_dir: Path, prepared: dict[str, str]) -> ShareResult:
    """The SWE-smith adapter over ``share_tasks()``: one image per repository.

    Three lines of policy over a general mechanism (see task_sharing.py):

        group  = the repository, read back off the task's own FROM tag
        image  = that repository's prepared image, from prepared.json
        setup  = ``git checkout <instance_id>``

    Why it is needed here: the adapter writes the per-task checkout into the
    task's own Dockerfile::

        FROM <repo image>
        RUN apt-get install -y git ... && mkdir -p /logs
        RUN git fetch && git checkout <instance_id>

    That ``<instance_id>`` gives each of the ~44k tasks a distinct environment
    content hash, so each would push its own image and deploy its own AgentCore
    runtime. Against a 1,000-runtime account quota that does not run.

    Why it is *cheap* here, which is the part that does not generalise: every task
    of a SWE-smith repository is a branch off the same commit, and
    prepare_swesmith_images.py already baked git, /logs and every task branch into
    the image. So the per-task delta is a local checkout with no network -- ~0 s,
    once, per session. A dataset whose delta is an install instead should think
    twice; README.md works that comparison through.
    """

    def group_of(task: TaskDir) -> str | None:
        # The prepared images are keyed by profile key, and the un-prepared image
        # the adapter chose encodes it: swesmith.arm64.<profile key>.
        if not task.base_image:
            return None
        stem = task.base_image.split("/")[-1].split(":")[0]
        prefix = f"swesmith.{ARM_ARCH}."
        return stem[len(prefix) :] if stem.startswith(prefix) else None

    return share_tasks(
        output_dir,
        group_of=group_of,
        image_of=prepared.get,
        setup_cmd=lambda task: f"cd /testbed && git checkout {task.instance_id}",
    )


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
        "repository instead of one per task. See share_swesmith_tasks(), and "
        "task_sharing.py for the same mechanism against another dataset.",
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
        ok_records = [
            r for r in prepared_records if r.get("status") == "ok" and r.get("image")
        ]
        # Resolve the registry once, and only if something actually needs it.
        registry = (
            ecr_registry()
            if any(not is_registry_qualified(r["image"]) for r in ok_records)
            else None
        )
        prepared = {r["key"]: qualify_image(r["image"], registry) for r in ok_records}
        report(share_swesmith_tasks(args.output_dir, prepared))


if __name__ == "__main__":
    main()
