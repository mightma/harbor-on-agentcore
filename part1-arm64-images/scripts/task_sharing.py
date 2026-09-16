"""Make one image -- and therefore one AgentCore runtime -- serve many tasks.

This is the dataset-independent half of part 1. ``share_tasks()`` is the extension
point: give it three functions and it rewrites a directory of generated Harbor
tasks so that tasks sharing an installed environment share an image.

    share_tasks(
        task_dir,
        group_of=...,    # TaskDir -> group key, or None to leave this task alone
        image_of=...,    # group key -> image reference, or None if you have none
        setup_cmd=...,   # TaskDir -> the shell command that adapts the image
    )

Nothing in the mechanism is specific to a dataset. Harbor keys a runtime on
``environment_content_hash(environment_dir, docker_image=...)``, and the three
edits below are what make that hash collapse across tasks:

1. **Delete ``environment/Dockerfile``.** With the directory empty the hash falls
   back to ``sha256(docker_image)`` (``environments/definition.py:102``) --
   identical for every task in a group, still distinct across groups. Whatever the
   deleted Dockerfile installed has to already be in the shared image.

2. **Point ``[environment].docker_image`` at the shared image.**

3. **Carry the per-task difference in ``[environment.healthcheck]``**, which lives
   in ``task.toml`` and is therefore *not* part of the content hash.
   ``Trial._prepare()`` starts the environment, runs the healthcheck, then installs
   and runs the agent, and ``run_healthcheck()`` returns as soon as the command
   exits 0 (``environments/base.py:1400``) -- so the setup runs exactly once, and a
   failure aborts the trial instead of handing the agent a wrongly-prepared tree.
   (``HealthcheckConfig``'s own docstring claims "all retries must pass". It is
   wrong; worth an upstream doc fix.)

Pair this with ``--ek share_by_content=true`` on the agentcore environment, which
drops the task name from the image tag and the runtime name. Without it the
identical hashes still produce one image and one runtime per task.

**What sharing is worth depends entirely on step 3's cost**, which is the one
judgement a customer has to make for their own dataset:

    Share when the per-task delta is *local and constant* -- SWE-smith's
    `git checkout` off a pre-fetched image is ~0 s. Do not share when it is *work*:
    SWE-bench Verified's delta is a clone, a history scrub and an editable install,
    minutes per trial, every trial, forever. The grouping that minimises runtime
    count is not automatically the grouping you want.

That is why this kit ships exactly one adapter over this helper --
``swesmith_tasks.py``, group = repository -- and deliberately does *not* group
SWE-bench Verified even though its ``env_image_key`` would cut 500 runtimes to 40.
The comparison is worked through in ``README.md``; read it before writing an
adapter for your own dataset.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

# Not parameters. run_healthcheck() stops at the first success, so these only
# decide how long a *failing* setup is retried before the trial is abandoned.
HEALTHCHECK_RETRIES = 2
HEALTHCHECK_INTERVAL_SEC = 2.0

# environment_content_hash() skips these when deciding whether the directory is
# empty (harbor/environments/definition.py:76). Anything else left behind keeps
# the per-task hash alive and silently defeats sharing.
ENV_HASH_IGNORES = frozenset({".DS_Store", ".git", "__pycache__"})


@dataclass
class TaskDir:
    """One generated task, parsed just enough for a grouping policy to decide."""

    path: Path
    doc: dict  # parsed task.toml
    instance_id: str
    base_image: str | None  # the FROM of environment/Dockerfile, if it has one

    @property
    def name(self) -> str:
        return self.path.name


class ShareResult(NamedTuple):
    shared: int
    skipped: int
    images: set[str]
    # Tasks whose environment/ still holds files after the Dockerfile was removed.
    # Their hash does not collapse; see ENV_HASH_IGNORES.
    not_collapsed: list[str]


def _from_image(dockerfile: Path) -> str | None:
    if not dockerfile.exists():
        return None
    for line in dockerfile.read_text().splitlines():
        if line.strip().upper().startswith("FROM "):
            return line.split(None, 1)[1].strip()
    return None


def _leftover_env_files(environment: Path) -> list[str]:
    if not environment.exists():
        return []
    return sorted(
        p.relative_to(environment).as_posix()
        for p in environment.rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and not ENV_HASH_IGNORES & set(p.relative_to(environment).parts)
    )


def read_task(path: Path) -> TaskDir | None:
    """Parse one task directory, or None if it is not one."""
    import toml

    toml_path = path / "task.toml"
    if not toml_path.exists():
        return None
    doc = toml.loads(toml_path.read_text())
    return TaskDir(
        path=path,
        doc=doc,
        instance_id=str(doc.get("metadata", {}).get("instance_id") or path.name),
        base_image=_from_image(path / "environment" / "Dockerfile"),
    )


def share_tasks(
    task_dir: Path,
    group_of: Callable[[TaskDir], str | None],
    image_of: Callable[[str], str | None],
    setup_cmd: Callable[[TaskDir], str],
    timeout_sec: float = 120.0,
) -> ShareResult:
    """Rewrite the tasks under ``task_dir`` so each group shares one image.

    ``group_of`` and ``image_of`` may both return None -- for a task whose group
    has no prepared image, meaning "leave it per-task". Those are counted as
    skipped rather than raised, because a partial build is the normal state
    halfway through part 1.
    """
    import toml

    shared = skipped = 0
    images: set[str] = set()
    not_collapsed: list[str] = []

    for path in sorted(p for p in task_dir.iterdir() if p.is_dir()):
        task = read_task(path)
        if task is None:
            continue

        key = group_of(task)
        image = image_of(key) if key is not None else None
        if image is None:
            skipped += 1
            continue

        environment = path / "environment"
        dockerfile = environment / "Dockerfile"
        if dockerfile.exists():
            dockerfile.unlink()
        leftover = _leftover_env_files(environment)
        if leftover:
            not_collapsed.append(f"{task.name}: {', '.join(leftover[:3])}")

        env = task.doc.setdefault("environment", {})
        env["docker_image"] = image
        env["healthcheck"] = {
            "command": setup_cmd(task),
            "timeout_sec": timeout_sec,
            "retries": HEALTHCHECK_RETRIES,
            "interval_sec": HEALTHCHECK_INTERVAL_SEC,
        }
        (path / "task.toml").write_text(toml.dumps(task.doc))
        shared += 1
        images.add(image)

    return ShareResult(shared, skipped, images, not_collapsed)


def report(result: ShareResult) -> None:
    """Print what share_tasks() did, including the two failure modes worth seeing."""
    print(
        f"{result.shared} tasks rewritten to share {len(result.images)} image(s); "
        f"{result.skipped} left per-task"
    )
    if result.skipped:
        print(
            "  (skipped tasks have no shared image for their group; they will still "
            "build their own image and runtime)"
        )
    if result.not_collapsed:
        print(
            f"  WARNING: {len(result.not_collapsed)} task(s) still have files under "
            "environment/, so their content hash does not collapse and each will "
            "deploy its own runtime anyway:"
        )
        for line in result.not_collapsed[:5]:
            print(f"    {line}")
