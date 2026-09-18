#!/usr/bin/env python
"""Build arm64 SWE-smith repository images so SWE-smith tasks can run on AgentCore Runtime.

    swesmith/build.sh --list
    swesmith/build.sh --limit 8 --concurrency 4
    swesmith/build.sh --repos mewwts__addict.75284f95,arrow-py__arrow.1d70d009
    swesmith/build.sh --limit 8 --push

Why this is needed
------------------
AgentCore Runtime only accepts arm64 images. Of the SWE datasets, only SWE-bench
Verified has arm64 images published (281 of its 500 instances, namespace
``swebench/sweb.eval.arm64.*``); SWE-Gym, SWE-bench's train split and SWE-smith
are amd64-only. That forces RL training and evaluation onto the same benchmark.
SWE-smith is the way out: it is the purpose-built training set (~60k synthetic
task instances), and crucially it is **one image per repository**, not one per
instance — so ~130 Python images cover the whole thing, and a useful training
slice is a handful of them.

What upstream does, and the two things that block arm64
-------------------------------------------------------
``swesmith.profiles.base.RepoProfile`` decides the architecture once, at import,
from the machine running it::

    arch: str = "x86_64" if platform.machine() not in {"aarch64", "arm64"} else "arm64"
    pltf: str = "linux/x86_64" if arch == "x86_64" else "linux/arm64/v8"

and ``image_name`` is derived from ``arch``. So on this x86 host every profile
names and builds an x86_64 image. That part is just an attribute.

The real blocker is in ``PythonProfile.build_image``::

    BASE_IMAGE_KEY = "jyangballin/swesmith.x86_64"
    ...
    dockerfile = get_dockerfile_env(self.pltf, self.arch, "py", base_image_key=BASE_IMAGE_KEY)

That base — ubuntu + miniconda + build tools, shared by every repo image — is
hardcoded and published for amd64 only. Asking for ``--platform linux/arm64``
against it yields ``FROM --platform=linux/arm64/v8 jyangballin/swesmith.x86_64``,
which is an amd64 image forced through qemu: it may even build, but the conda
environment inside would be the wrong architecture, so it is not a thing you want
to evaluate on.

How this script gets around it
------------------------------
Rather than copy upstream's setup-command list (which would silently rot), it
patches the single symbol ``build_image`` reaches for and lets the rest of
upstream run untouched:

1. Render swebench's *own* arm64 base Dockerfile
   (``get_dockerfile_base("linux/arm64/v8", "arm64", "py", ...)``, which correctly
   fetches ``Miniconda3-...-Linux-aarch64.sh``) and build it once as
   ``swesmith.arm64.base:latest``.
2. Set ``arch``/``pltf`` on the selected profile classes to arm64, so
   ``image_name`` becomes ``swesmith.arm64.<owner>_1776_<repo>.<commit8>``.
3. Monkeypatch ``swesmith.profiles.python.get_dockerfile_env`` with a wrapper that
   substitutes ``base_image_key``, then call each profile's own
   ``build_image()``. Upstream's setup script, install commands and env yml are
   used verbatim.
4. Assert the built image really is arm64, because a qemu build that silently
   produced amd64 is exactly the failure this is trying to avoid.

Cost, honestly
--------------
These build under qemu on an x86 host: ``conda env create`` plus the repo's own
install, emulated. Budget tens of minutes per image and run several in parallel;
this box has 192 vCPU, so concurrency is nearly free while each individual build
is slow. A native arm64 builder (``DOCKER_HOST=ssh://<graviton>`` or CodeBuild
ARM) is dramatically faster and is the right answer for the full ~130.

The task side is swesmith/tasks.py, which applies the same arch override
to Harbor's swesmith adapter so the generated Dockerfiles reference these tags.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ARM_PLTF = "linux/arm64/v8"
ARM_ARCH = "arm64"
BASE_TAG = "swesmith.arm64.base:latest"
# Pinned so a rebuild months from now produces the same base. These are the
# values swebench itself defaults to for its py images.
UBUNTU_VERSION = "22.04"
CONDA_VERSION = "py311_23.11.0-2"


def _run(cmd: list[str], log_path: Path, timeout: int) -> tuple[int, str]:
    with log_path.open("w") as log:
        try:
            proc = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
            )
            return proc.returncode, ""
        except subprocess.TimeoutExpired:
            return 124, f"timed out after {timeout}s"


def python_profiles() -> dict[str, type]:
    """The Python repo profiles, keyed by `<owner>__<repo>.<commit8>`.

    The registry also holds `swesmith/<key>` aliases for the mirror repos; those
    are the same profiles under a second name, so they are dropped.
    """
    from swesmith.profiles import registry

    out = {}
    for key, profile_cls in registry.items():
        if key.startswith("swesmith/"):
            continue
        if getattr(profile_cls, "__module__", "") != "swesmith.profiles.python":
            continue
        out[key] = profile_cls
    return dict(sorted(out.items()))


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


def build_base(build_root: Path, timeout: int) -> None:
    from swebench.harness.dockerfiles import get_dockerfile_base

    check = subprocess.run(
        ["docker", "image", "inspect", BASE_TAG],
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        print(f"base {BASE_TAG} already present")
        return

    dockerfile = get_dockerfile_base(
        ARM_PLTF,
        ARM_ARCH,
        "py",
        ubuntu_version=UBUNTU_VERSION,
        conda_version=CONDA_VERSION,
    )
    base_dir = build_root / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "Dockerfile").write_text(dockerfile)

    print(f"building {BASE_TAG} (arm64 ubuntu {UBUNTU_VERSION} + miniconda)")
    start = time.time()
    code, note = _run(
        [
            "docker",
            "build",
            "--platform",
            f"linux/{ARM_ARCH}",
            "--provenance=false",
            "-t",
            BASE_TAG,
            str(base_dir),
        ],
        base_dir / "build.log",
        timeout,
    )
    if code != 0:
        raise SystemExit(
            f"base build failed ({note or 'rc=' + str(code)}); see {base_dir / 'build.log'}"
        )
    print(f"base built in {time.time() - start:.0f}s")


def patch_for_arm64(profile_classes: list[type]) -> None:
    """Point the profiles at arm64 and at our own base image.

    The arch has to be set on the *instance*, not the class. Profiles are
    dataclasses whose `arch`/`pltf` are fields with defaults ("x86_64" /
    "linux/x86_64", chosen at import from platform.machine()), so a dataclass
    __init__ reassigns them from its own baked-in defaults and silently undoes a
    class-attribute patch. They are also singletons (SingletonMeta), so patching
    the one instance is enough and every later `profile_cls()` sees it.
    """
    import swesmith.profiles.python as pymod
    from swebench.harness.dockerfiles import get_dockerfile_env as real_get_dockerfile_env

    for profile_cls in profile_classes:
        profile = profile_cls()
        profile.arch = ARM_ARCH
        profile.pltf = ARM_PLTF

    def get_dockerfile_env_arm64(platform, arch, language, **kwargs):
        # build_image passes base_image_key="jyangballin/swesmith.x86_64"; swap in
        # the arm64 base built above and leave everything else to upstream.
        kwargs["base_image_key"] = BASE_TAG.split(":")[0]
        return real_get_dockerfile_env(platform, arch, language, **kwargs)

    pymod.get_dockerfile_env = get_dockerfile_env_arm64

    def build_image_cli(
        image_name: str,
        setup_scripts: dict,
        dockerfile: str,
        platform: str,
        client=None,
        build_dir: Path | None = None,
        nocache: bool = False,
    ) -> None:
        """Build via the docker CLI instead of swebench's docker-py path.

        swebench builds with docker-py (`client.api.build`), which goes through
        the legacy builder. Against a locally built arm64 base that fails on this
        host with

            BuildImageError: ... NotFound: parent snapshot sha256:... does not
            exist: not found

        -- a containerd-snapshotter mismatch between the BuildKit-produced base
        and the legacy builder consuming it. Shelling out to `docker build` uses
        BuildKit for both halves and the snapshot resolves.

        Only the build *mechanism* is replaced: the Dockerfile and the
        setup_env.sh contents are whatever upstream's build_image computed.
        """
        if build_dir is None:
            raise ValueError("build_dir is required")
        build_dir = Path(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)
        for name, content in (setup_scripts or {}).items():
            (build_dir / name).write_text(content)
        (build_dir / "Dockerfile").write_text(dockerfile)

        command = [
            "docker",
            "build",
            "--platform",
            platform,
            # Keep it a single-platform image: an OCI index would be rejected
            # downstream by AgentCore Runtime, the same reason the harbor
            # provider passes this flag.
            "--provenance=false",
            "-t",
            image_name,
            str(build_dir),
        ]
        if nocache:
            command.insert(2, "--no-cache")
        log_path = build_dir / "build_image.log"
        code, note = _run(command, log_path, 10800)
        if code != 0:
            raise RuntimeError(
                f"docker build failed for {image_name} ({note or 'rc=' + str(code)}); "
                f"see {log_path}"
            )

    pymod.build_image_sweb = build_image_cli


def amd64_image_name(profile) -> str:
    """The published x86_64 image for this profile, regardless of the patched arch."""
    return (
        f"{profile.org_dh}/swesmith.x86_64."
        f"{profile.owner}_1776_{profile.repo}.{profile.commit[:8]}"
    ).lower()


def _portable_pip_deps(pip_deps: list[str], profile) -> list[str]:
    """Drop requirements that only exist inside the exported image.

    `conda env export` lists the environment's *installed* distributions, which
    includes the repository itself: swesmith's own setup installed it from source,
    so setuptools-scm gave it a version that was never published
    (`monkeytype==23.3.1.dev1`, `termcolor==0.1.dev1`). Re-creating the
    environment from that list therefore asks PyPI for a version that does not
    exist and `conda env create` dies with `CondaEnvException: Pip failed`, after
    ~10 minutes of qemu. Two of the first four repositories built failed this way;
    neither failure had anything to do with the architecture.

    The repository is installed separately by the upstream `setup_env.sh` install
    command, so it must not be in this list at all. Local version segments
    (`+local`) are dropped for the same reason.
    """
    repo_names = set()
    for attr in ("repo_name", "repo", "mirror_name"):
        value = getattr(profile, attr, None)
        if isinstance(value, str) and value:
            # profile keys look like "Instagram__MonkeyType.70c3acf6": the
            # distribution name is the segment after "__", before the commit.
            stem = value.split("/")[-1].split("__")[-1].split(".")[0]
            repo_names.add(_canonical_dist(stem))

    kept: list[str] = []
    for dep in pip_deps:
        name = _canonical_dist(re.split(r"[=<>!~\[;]", dep, maxsplit=1)[0])
        version = dep.split("==", 1)[1].strip() if "==" in dep else ""
        unpublished = ".dev" in version or "+" in version
        if name in repo_names or unpublished or _is_amd64_only(name):
            continue
        kept.append(dep)
    return kept


# CUDA wheels are published for x86_64 (and sbsa on some releases) but not for the
# plain aarch64 tag these environments resolve, so an exported amd64 environment
# that installed torch drags in requirements that cannot exist here:
#
#   ERROR: Could not find a version that satisfies the requirement
#          nvidia-cudnn-cu12==9.1.0.70
#
# They are not needed either: the arm64 torch wheel is CPU-only, which is also why
# the same Dockerfile produces a 382 MB image on arm64 against 6.2 GB on amd64
# (GUIDE.md section 6.3). Dropping them lets torch pull whatever it actually needs.
_AMD64_ONLY_PREFIXES = ("nvidia-", "triton", "cupy-cuda", "tensorrt")


def _is_amd64_only(name: str) -> bool:
    return name.startswith(_AMD64_ONLY_PREFIXES)


def _canonical_dist(name: str) -> str:
    """PEP 503 normalisation, so monkey_type / MonkeyType / monkey-type match."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def ensure_env_yml(profile, timeout: int) -> Path:
    """Make sure the conda environment spec `build_image` reads actually exists.

    `PythonProfile.build_image` opens `self._env_yml`
    (LOG_DIR_ENV/<repo>/sweenv_<repo>.yml) and heredocs it into the image as the
    argument to `conda env create`. Nothing in the swesmith *package* writes that
    file -- it is produced by a separate step in the upstream repo that the 0.0.6
    wheel does not ship -- so a fresh checkout fails with:

        FileNotFoundError: logs/build_images/env/<repo>/sweenv_<repo>.yml

    The published amd64 image already contains the built environment, and amd64
    runs natively on this x86 host, so the spec is recovered from there rather
    than guessed.

    `conda env export` is not usable as-is: its conda section pins amd64 build
    strings (`bzip2==1.0.8=h5eee18b_6`) and linux-64-only packages
    (`ld_impl_linux-64`), none of which resolve on arm64. What *is* portable is
    the `pip:` section -- the repo's actual Python dependencies -- plus the Python
    version. The low-level libraries come back automatically as dependencies of
    python, so they are deliberately dropped.
    """
    env_yml = profile._env_yml
    if env_yml.exists():
        return env_yml

    image = amd64_image_name(profile)
    export = (
        "source /opt/miniconda3/bin/activate && conda activate testbed && "
        "conda env export && python -V"
    )
    subprocess.run(
        ["docker", "pull", "-q", "--platform", "linux/amd64", image],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    out = subprocess.run(
        ["docker", "run", "--rm", "--platform", "linux/amd64", image, "bash", "-lc", export],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    ).stdout

    pip_deps: list[str] = []
    in_pip = False
    python_version = profile.python_version
    for line in out.splitlines():
        if line.strip().startswith("- pip:"):
            in_pip = True
            continue
        if in_pip:
            stripped = line.strip()
            if stripped.startswith("- "):
                pip_deps.append(stripped[2:].strip())
                continue
            in_pip = False
        if line.startswith("Python "):
            python_version = line.split()[1]

    pip_deps = _portable_pip_deps(pip_deps, profile)

    lines = [
        "name: testbed",
        "channels:",
        "  - defaults",
        "  - conda-forge",
        "dependencies:",
        f"  - python=={python_version}",
        "  - pip",
        "  - setuptools",
        "  - wheel",
    ]
    if pip_deps:
        lines.append("  - pip:")
        lines += [f"      - {dep}" for dep in pip_deps]
    env_yml.parent.mkdir(parents=True, exist_ok=True)
    env_yml.write_text("\n".join(lines) + "\n")
    return env_yml


def assert_arm64(image: str) -> None:
    out = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Architecture}}", image],
        capture_output=True,
        text=True,
    )
    got = out.stdout.strip()
    if got != ARM_ARCH:
        raise RuntimeError(f"{image} built as {got!r}, expected {ARM_ARCH!r}")


def build_one(key: str, profile_cls: type, timeout: int) -> dict:
    profile = profile_cls()
    image = profile.image_name
    start = time.time()

    exists = subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, text=True
    )
    if exists.returncode == 0:
        try:
            assert_arm64(image)
            return {"key": key, "image": image, "status": "cached", "sec": 0}
        except RuntimeError as err:
            return {"key": key, "image": image, "status": "wrong-arch", "note": str(err)}

    try:
        ensure_env_yml(profile, timeout)
        # Upstream writes its own build log under LOG_DIR_ENV/<repo_name>.
        profile.build_image()
    except Exception as err:  # noqa: BLE001 - one bad repo must not stop the batch
        return {
            "key": key,
            "image": image,
            "status": "error",
            "note": f"{type(err).__name__}: {err}",
            "sec": round(time.time() - start),
        }

    elapsed = round(time.time() - start)
    check = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Architecture}} {{.Size}}", image],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        return {
            "key": key,
            "image": image,
            "status": "missing",
            "note": "build_image returned but the tag does not exist; see swesmith's build log",
            "sec": elapsed,
        }
    arch, size = check.stdout.split()
    if arch != ARM_ARCH:
        return {
            "key": key,
            "image": image,
            "status": "wrong-arch",
            "note": f"built as {arch}",
            "sec": elapsed,
        }
    return {
        "key": key,
        "image": image,
        "status": "ok",
        "sec": elapsed,
        "size_mb": round(int(size) / 1e6),
    }


def push(images: list[str], repository: str, region: str) -> None:
    account = subprocess.run(
        ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    registry = f"{account}.dkr.ecr.{region}.amazonaws.com"

    subprocess.run(
        ["aws", "ecr", "describe-repositories", "--repository-names", repository, "--region", region],
        capture_output=True,
    ).returncode == 0 or subprocess.run(
        ["aws", "ecr", "create-repository", "--repository-name", repository, "--region", region],
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

    for image in images:
        tag = image.split("/")[-1].replace(":", "-")
        target = f"{registry}/{repository}:{tag}"
        subprocess.run(["docker", "tag", image, target], check=True)
        print(f"pushing {target}")
        subprocess.run(["docker", "push", target], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="List Python repo profiles and exit")
    parser.add_argument("--repos", default=None, help="Comma-separated profile keys to build")
    parser.add_argument("--limit", type=int, default=None, help="Build the first N profiles")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--timeout", type=int, default=10800, help="Per-image build timeout in seconds"
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=scratch_root("SWESMITH_BUILD_ROOT", "swesmith-arm64"),
    )
    parser.add_argument("--push", action="store_true", help="Also push to ECR")
    parser.add_argument(
        "--ecr-repository",
        default=os.environ.get("SWESMITH_ECR_REPOSITORY", "swesmith-arm64"),
    )
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    args = parser.parse_args()

    profiles = python_profiles()
    if args.list:
        print(f"{len(profiles)} Python repo profiles:")
        for key, profile_cls in profiles.items():
            print(f"  {key:55} {profile_cls.__name__}")
        return

    if args.repos:
        wanted = [key.strip() for key in args.repos.split(",") if key.strip()]
        missing = [key for key in wanted if key not in profiles]
        if missing:
            raise SystemExit(f"unknown profile keys: {missing}\nrun --list to see them")
        selected = {key: profiles[key] for key in wanted}
    else:
        keys = list(profiles)[: args.limit] if args.limit else list(profiles)
        selected = {key: profiles[key] for key in keys}

    if not selected:
        raise SystemExit("nothing selected")

    args.build_root = require_root(args.build_root, "SWESMITH_BUILD_ROOT")
    build_base(args.build_root, args.timeout)
    patch_for_arm64(list(selected.values()))

    print(f"building {len(selected)} arm64 repo images, concurrency {args.concurrency}")
    print("(qemu emulation: tens of minutes each is normal)")
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(build_one, key, cls, args.timeout): key
            for key, cls in selected.items()
        }
        for future in futures:
            result = future.result()
            results.append(result)
            note = f" {result.get('note', '')}".rstrip()
            print(
                f"  {result['status']:11} {result['key']:50} "
                f"{result.get('sec', 0):>5}s {result.get('size_mb', '')}{note}"
            )

    summary = args.build_root / "summary.json"
    summary.write_text(json.dumps(results, indent=2))
    ok = [r for r in results if r["status"] in ("ok", "cached")]
    print(f"\n{len(ok)}/{len(results)} usable; summary written to {summary}")

    if args.push and ok:
        push([r["image"] for r in ok], args.ecr_repository, args.region)

    if len(ok) != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
