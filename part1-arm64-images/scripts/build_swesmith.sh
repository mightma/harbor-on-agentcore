#!/usr/bin/env bash
# Build the arm64 SWE-smith repository images, then wrap them for shared use.
#
#   scripts/build_swesmith.sh --list                        # what would be built
#   scripts/build_swesmith.sh --limit 4 --concurrency 4     # try four repos first
#   scripts/build_swesmith.sh --concurrency 12 --push       # the real run, ~2.5 h
#
# This is the expensive step in the whole kit. Every image is built under qemu
# because no arm64 SWE-smith image exists anywhere; median 843 s per repo at
# concurrency 12 on a 192-vCPU host.
#
# Two things that will waste hours if you skip them:
#
#   1. The qemu binfmt handler must be registered, and it disappears on some
#      hosts (it vanished mid-session repeatedly while this was being written).
#      This script refuses to start without it.
#   2. Docker Hub's anonymous pull limit is 100/hour/IP and a full run makes ~134
#      pulls of the amd64 source images. Log in, or build in two batches.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART="$(cd "$HERE/.." && pwd)"
set -a; . "$PART/../config.env"; set +a

if [ ! -e /proc/sys/fs/binfmt_misc/qemu-aarch64 ]; then
  echo "qemu-aarch64 binfmt handler is not registered; arm64 builds will fail." >&2
  echo "  docker run --privileged --rm tonistiigi/binfmt --install arm64" >&2
  exit 1
fi

export SWESMITH_BUILD_ROOT="${SWESMITH_BUILD_ROOT:-$KIT_WORK_DIR/swesmith-arm64}"
mkdir -p "$SWESMITH_BUILD_ROOT"

# Stage 1: recreate each repo's conda env on arm64 from the published amd64 image
# and commit it as swesmith.arm64.<repo>.
"$PART/.venv/bin/python" -u "$HERE/build_swesmith_images.py" "$@"

# Stage 2: bake in git, uv, /logs and every task branch (`git fetch --all`), so a
# task's only per-task work is a local `git checkout`. This is what lets one image
# serve every task of a repository.
#
# Skipped for the inspection-only flags, which do not produce images.
case " $* " in
  *" --list "*) exit 0 ;;
esac

push=()
case " $* " in
  *" --push "*) push=(--push) ;;
esac
exec "$PART/.venv/bin/python" -u "$HERE/prepare_swesmith_images.py" \
  --concurrency 12 "${push[@]+"${push[@]}"}" \
  --output "$SWESMITH_BUILD_ROOT/prepared.json"
