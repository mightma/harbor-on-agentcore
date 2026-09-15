#!/usr/bin/env bash
# Generate Harbor task directories for the prepared SWE-smith arm64 images.
#
#   scripts/swesmith_tasks.sh                       # all repos in the manifest
#   scripts/swesmith_tasks.sh --repos mewwts__addict.75284f95 --limit 50
#
# Reads data/swesmith-manifests/ (durable) and writes to
# $HARBOR_DATASETS/swesmith-arm64 (disposable -- this regenerates it in ~10
# minutes, so never treat the task dirs as an artifact worth protecting).
#
# --shared-images is what makes 44,489 tasks resolve to 119 runtimes instead of
# 44,489: it deletes each task's environment/Dockerfile so the environment
# content hash falls back to sha256(docker_image), and moves the per-task
# `git checkout` into task.toml's [environment.healthcheck], which is not part of
# the hash. Pair it with `--ek share_by_content=true` in parts 2-4.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART="$(cd "$HERE/.." && pwd)"
set -a; . "$PART/../config.env"; set +a

OUT="${SWESMITH_TASKS:-$HARBOR_DATASETS/swesmith-arm64}"
MANIFESTS="${SWESMITH_MANIFESTS:-$PART/data/swesmith-manifests}"

if [ ! -f "$MANIFESTS/prepared.json" ]; then
  echo "no prepared.json under $MANIFESTS -- run scripts/prepare_swesmith_images.py first" >&2
  exit 1
fi

# Default to every repo the manifest knows about; callers can override with
# --repos. Passing the keys explicitly (rather than letting the adapter walk the
# whole registry) is what keeps this to the repos you actually built.
extra=("$@")
if ! printf '%s\n' "${extra[@]+"${extra[@]}"}" | grep -q -- '--repos'; then
  extra+=(--repos "$(paste -sd, "$MANIFESTS/prepared_profile_keys.txt")")
fi
if ! printf '%s\n' "${extra[@]}" | grep -q -- '--limit'; then
  extra+=(--limit 0)
fi

rm -rf "$OUT"; mkdir -p "$OUT"
exec "$PART/.venv/bin/python" -u "$HERE/swesmith_tasks.py" \
  --output-dir "$OUT" \
  --shared-images "$MANIFESTS/prepared.json" \
  "${extra[@]}"
