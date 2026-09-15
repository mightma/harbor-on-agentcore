#!/usr/bin/env bash
# Materialise the training task set: SWE-smith, restricted to the oracle-clean
# images that part 2's gate identified.
#
#   scripts/make_train_set.sh
#   ALLOWLIST=/path/to/gate_passing_tasks.txt scripts/make_train_set.sh
#
# Symlinks rather than copies -- 38,815 task dirs, and the originals are already
# on local disk. Harbor reads task dirs, it does not write to them.
#
# Why restrict at all: an image whose oracle scores 0.0 cannot reward a correct
# patch, so every rollout on it is a guaranteed 0. That is not a hard task, it is
# a dead signal, and in GRPO a whole group of dead signals contributes no
# advantage while still costing a full rollout each.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART="$(cd "$HERE/.." && pwd)"
set -a; . "$PART/../config.env"; set +a

SRC="${SRC:-$HARBOR_DATASETS/swesmith-arm64}"
DST="${DST:-$HARBOR_DATASETS/swesmith-arm64-train}"
ALLOWLIST="${ALLOWLIST:-$KIT_STATE_DIR/gate_passing_tasks.txt}"

[ -d "$SRC" ] || { echo "no task set at $SRC -- run part 1 first" >&2; exit 1; }

rm -rf "$DST"; mkdir -p "$DST"

if [ -f "$ALLOWLIST" ]; then
  n=0
  while read -r name; do
    [ -n "$name" ] && [ -d "$SRC/$name" ] || continue
    ln -s "$SRC/$name" "$DST/$name"; n=$((n + 1))
  done < "$ALLOWLIST"
  echo "linked $n of $(grep -c . "$ALLOWLIST") allowlisted tasks into $DST"
else
  echo "no allowlist at $ALLOWLIST -- using the whole task set unrestricted." >&2
  echo "Run part 2's gate_report.py --allowlist-out to produce one; without it" >&2
  echo "you will train on images whose reward path is dead." >&2
  for d in "$SRC"/*/; do ln -s "$d" "$DST/$(basename "$d")"; done
  echo "linked $(find "$DST" -maxdepth 1 -mindepth 1 | wc -l) tasks into $DST"
fi
