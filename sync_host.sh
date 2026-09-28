#!/usr/bin/env bash
# Push this working copy to another host without destroying what that host generated.
#
#   ./sync_host.sh xixia                      # code only
#   ./sync_host.sh xixia --state-from haiyuan  # code, then refresh data/ from a host that has it
#   ./sync_host.sh xixia --dry-run
#
# Why this is a script and not an rsync you type. `rsync -a --delete` is the right
# way to push code -- it removes files you deleted here -- but the lists under
# `swebench/data/` and `swesmith/data/` are *generated* and deliberately not in git
# (every file there is a script output). So they exist on the host that ran the
# scripts and nowhere else, including here. A --delete sync from this machine
# therefore wipes them, and the symptom appears much later as a TASK_LIST that
# "cannot be found" or, worse, an eval that silently runs the wrong set.
#
# That has happened twice. Both times the fix was an --exclude typed on the command
# line, which is to say the fix lived in a shell history and did not survive the
# next invocation. Hence this file: the excludes are the thing being versioned.
#
# What that means for you: after cloning onto a fresh host you still have to either
# regenerate the lists (part 1 and part 2) or copy them, which --state-from does.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET="${1:?usage: sync_host.sh <ssh-host> [--state-from <ssh-host>] [--dry-run]}"
shift
STATE_FROM=""
RSYNC_EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --state-from) STATE_FROM="${2:?--state-from needs an ssh host}"; shift 2 ;;
    --dry-run) RSYNC_EXTRA+=(--dry-run --itemize-changes); shift ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Generated on the host that ran the scripts, and nowhere else. Never deleted,
# never overwritten by a code sync.
STATE_DIRS=(
  part1-arm64-images/swebench/data
  part1-arm64-images/swesmith/data
)

# .git is *included* on purpose: a host whose .git is stale reports files as
# untracked that are committed here, which makes `git status` there actively
# misleading. It is only a few MB.
EXCLUDES=(
  --exclude '*/.venv'
  --exclude '__pycache__'
  --exclude '.mypy_cache'
  --exclude '.ruff_cache'
  # Holds an account id, a role ARN and WANDB_API_KEY, and its paths are per-host.
  --exclude 'config.env'
)
for d in "${STATE_DIRS[@]}"; do
  EXCLUDES+=(--exclude "/$d/")
done

echo "code -> $TARGET"
rsync -a --delete "${EXCLUDES[@]}" ${RSYNC_EXTRA[@]+"${RSYNC_EXTRA[@]}"} \
  "$HERE/" "$TARGET:harbor-on-agentcore/"

if [ -n "$STATE_FROM" ]; then
  for d in "${STATE_DIRS[@]}"; do
    echo "state -> $TARGET:$d (from $STATE_FROM)"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    # Via this host because the two targets need not be able to reach each other.
    # No --delete in either direction: this adds and updates, it never removes.
    rsync -a ${RSYNC_EXTRA[@]+"${RSYNC_EXTRA[@]}"} \
      "$STATE_FROM:harbor-on-agentcore/$d/" "$tmp/"
    rsync -a ${RSYNC_EXTRA[@]+"${RSYNC_EXTRA[@]}"} \
      "$tmp/" "$TARGET:harbor-on-agentcore/$d/"
    rm -rf "$tmp"
    trap - EXIT
  done
fi

# Report rather than assume. A missing list is the failure this script exists to
# prevent, so say so here instead of letting part 3 discover it.
echo
echo "on $TARGET:"
# shellcheck disable=SC2029  # $d is meant to expand locally
ssh "$TARGET" "cd harbor-on-agentcore && git log --oneline -1 && for d in ${STATE_DIRS[*]}; do
  n=\$(ls \"\$d\" 2>/dev/null | wc -l)
  printf '  %-44s %s file(s)%s\n' \"\$d\" \"\$n\" \"\$([ \"\$n\" -eq 0 ] && echo '   <- empty: regenerate, or use --state-from' || true)\"
done"
