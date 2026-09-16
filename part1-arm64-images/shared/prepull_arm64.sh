#!/usr/bin/env bash
# Pull the prebuilt arm64 SWE-bench base images into the local Docker image store,
# paced to stay under Docker Hub's anonymous rate limit.
#
#   shared/prepull_arm64.sh run/swebv-arm64-eval.txt          # 73 images
#   PER_HOUR=90 shared/prepull_arm64.sh run/swebv-arm64-train.txt
#
# Why this exists at all: the agentcore provider builds each task's
# environment/Dockerfile with `docker build` and no `--pull`, so a base image that
# is already in the local store is used as-is and Docker Hub is never contacted.
# Without this step every trial resolves `FROM swebench/sweb.eval.arm64.<id>`
# against the registry, and Docker Hub's anonymous limit — 100 pulls per rolling
# hour per source IP, manifest HEADs included — turns into
#
#   ERROR: failed to solve: ... 429 Too Many Requests
#   harbor.environments.agentcore.image.ImageBuildError
#
# which Harbor reports as a build failure with nothing about rate limits in it.
#
# The pacing is deliberately dumb: PER_HOUR pulls, then sleep out the rest of the
# hour. The limit is a rolling window, so a token-bucket would finish sooner, but
# a run that gets throttled halfway leaves a half-pulled task set behind, and the
# whole point of prepulling is that the eval run afterwards has no registry
# dependency at all.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART="$(cd "$HERE/.." && pwd)"
set -a; . "$PART/../config.env"; set +a

LIST="${1:?usage: prepull_arm64.sh <instance-id-list> }"
PER_HOUR="${PER_HOUR:-90}"
CONC="${CONC:-6}"
LOG="${LOG:-$KIT_WORK_DIR/prepull-arm64.log}"

mkdir -p "$(dirname "$LOG")"

# swebench names an instance image sweb.eval.<arch>.<instance_id>:latest with the
# instance id lowercased and `__` spelled `_1776_`.
image_for() { echo "swebench/sweb.eval.arm64.$(echo "$1" | tr 'A-Z' 'a-z' | sed 's/__/_1776_/'):latest"; }

mapfile -t ids < <(grep -v '^[[:space:]]*$' "$LIST")
total=${#ids[@]}
echo "$(date -Is) prepull start list=$LIST total=$total per_hour=$PER_HOUR conc=$CONC" | tee -a "$LOG"

pulled=0
window_start=$(date +%s)
window_count=0

for id in "${ids[@]}"; do
  image=$(image_for "$id")

  # Already local? Costs no quota, so check before spending any.
  if docker image inspect "$image" >/dev/null 2>&1; then
    echo "$(date -Is) have $id" | tee -a "$LOG"
    continue
  fi

  # Spend the hour's budget, then wait for the window to roll over.
  if [ "$window_count" -ge "$PER_HOUR" ]; then
    elapsed=$(( $(date +%s) - window_start ))
    remain=$(( 3660 - elapsed ))
    if [ "$remain" -gt 0 ]; then
      echo "$(date -Is) budget spent ($window_count), sleeping ${remain}s" | tee -a "$LOG"
      sleep "$remain"
    fi
    window_start=$(date +%s)
    window_count=0
  fi

  while [ "$(jobs -rp | wc -l)" -ge "$CONC" ]; do sleep 2; done
  window_count=$((window_count + 1))
  pulled=$((pulled + 1))
  (
    start=$(date +%s)
    # Retry on 429 rather than skipping. The budget above is an estimate of the
    # limit, not a measurement of it: the window is rolling, other work on this
    # IP spends from the same bucket, and a manifest HEAD counts too. When the
    # bucket really is empty the only thing to do is wait for it to refill.
    for attempt in 1 2 3 4 5 6 7 8; do
      if out=$(docker pull -q --platform linux/arm64 "$image" 2>&1); then
        echo "$(date -Is) ok   $id ($(( $(date +%s) - start ))s) [$pulled/$total]" | tee -a "$LOG"
        exit 0
      fi
      if echo "$out" | grep -qiE '429|too many requests|toomanyrequests'; then
        echo "$(date -Is) 429  $id, attempt $attempt, sleeping 600s" | tee -a "$LOG"
        sleep 600
        continue
      fi
      echo "$(date -Is) FAIL $id: $(echo "$out" | tail -1)" | tee -a "$LOG"
      exit 1
    done
    echo "$(date -Is) GAVEUP $id: still rate limited" | tee -a "$LOG"
  ) &
done
wait

echo "$(date -Is) prepull done attempted=$pulled" | tee -a "$LOG"
docker image ls --format '{{.Repository}}' | grep -c 'sweb.eval.arm64' | xargs echo "local arm64 base images:" | tee -a "$LOG"
df -h "${KIT_WORK_DIR:-/}" | tail -1 | tee -a "$LOG"
