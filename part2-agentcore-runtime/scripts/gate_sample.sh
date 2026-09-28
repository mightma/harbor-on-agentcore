#!/usr/bin/env bash
# Oracle a random sample of tasks, to check the part of the reward path the gate
# does not reach.
#
#   scripts/gate_sample.sh "$HARBOR_DATASETS/swesmith-arm64" 50
#   scripts/gate_sample.sh "$HARBOR_DATASETS/swesmith-arm64" 50 16   # concurrency
#   SAMPLE_SEED=7 scripts/gate_sample.sh "$HARBOR_DATASETS/swesmith-arm64" 50
#
# `deploy_runtimes.sh` gates one task **per image**, which is the right unit for
# "can this image reward a correct patch" and is what makes 122 trials stand in for
# 45,844 tasks. But it leaves the per-task half of the shared-image mechanism
# unverified: every other task on that image differs from the gated one only by the
# `git checkout <instance_id>` in its healthcheck, and nothing has ever run that
# checkout for the other 39,564. If a branch is missing from the baked set, or the
# task's own golden patch does not apply to it, the trial fails at reward 0 -- and
# in training that is indistinguishable from a hard task.
#
# So this samples across the allowlist instead of stratifying by image: a random 50
# of a set where one repository holds 2,389 tasks and another holds 8 lands
# proportionally, which is exactly what a training batch will do.
#
# It creates no runtimes that deploy_runtimes.sh did not already create -- the
# sampled tasks share those images by construction.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART="$(cd "$HERE/.." && pwd)"
set -a; . "$PART/../config.env"; set +a

TASK_ROOT="${1:?usage: gate_sample.sh <task-root> [n-tasks] [n-concurrent]}"
N_TASKS="${2:-50}"
N_CONCURRENT="${3:-16}"
ALLOWLIST="${ALLOWLIST:-$KIT_STATE_DIR/gate_passing_tasks.txt}"

export HARBOR_AGENTCORE_ROLE_ARN="$ACR_EXECUTION_ROLE_ARN"

# Sample from the allowlist, not from the task root. A task on an image that failed
# the gate is already known to score 0, so including it would report a failure this
# check is not looking for.
if [ ! -r "$ALLOWLIST" ]; then
  echo "no allowlist at $ALLOWLIST" >&2
  echo "run gate_report.py --allowlist-out \"\$KIT_STATE_DIR\" after the gate" >&2
  exit 1
fi

mapfile -t PICKED < <(
  "$PART/.venv/bin/python" - "$TASK_ROOT" "$ALLOWLIST" "$N_TASKS" "${SAMPLE_SEED:-0}" <<'PY'
import pathlib, random, re, sys

root, allowlist, want, seed = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
names = sorted({line.strip() for line in open(allowlist) if line.strip()})
names = [n for n in names if (root / n / "task.toml").exists()]
random.Random(seed).shuffle(names)
picked = sorted(names[:want])

images = set()
for name in picked:
    text = (root / name / "task.toml").read_text()
    m = re.search(r'^\s*docker_image\s*=\s*"([^"]+)"', text, re.M)
    images.add(m.group(1) if m else f"<per-task:{name}>")
print("\n".join(picked))
print(f"# {len(picked)} task(s) over {len(images)} image(s)", file=sys.stderr)
PY
)

if [ "${#PICKED[@]}" -eq 0 ]; then
  echo "the allowlist named no task that exists under $TASK_ROOT" >&2
  echo "regenerate the task dirs, or re-gate: the two have drifted apart" >&2
  exit 1
fi

include_flags=()
for name in "${PICKED[@]}"; do
  include_flags+=(--include-task-name "$name")
done

JOB_NAME="${JOB_NAME:-gate-sample-$(date +%Y%m%d-%H%M%S)}"
echo "oracle over ${#PICKED[@]} sampled task(s) from $TASK_ROOT, seed ${SAMPLE_SEED:-0}, n=$N_CONCURRENT"

set +e
"$PART/.venv/bin/harbor" run \
  -p "$TASK_ROOT" \
  -a oracle \
  -e agentcore \
  --ek share_by_content=true \
  --ek delete_runtime=false \
  --ek prune_local_images=false \
  --ek exec_timeout_sec=3600 \
  -n "$N_CONCURRENT" \
  -o "$HARBOR_JOBS" \
  --job-name "$JOB_NAME" \
  "${include_flags[@]}"
rc=$?
set -e

echo
echo "results: $HARBOR_JOBS/$JOB_NAME (harbor rc=$rc)"
echo "every trial should score 1.0; read any that did not with:"
echo "  uv run scripts/gate_report.py \"$HARBOR_JOBS/$JOB_NAME\""
exit "$rc"
