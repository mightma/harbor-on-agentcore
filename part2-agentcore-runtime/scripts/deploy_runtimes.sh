#!/usr/bin/env bash
# Pre-create one AgentCore runtime per task image, and gate each one's reward path.
#
#   scripts/deploy_runtimes.sh "$HARBOR_DATASETS/swesmith-arm64" 20
#   scripts/deploy_runtimes.sh "$HARBOR_DATASETS/swebv-arm64/eval" 16
#   REPO_PREFIXES=hukkin__tomli.443a0c1b scripts/deploy_runtimes.sh <task-root> 4
#
# It runs Harbor's `oracle` agent over exactly one task per image. That single
# pass does three jobs at once:
#
#   1. builds and pushes the provider's sandbox-wrapped image, once per image
#      rather than once per task (that is what --ek share_by_content=true buys),
#   2. creates the runtime and leaves it deployed (--ek delete_runtime=false),
#   3. verifies the reward path, because the oracle applies the golden patch: a
#      score below 1.0 means that image's tests do not pass even when they should.
#
# Step 3 is the reason this exists rather than just deploying runtimes. A dead
# reward path is invisible in RL -- the loop runs, checkpoints land, and every
# reward is 0. Both failures this kit hit (numpy SIGILL, truncated golden patches)
# would have been silent without it.
#
# REPO_PREFIXES narrows the pass to a comma-separated list of repo keys, matched
# case-insensitively because task dir names lowercase the owner segment while
# profile keys do not (Knio__dominate vs knio__dominate). That is what makes it
# cheap to re-gate only the images that failed a previous pass.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART="$(cd "$HERE/.." && pwd)"
set -a; . "$PART/../config.env"; set +a

TASK_ROOT="${1:?usage: deploy_runtimes.sh <task-root> [n-concurrent]}"
N_CONCURRENT="${2:-16}"

export HARBOR_AGENTCORE_ROLE_ARN="$ACR_EXECUTION_ROLE_ARN"

# The two task sets in this kit have different shapes, and the difference decides
# how many trials this pass needs:
#
#   SWE-smith   task.toml sets [environment].docker_image and environment/ is
#               empty. Every task of a repository names the same image, so they
#               share one runtime and one representative task gates all of them.
#               119 trials cover 44,489 tasks.
#
#   SWE-bench   each task has its own environment/Dockerfile and no
#               docker_image. Every task is its own environment, so there is
#               nothing to group by and every task needs its own trial.
#               70 trials cover 70 tasks.
#
# Grouping by docker_image and finding none used to abort with "generate them in
# part 1 first", which was wrong and actively misleading: the tasks were there.
mapfile -t PICKED < <(
  "$PART/.venv/bin/python" - "$TASK_ROOT" "${REPO_PREFIXES:-}" <<'PY'
import sys, re, pathlib
root = pathlib.Path(sys.argv[1])
prefixes = tuple(p.lower() for p in sys.argv[2].split(",") if p)
shared, per_task = {}, []
for toml_path in sorted(root.glob("*/task.toml")):
    name = toml_path.parent.name
    if prefixes and not name.lower().startswith(prefixes):
        continue
    m = re.search(r'^\s*docker_image\s*=\s*"([^"]+)"', toml_path.read_text(), re.M)
    if m:
        shared.setdefault(m.group(1), name)
    else:
        per_task.append(name)
# A task set is one shape or the other; if it somehow mixes, cover both.
for name in list(shared.values()) + per_task:
    print(name)
PY
)

if [ "${#PICKED[@]}" -eq 0 ]; then
  echo "no task.toml files under $TASK_ROOT" >&2
  echo "generate the task set in part 1 first" >&2
  exit 1
fi

echo "deploying ${#PICKED[@]} runtimes at -n $N_CONCURRENT"

include_flags=()
for name in "${PICKED[@]}"; do
  include_flags+=(--include-task-name "$name")
done

JOB_NAME="${JOB_NAME:-gate-$(date +%Y%m%d-%H%M%S)}"

exec "$PART/.venv/bin/harbor" run \
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
