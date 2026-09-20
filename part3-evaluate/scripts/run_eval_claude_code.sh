#!/usr/bin/env bash
# Evaluate Claude Code as the harness on the arm64 slice of SWE-bench Verified,
# with every trial's sandbox an AgentCore Runtime session.
#
#   scripts/run_eval_claude_code.sh                              # $EVAL_BEDROCK_MODEL
#   scripts/run_eval_claude_code.sh us.anthropic.claude-sonnet-5
#   N_CONCURRENT=8 scripts/run_eval_claude_code.sh               # gentler
#
# Unlike the terminus-2 paths, the model call originates INSIDE the sandbox: harbor
# installs the CLI there and it authenticates with the runtime's execution role via
# IMDS. So this needs the execution role to allow bedrock:InvokeModel, and it means
# the role is exposed to whatever the agent runs. See configs/eval-claude-code.yaml.
#
# Prerequisites: part 1 generated the task dirs and pulled the arm64 bases; part 2
# deployed the runtimes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART="$(cd "$HERE/.." && pwd)"
set -a; . "$PART/../config.env"; set +a

MODEL="${1:-$EVAL_BEDROCK_MODEL}"
TASKS="${TASKS:-$HARBOR_DATASETS/swebv-arm64}"
TASK_LIST="${TASK_LIST:-$PART/../part1-arm64-images/swebench/data/swebv-arm64-gated.txt}"
N_CONCURRENT="${N_CONCURRENT:-16}"
JOB_NAME="${JOB_NAME:-swebv-claudecode-$(basename "$MODEL")-$(date +%Y%m%d-%H%M%S)}"

export HARBOR_AGENTCORE_ROLE_ARN="$ACR_EXECUTION_ROLE_ARN"

# The switch that puts claude-code on Bedrock is this variable, not the model
# prefix. Harbor reads it here and re-exports it into the sandbox itself.
export CLAUDE_CODE_USE_BEDROCK=1

# A Bedrock API key in the shell profile would be forwarded into the sandbox and
# would silently become the credential under test instead of the execution role.
unset AWS_BEARER_TOKEN_BEDROCK

config="$TMPDIR/eval_claude_code_$JOB_NAME.yaml"
sed "s|bedrock/PLACEHOLDER|bedrock/$MODEL|" "$PART/configs/eval-claude-code.yaml" > "$config"

# SMOKE=n runs only the first n instances of the list -- for proving a new path
# works before paying for all 70. Do NOT try to get that effect by pointing
# TASK_LIST somewhere empty: an unreadable list used to mean "no filters", which
# silently ran the whole set.
include_flags=()
if [ -n "${SMOKE:-}" ]; then
  [ -r "$TASK_LIST" ] || { echo "SMOKE needs a readable TASK_LIST: $TASK_LIST" >&2; exit 1; }
  while read -r instance; do
    [ -n "$instance" ] && include_flags+=(-i "*$instance")
    [ "$((${#include_flags[@]} / 2))" -ge "$SMOKE" ] && break
  done < "$TASK_LIST"
  echo "SMOKE=$SMOKE: restricting to $((${#include_flags[@]} / 2)) instance(s)"
elif [ -n "$TASK_LIST" ]; then
  # A TASK_LIST that cannot be read is a mistake, not a request for everything.
  if [ ! -r "$TASK_LIST" ] || [ ! -s "$TASK_LIST" ]; then
    echo "TASK_LIST is set but not a readable non-empty file: $TASK_LIST" >&2
    echo "to run the whole task set on purpose, pass TASK_LIST=''" >&2
    exit 1
  fi
  while read -r instance; do
    [ -n "$instance" ] && include_flags+=(-i "*$instance")
  done < "$TASK_LIST"
fi

if [ "${#include_flags[@]}" -eq 0 ]; then
  echo "no instance filter: running every task under $TASKS" >&2
fi

echo "job=$JOB_NAME harness=claude-code model=bedrock/$MODEL"
echo "pool=$(find "$TASKS" -mindepth 1 -maxdepth 1 -type d | wc -l) from $TASKS" \
     "selected=$((${#include_flags[@]} / 2)) n=$N_CONCURRENT"
# pool is how many task dirs exist, selected is how many will actually run: the task
# directory is generated before the gate, so it still holds the tasks the gate later
# rejected. TASK_LIST is what excludes them.

set +e
"$PART/.venv/bin/harbor" run -c "$config" -p "$TASKS" \
  "${include_flags[@]+"${include_flags[@]}"}" \
  -n "$N_CONCURRENT" \
  -o "$HARBOR_JOBS" --job-name "$JOB_NAME" "${@:2}" 2>&1 \
  | sed 's/\x1b\[[0-9;]*m//g' \
  | grep -avE "Running trials|starting environment|running (agent|verifier)"
rc=${PIPESTATUS[0]}
set -e

echo "results: $HARBOR_JOBS/$JOB_NAME (harbor rc=$rc)"
"$PART/.venv/bin/python" "$HERE/summarize.py" "$HARBOR_JOBS/$JOB_NAME" || true
exit "$rc"
