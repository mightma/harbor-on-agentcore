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

# SMOKE / SAMPLE narrow the gated list further; see select_tasks.sh for which
# to use. A list that cannot be read is a mistake there, not "run everything".
. "$HERE/select_tasks.sh"

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
