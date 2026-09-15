#!/usr/bin/env bash
# Evaluate a Bedrock model on the arm64 slice of SWE-bench Verified, with every
# trial's sandbox an AgentCore Runtime session.
#
#   scripts/run_eval_bedrock.sh                                  # $EVAL_BEDROCK_MODEL
#   scripts/run_eval_bedrock.sh us.anthropic.claude-sonnet-5
#   N_CONCURRENT=8 scripts/run_eval_bedrock.sh                   # go gentler
#
# No GPU and no local serving: terminus-2 runs in this process and calls Bedrock
# with this host's credentials. The sandbox only ever sees shell commands, so its
# execution role needs no bedrock:InvokeModel.
#
# Prerequisites: part 1 generated the task dirs and pulled the arm64 bases; part 2
# deployed the runtimes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART="$(cd "$HERE/.." && pwd)"
set -a; . "$PART/../config.env"; set +a

MODEL="${1:-$EVAL_BEDROCK_MODEL}"
TASKS="${TASKS:-$HARBOR_DATASETS/swebv-arm64/eval}"
TASK_LIST="${TASK_LIST:-$PART/../part1-arm64-images/data/swebv-arm64-eval-verified.txt}"
N_CONCURRENT="${N_CONCURRENT:-16}"
JOB_NAME="${JOB_NAME:-swebv-bedrock-$(basename "$MODEL")-$(date +%Y%m%d-%H%M%S)}"

export HARBOR_AGENTCORE_ROLE_ARN="$ACR_EXECUTION_ROLE_ARN"

config="$TMPDIR/eval_bedrock_$JOB_NAME.yaml"
sed "s|bedrock/PLACEHOLDER|bedrock/$MODEL|" "$PART/configs/eval-bedrock.yaml" > "$config"

# Restrict to the oracle-verified subset. Three of the 73 eval tasks have a
# ceiling of 0 -- two sphinx instances whose PASS_TO_PASS tests fail regardless of
# the patch, and psf__requests-2317 whose verifier hangs on network calls.
# Including them just subtracts a constant from the score.
include_flags=()
if [ -f "$TASK_LIST" ]; then
  while read -r instance; do
    [ -n "$instance" ] && include_flags+=(-i "*$instance")
  done < "$TASK_LIST"
fi

echo "job=$JOB_NAME model=bedrock/$MODEL"
echo "tasks=$(find "$TASKS" -mindepth 1 -maxdepth 1 -type d | wc -l) from $TASKS" \
     "selected=$((${#include_flags[@]} / 2)) n=$N_CONCURRENT"

# Harbor's own AWS calls read the instance role directly. Static credentials in the
# environment are how a long job dies halfway through when the session token
# expires, so make sure LiteLLM and boto3 both use the role.
unset AWS_BEARER_TOKEN_BEDROCK

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
