#!/usr/bin/env bash
# Evaluate a Bedrock model on the arm64 slice of SWE-bench Verified, with every
# trial's sandbox an AgentCore Runtime session.
#
#   scripts/run_eval_bedrock.sh                                  # $EVAL_BEDROCK_MODEL
#   scripts/run_eval_bedrock.sh us.anthropic.claude-sonnet-5
#   N_CONCURRENT=8 scripts/run_eval_bedrock.sh                   # go gentler
#   SANDBOX=docker N_CONCURRENT=12 scripts/run_eval_bedrock.sh   # same scaffold, local containers
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
TASKS="${TASKS:-$HARBOR_DATASETS/swebv-arm64}"
TASK_LIST="${TASK_LIST:-$PART/../part1-arm64-images/swebench/data/swebv-arm64-gated.txt}"
N_CONCURRENT="${N_CONCURRENT:-16}"
# SANDBOX=docker runs the identical scaffold against local containers instead of
# AgentCore sessions, which is the only honest way to compare the two: one config,
# one switch, so the two runs cannot drift apart in anything but the sandbox.
#
# The ceiling moves when you flip it. On AgentCore the limit is the service's
# session-creation rate and each rollout gets its own 2 vCPU microVM, so this host
# stays idle; on docker every container is 2 vCPU of *this* box, so N_CONCURRENT
# above ~vCPU/2 only makes each trial slower. Set it deliberately.
SANDBOX="${SANDBOX:-agentcore}"
JOB_NAME="${JOB_NAME:-swebv-bedrock-$(basename "$MODEL")-$(date +%Y%m%d-%H%M%S)}"

export HARBOR_AGENTCORE_ROLE_ARN="$ACR_EXECUTION_ROLE_ARN"

config="$TMPDIR/eval_bedrock_$JOB_NAME.yaml"
sed "s|bedrock/PLACEHOLDER|bedrock/$MODEL|" "$PART/configs/eval-bedrock.yaml" > "$config"

# Restrict to the oracle-verified subset. Three of the 73 eval tasks have a
# ceiling of 0 -- two sphinx instances whose PASS_TO_PASS tests fail regardless of
# the patch, and psf__requests-2317 whose verifier hangs on network calls.
# Including them just subtracts a constant from the score.
# SMOKE=n runs only the first n instances of the list -- for proving a path works
# before paying for all 70. Do NOT try to get that effect by pointing TASK_LIST
# somewhere empty: an unreadable list used to mean "no filters", which silently ran
# the whole set.
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

echo "job=$JOB_NAME model=bedrock/$MODEL sandbox=$SANDBOX"
echo "pool=$(find "$TASKS" -mindepth 1 -maxdepth 1 -type d | wc -l) from $TASKS" \
     "selected=$((${#include_flags[@]} / 2)) n=$N_CONCURRENT"
# pool is how many task dirs exist, selected is how many will actually run: the task
# directory is generated before the gate, so it still holds the tasks the gate later
# rejected. TASK_LIST is what excludes them.

# Harbor's own AWS calls read the instance role directly. Static credentials in the
# environment are how a long job dies halfway through when the session token
# expires, so make sure LiteLLM and boto3 both use the role.
unset AWS_BEARER_TOKEN_BEDROCK

set +e
"$PART/.venv/bin/harbor" run -c "$config" -p "$TASKS" \
  "${include_flags[@]+"${include_flags[@]}"}" \
  -e "$SANDBOX" \
  -n "$N_CONCURRENT" \
  -o "$HARBOR_JOBS" --job-name "$JOB_NAME" "${@:2}" 2>&1 \
  | sed 's/\x1b\[[0-9;]*m//g' \
  | grep -avE "Running trials|starting environment|running (agent|verifier)"
rc=${PIPESTATUS[0]}
set -e

echo "results: $HARBOR_JOBS/$JOB_NAME (harbor rc=$rc)"
"$PART/.venv/bin/python" "$HERE/summarize.py" "$HARBOR_JOBS/$JOB_NAME" || true
exit "$rc"
