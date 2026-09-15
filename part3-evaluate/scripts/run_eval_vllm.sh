#!/usr/bin/env bash
# Evaluate a locally served open-weights policy on the arm64 slice of SWE-bench
# Verified, with every trial's sandbox an AgentCore Runtime session.
#
#   scripts/run_eval_vllm.sh                                 # $POLICY_MODEL
#   scripts/run_eval_vllm.sh Qwen/Qwen3.5-4B
#   scripts/run_eval_vllm.sh "$KIT_WORK_DIR/runs/.../exports/global_step_20"
#   SERVE_GPUS=0,1 DP=2 scripts/run_eval_vllm.sh              # bigger policy
#
# Same scaffold as run_eval_bedrock.sh on purpose, so the two numbers compare.
# The model call goes to vLLM on 127.0.0.1: the sandbox never needs a network path
# back to this host, and no credentials or endpoints are injected into it.
#
# Prerequisites: part 1 generated the task dirs and pulled the arm64 bases; part 2
# deployed the runtimes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART="$(cd "$HERE/.." && pwd)"
set -a; . "$PART/../config.env"; set +a

MODEL="${1:-$POLICY_MODEL}"
SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL")}"
TASKS="${TASKS:-$HARBOR_DATASETS/swebv-arm64/eval}"
TASK_LIST="${TASK_LIST:-$PART/../part1-arm64-images/data/swebv-arm64-eval-verified.txt}"
N_CONCURRENT="${N_CONCURRENT:-24}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
PORT="${PORT:-8000}"
SERVE_GPUS="${SERVE_GPUS:-0}"
DP="${DP:-1}"
# Point this at any vLLM install to skip `uv sync --extra serve`, e.g. part 4's
# SkyRL venv.
VLLM="${VLLM:-$PART/.venv/bin/vllm}"
JOB_NAME="${JOB_NAME:-swebv-vllm-$SERVED_NAME-$(date +%Y%m%d-%H%M%S)}"

export HARBOR_AGENTCORE_ROLE_ARN="$ACR_EXECUTION_ROLE_ARN"

if [ ! -x "$VLLM" ]; then
  echo "no vllm at $VLLM -- run 'uv sync --extra serve' or set VLLM=<path>" >&2
  exit 1
fi

config="$TMPDIR/eval_vllm_$JOB_NAME.yaml"
sed "s|hosted_vllm/PLACEHOLDER|hosted_vllm/$SERVED_NAME|; s|127.0.0.1:8000|127.0.0.1:$PORT|" \
  "$PART/configs/eval-vllm.yaml" > "$config"

include_flags=()
if [ -f "$TASK_LIST" ]; then
  while read -r instance; do
    [ -n "$instance" ] && include_flags+=(-i "*$instance")
  done < "$TASK_LIST"
fi

echo "job=$JOB_NAME model=$MODEL served=$SERVED_NAME"
echo "tasks=$(find "$TASKS" -mindepth 1 -maxdepth 1 -type d | wc -l) from $TASKS" \
     "selected=$((${#include_flags[@]} / 2)) n=$N_CONCURRENT gpus=$SERVE_GPUS dp=$DP"

echo "serving $MODEL as $SERVED_NAME on :$PORT"
CUDA_VISIBLE_DEVICES="$SERVE_GPUS" "$VLLM" serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --data-parallel-size "$DP" \
  --gpu-memory-utilization 0.85 \
  --no-enable-log-requests \
  > "$HARBOR_JOBS/$JOB_NAME.vllm.log" 2>&1 &
vllm_pid=$!
trap 'kill $vllm_pid 2>/dev/null || true' EXIT

for _ in $(seq 1 240); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  if ! kill -0 $vllm_pid 2>/dev/null; then
    echo "vllm died; see $HARBOR_JOBS/$JOB_NAME.vllm.log" >&2
    exit 1
  fi
  sleep 5
done
echo "vllm ready"

unset AWS_BEARER_TOKEN_BEDROCK

# The provider pushes each wrapped image to ECR on first use, and `docker push`
# needs a login. Do it once here rather than have 24 concurrent trials discover it.
aws ecr get-login-password --region "$AWS_REGION" 2>/dev/null \
  | docker login --username AWS --password-stdin \
      "$(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com" \
      >/dev/null 2>&1 || echo "warning: ECR login failed; the provider will retry its own" >&2

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
