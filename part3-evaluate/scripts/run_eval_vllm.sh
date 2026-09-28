#!/usr/bin/env bash
# Evaluate a locally served open-weights policy on the arm64 slice of SWE-bench
# Verified, with every trial's sandbox an AgentCore Runtime session.
#
#   scripts/run_eval_vllm.sh                                 # $POLICY_MODEL
#   scripts/run_eval_vllm.sh Qwen/Qwen3.5-4B
#   scripts/run_eval_vllm.sh "$KIT_WORK_DIR/runs/.../exports/global_step_20"
#   SERVE_GPUS=0,1 DP=2 scripts/run_eval_vllm.sh              # two replicas
#   SERVE_GPUS=0,1,2,3 TP=4 scripts/run_eval_vllm.sh          # one model, four GPUs
#
# DP or TP, and the difference matters once the policy is large: DP puts a *whole
# copy* on each GPU and serves them round-robin, TP splits one copy across them.
# A mixture of experts is where this bites, because what has to fit is the total
# parameter count and not the active one -- Qwen3.5-35B-A3B activates 3B per token
# but is 35.95B of weights, i.e. **72 GB at bf16**. That still fits one 143 GB H200
# (DP=1, TP=1, leaving ~50 GB of KV cache at 0.85 utilisation), fits nowhere on an
# 80 GB H100, and with TP=4 drops to 18 GB a GPU and leaves room for a long
# context. Set SERVE_GPUS to as many GPUs as DP*TP.
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
TASKS="${TASKS:-$HARBOR_DATASETS/swebv-arm64}"
TASK_LIST="${TASK_LIST:-$PART/../part1-arm64-images/swebench/data/swebv-arm64-gated.txt}"
N_CONCURRENT="${N_CONCURRENT:-24}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
PORT="${PORT:-8000}"
SERVE_GPUS="${SERVE_GPUS:-0}"
DP="${DP:-1}"
TP="${TP:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
# Point this at any vLLM install to skip `uv sync --extra serve`, e.g. part 4's
# SkyRL venv.
VLLM="${VLLM:-$PART/.venv/bin/vllm}"
# Parity with run_eval_bedrock.sh, which had this and this script did not -- so the
# vLLM path could not be run against local containers at all. Setting SANDBOX here
# without the -e below is precisely how this kit came to publish a docker comparison
# that had run on AgentCore, so the two belong in the same edit.
SANDBOX="${SANDBOX:-agentcore}"
JOB_NAME="${JOB_NAME:-swebv-vllm-$SERVED_NAME-$(date +%Y%m%d-%H%M%S)}"

export HARBOR_AGENTCORE_ROLE_ARN="$ACR_EXECUTION_ROLE_ARN"

# FlashInfer's top-k/top-p sampler is JIT-compiled on first use, and "first use" is
# vLLM's own warmup -- so without this the server dies *after* loading 72 GB of
# weights, with a traceback whose visible end is the useless "Engine core
# initialization failed. See root cause above." The root cause, 200 lines up:
#
#   flashinfer/jit/cpp_ext.py, in get_cuda_path
#   RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
#                 doesn't exist
#
# This is the same failure shape as part 4's FLA_TILELANG=0: a library prefers a
# JIT kernel, checks only loosely for a toolchain, and fails at codegen. The same
# remedy applies -- take the PyTorch-native path.
#
# Nothing is lost here. The eval runs at temperature 0.0, so a fused top-k/top-p
# kernel is sampling from a distribution it never consults; even for part 4's
# rollouts the native path differs in throughput, not in correctness.
#
# The alternative is to give FlashInfer a toolchain: torch ships one inside the
# venv at nvidia/cu13 (a real nvcc, ptxas and headers), so CUDA_HOME=<that> also
# gets past this. It is not the default because that nvcc is 13.4 against a torch
# built for 13.0, which is how you buy "CUDA compiler and CUDA toolkit headers are
# incompatible" a few minutes later instead of now.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

if [ ! -x "$VLLM" ]; then
  echo "no vllm at $VLLM -- run 'uv sync --extra serve' or set VLLM=<path>" >&2
  exit 1
fi

config="$TMPDIR/eval_vllm_$JOB_NAME.yaml"
sed "s|hosted_vllm/PLACEHOLDER|hosted_vllm/$SERVED_NAME|; s|127.0.0.1:8000|127.0.0.1:$PORT|" \
  "$PART/configs/eval-vllm.yaml" > "$config"

# SMOKE / SAMPLE narrow the gated list further; see select_tasks.sh for which
# to use. A list that cannot be read is a mistake there, not "run everything".
. "$HERE/select_tasks.sh"

echo "job=$JOB_NAME model=$MODEL served=$SERVED_NAME sandbox=$SANDBOX"
echo "pool=$(find "$TASKS" -mindepth 1 -maxdepth 1 -type d | wc -l) from $TASKS" \
     "selected=$((${#include_flags[@]} / 2)) n=$N_CONCURRENT gpus=$SERVE_GPUS dp=$DP tp=$TP"
# pool is how many task dirs exist, selected is how many will actually run: the task
# directory is generated before the gate, so it still holds the tasks the gate later
# rejected. TASK_LIST is what excludes them.

# Catch the mismatch here: vLLM otherwise starts, spends minutes loading weights,
# and only then fails to place the grid -- or worse, silently uses fewer GPUs than
# you are paying for.
n_gpus=$(awk -F, '{print NF}' <<< "$SERVE_GPUS")
if [ "$((DP * TP))" -ne "$n_gpus" ]; then
  echo "DP*TP ($DP*$TP) must equal the number of GPUs in SERVE_GPUS ($n_gpus: $SERVE_GPUS)" >&2
  exit 1
fi

echo "serving $MODEL as $SERVED_NAME on :$PORT (dp=$DP tp=$TP gpus=$SERVE_GPUS)"
CUDA_VISIBLE_DEVICES="$SERVE_GPUS" "$VLLM" serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --data-parallel-size "$DP" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
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
