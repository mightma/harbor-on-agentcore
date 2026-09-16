#!/usr/bin/env bash
# End-to-end: real vLLM behind the recording proxy, then attach to a trial.
#
#   scripts/e2e_vllm.sh                       # $POLICY_MODEL
#   scripts/e2e_vllm.sh Qwen/Qwen3-1.7B       # anything small is fine
#   VLLM=/path/to/vllm scripts/e2e_vllm.sh    # reuse an existing vLLM install
#
# Needs one GPU and a model on disk; needs no sandbox, no AWS and no docker. It
# checks the assumption the whole proxy rests on -- that vLLM answers
# return_token_ids with prompt_token_ids and per-choice token_ids -- and then runs
# the attach seam over a synthetic trial directory.
#
# For the parts this cannot cover (an installed harness inside an AgentCore
# session reaching the proxy over VPC mode) see README.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART="$(cd "$HERE/.." && pwd)"
# config.env sets HF_HOME to the kit's own scratch; a caller pointing at an
# existing model cache should win, or this test re-downloads weights it has.
_hf_home_pre="${HF_HOME:-}"
[ -f "$PART/../config.env" ] && { set -a; . "$PART/../config.env"; set +a; }
[ -n "$_hf_home_pre" ] && export HF_HOME="$_hf_home_pre"

MODEL="${1:-${POLICY_MODEL:-Qwen/Qwen3-1.7B}}"
SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL")}"
VLLM="${VLLM:-$PART/.venv/bin/vllm}"
PORT="${PORT:-8000}"
PROXY_PORT="${PROXY_PORT:-8010}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
SERVE_GPUS="${SERVE_GPUS:-0}"
WORK="${WORK:-$(mktemp -d)}"
mkdir -p "$WORK"   # mktemp made its own; a caller-supplied one may not exist yet
RECORD="$WORK/rollouts.jsonl"

if [ ! -x "$VLLM" ]; then
  echo "no vllm at $VLLM -- set VLLM=<path to a vllm binary>" >&2
  exit 1
fi

echo "model=$MODEL served=$SERVED_NAME work=$WORK"

# enforce-eager: this is a smoke test, and torch.compile would spend minutes and
# hundreds of MB of cache to make a handful of requests marginally faster.
CUDA_VISIBLE_DEVICES="$SERVE_GPUS" "$VLLM" serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.5 \
  --enforce-eager \
  --no-enable-log-requests \
  > "$WORK/vllm.log" 2>&1 &
vllm_pid=$!

python3 "$HERE/record_proxy.py" \
  --upstream "http://127.0.0.1:$PORT/v1" \
  --host 127.0.0.1 --port "$PROXY_PORT" \
  --record "$RECORD" \
  > "$WORK/proxy.log" 2>&1 &
proxy_pid=$!

cleanup() {
  kill "$vllm_pid" "$proxy_pid" 2>/dev/null || true
  wait "$vllm_pid" 2>/dev/null || true
}
trap cleanup EXIT

echo "waiting for vllm (log: $WORK/vllm.log)"
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  if ! kill -0 "$vllm_pid" 2>/dev/null; then
    echo "vllm died; tail of its log:" >&2
    tail -20 "$WORK/vllm.log" >&2
    exit 1
  fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PROXY_PORT/v1/models" >/dev/null \
  || { echo "proxy not answering; see $WORK/proxy.log" >&2; exit 1; }
echo "vllm and proxy both up"

python3 "$HERE/e2e_probe.py" \
  --proxy "http://127.0.0.1:$PROXY_PORT/v1" \
  --upstream "http://127.0.0.1:$PORT/v1" \
  --record "$RECORD" \
  --model "$SERVED_NAME"
probe_rc=$?

# The seam, over a synthetic trial: a real one needs a sandbox, but the file this
# writes into is exactly a harbor trial's result.json.
echo
echo "attach seam against a synthetic job dir"
job="$WORK/job"
trial="$job/e2e__probe"
mkdir -p "$trial/agent"
printf '{"agent_result": {"rollout_details": null}}' > "$trial/result.json"
printf '{"agent": {"name": "mini-swe-agent"}}' > "$trial/config.json"
python3 - "$RECORD" "$trial/agent/trajectory.json" <<'PY'
import json, sys
first = json.loads(open(sys.argv[1]).readline())
# The proxy stores a hash and a 200-char head; the head is enough for the synthetic
# trial because the probe's instruction is shorter than that.
json.dump({"steps": [{"source": "user", "message": first["first_user_head"]}]},
          open(sys.argv[2], "w"))
PY
python3 "$HERE/attach_rollouts.py" "$job" --record "$RECORD" || true
python3 - "$trial/result.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
details = d["agent_result"]["rollout_details"]
assert isinstance(details, list) and details, "rollout_details not attached"
one = details[0]
n = len(one["prompt_token_ids"])
assert n == len(one["completion_token_ids"]) == len(one["logprobs"]), "ragged RolloutDetail"
print(f"  ok    trial result.json now carries {n} turn(s) of token ids and logprobs")
PY

echo
echo "artifacts: $WORK (vllm.log, proxy.log, rollouts.jsonl, job/)"
exit "$probe_rc"
