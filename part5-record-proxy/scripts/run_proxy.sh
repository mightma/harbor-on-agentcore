#!/usr/bin/env bash
# Run the recording proxy next to a vLLM the sandbox cannot see, on an address it
# can.
#
#   scripts/run_proxy.sh                          # proxy :8010 -> vLLM :8000
#   UPSTREAM=http://127.0.0.1:8001/v1 scripts/run_proxy.sh
#
# Prints the two environment variables a Harbor job needs so an installed harness
# inside the sandbox sends its model calls here. Start vLLM yourself (part 3's
# scripts/run_eval_vllm.sh serves one, or SkyRL owns the engine in part 4).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART="$(cd "$HERE/.." && pwd)"
[ -f "$PART/../config.env" ] && { set -a; . "$PART/../config.env"; set +a; }

UPSTREAM="${UPSTREAM:-http://127.0.0.1:8000/v1}"
PROXY_PORT="${PROXY_PORT:-8010}"
RECORD="${RECORD:-${HARBOR_JOBS:-/tmp}/rollouts-$(date +%Y%m%d-%H%M%S).jsonl}"

# The sandbox reaches the host on its private address; 127.0.0.1 is the sandbox
# itself. This is the value that goes into OPENAI_BASE_URL.
PRIVATE_IP="${PRIVATE_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"

cat <<INFO
upstream : $UPSTREAM
proxy    : http://0.0.0.0:$PROXY_PORT   (private: http://$PRIVATE_IP:$PROXY_PORT)
record   : $RECORD

For a harbor job using an installed harness, export these on the HOST -- harbor
injects them into the sandbox:

  export OPENAI_BASE_URL="http://$PRIVATE_IP:$PROXY_PORT/v1"
  export OPENAI_API_BASE="\$OPENAI_BASE_URL"
  export MSWEA_API_KEY=unused-but-required

and use configs/rollout-mini-swe-agent.yaml, whose network_mode: VPC is what gives
the sandbox a route to this address. Afterwards:

  scripts/attach_rollouts.py "\$HARBOR_JOBS/<job>" --record "$RECORD"

INFO

exec python3 "$HERE/record_proxy.py" \
  --upstream "$UPSTREAM" \
  --port "$PROXY_PORT" \
  --record "$RECORD" \
  "$@"
