#!/usr/bin/env bash
# Is this training run producing rewards, and if not, which layer is eating them?
#
#   scripts/check_run.sh                      # the most recent run
#   scripts/check_run.sh <run-name-or-path>
#
# Run it while training is going. A reward of 0 can originate in any of three layers
# and only the trial directory distinguishes them:
#
#   trials with no result.json at all   the trial raised before finishing. Nothing
#                                       about the model. The cause is in the infra
#                                       log, which this script extracts.
#   trials that finished with reward 0  the sandbox and verifier worked and the
#                                       patch did not pass. That is either a real
#                                       model failure or a dead reward path -- part
#                                       2's gate is what separates those.
#   no trials at all                    the generator never got that far.
#
# This exists because that first case is invisible from the training metrics.
# **[measured]** one run reached 1152 trial directories, 0 completions and 5 saved
# checkpoints while reporting a reward of 0 for everything: the execution role never
# reached the Ray workers, so every trial raised before creating a sandbox. The loop
# does not care, and neither does the loss.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART="$(cd "$HERE/.." && pwd)"
set -a; . "$PART/../config.env"; set +a

RUN="${1:-}"
if [ -z "$RUN" ]; then
  RUN="$(ls -dt "$KIT_WORK_DIR"/runs/*/ 2>/dev/null | head -1)"
  [ -n "$RUN" ] || { echo "no runs under $KIT_WORK_DIR/runs" >&2; exit 1; }
elif [ ! -d "$RUN" ]; then
  RUN="$KIT_WORK_DIR/runs/$RUN"
fi
RUN="${RUN%/}"
[ -d "$RUN" ] || { echo "no such run: $RUN" >&2; exit 1; }
echo "run: $RUN"

trials="$RUN/trials"
if [ ! -d "$trials" ]; then
  echo "  no trials/ directory -- the generator has not created a trial yet"
  exit 0
fi

n_dirs=$(find "$trials" -mindepth 1 -maxdepth 1 -type d | wc -l)
n_done=$(find "$trials" -mindepth 2 -maxdepth 2 -name result.json | wc -l)
echo "  trial dirs      : $n_dirs"
echo "  completed       : $n_done"

if [ "$n_done" -gt 0 ]; then
  "$PART/.venv/bin/python" - "$trials" <<'PY'
import json, pathlib, sys, collections
root = pathlib.Path(sys.argv[1])
rew, exc = collections.Counter(), collections.Counter()
no_rd = 0
for p in root.glob("*/result.json"):
    d = json.loads(p.read_text())
    rew[((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")] += 1
    if e := (d.get("exception_info") or {}).get("exception_type"):
        exc[e] += 1
    if not (d.get("agent_result") or {}).get("rollout_details"):
        no_rd += 1
total = sum(rew.values())
ones = rew.get(1.0, 0)
print(f"  reward 1.0      : {ones}/{total}" + (f" = {100*ones/total:.1f}%" if total else ""))
print(f"  reward dist     : {dict(sorted((k, v) for k, v in rew.items() if k is not None))}")
if exc:
    print(f"  exceptions      : {dict(exc)}")
# Step-wise RL needs per-turn token ids and logprobs; without them the gradient path
# is dead even when the reward is not.
if no_rd:
    print(f"  !! rollout_details empty in {no_rd}/{total} -- step-wise RL has no token ids")
PY
fi

if [ "$n_done" -lt "$n_dirs" ]; then
  echo
  echo "  $((n_dirs - n_done)) trial(s) produced no result.json."
  empty=$(find "$trials" -name trial.log -size 0 | wc -l)
  echo "  0-byte trial.log: $empty  (a trial that raised before starting its sandbox)"
  # The generator logs one WARNING per failed trajectory. It lands in the *Ray worker*
  # log, not in the run's own infra log -- the trial runs inside the worker, which is
  # the same reason the role ARN had to travel in the config. Search both.
  logs=()
  while IFS= read -r f; do logs+=("$f"); done < <(
    ls -t "$RUN"/logs/infra-*.log 2>/dev/null
    ls -t "$TMPDIR"/ray/session_*/logs/worker-*.err 2>/dev/null | head -40
  )
  if [ "${#logs[@]}" -gt 0 ]; then
    echo
    echo "  distinct failure reasons (generator warnings, collapsed):"
    # Strip ANSI colour, the per-trajectory ids and the trailing "Results: None", so
    # 1792 identical failures collapse to one line with a count.
    found=$(grep -ohE "attempt [0-9]+/[0-9]+ failed: .*" "${logs[@]}" 2>/dev/null \
      | sed -E 's/\x1b\[[0-9;]*m//g; s/^attempt [0-9]+\/[0-9]+ failed: //; s/\. Results:.*$//' \
      | sed -E 's/[0-9a-f]{8,}/<id>/g' \
      | cut -c1-200 | sort | uniq -c | sort -rn | head -5)
    if [ -n "$found" ]; then
      echo "$found" | sed 's/^/    /'
    else
      echo "    (none found -- look in $TMPDIR/ray/session_*/logs/worker-*.err)"
    fi
  fi
fi
