# Sourced by the run_eval_* scripts: turn TASK_LIST into Harbor -i flags.
#
#   TASK_LIST=<file>   run only these task names (normally the oracle-gated list)
#   TASK_LIST=''       run every task under $TASKS, on purpose
#   SMOKE=n            the *first* n of the list -- proving a path, not measuring
#   SAMPLE=n           a *random* n of it; SAMPLE_SEED=k repeats the same draw
#
# Sets `include_flags`, and expects TASK_LIST and PART to be set. Sourced rather
# than exec'd because it returns an array.
#
# Why SAMPLE exists next to SMOKE. Every list here is sorted, so SMOKE takes the
# head of an alphabet: on SWE-bench that is 50 astropy instances, and on SWE-smith
# it is 50 tasks off one repository -- one image, one runtime, one conda
# environment. As a shakedown that is the point. As a measurement it is worthless,
# and the failure is invisible because 50 trials did run. SAMPLE draws across the
# whole list, so 50 SWE-smith tasks land on ~50 different images.
#
# Why the size guard. Harbor takes task names as repeated -i flags and offers no
# file form, so the 39,686-task SWE-smith allowlist is ~2.3 MB of argv against an
# ARG_MAX of 2 MB. Without the guard the run dies inside execve with E2BIG, which
# reads like a broken sandbox rather than a command line that is too long. There is
# no way to filter that many tasks through this interface: either point TASKS at a
# directory that already holds only what you want (part 4's make_train_set.sh
# builds one out of the same allowlist), or SAMPLE it.
ARGV_BUDGET_BYTES="${ARGV_BUDGET_BYTES:-1500000}"

if [ -n "${SMOKE:-}" ] && [ -n "${SAMPLE:-}" ]; then
  echo "set SMOKE or SAMPLE, not both: SMOKE takes the head of the list, SAMPLE draws from it" >&2
  exit 1
fi

_names=()
if [ -n "${SMOKE:-}${SAMPLE:-}" ] && [ ! -r "${TASK_LIST:-}" ]; then
  echo "SMOKE/SAMPLE need a readable TASK_LIST: ${TASK_LIST:-<unset>}" >&2
  exit 1
fi

if [ -n "${SMOKE:-}" ]; then
  mapfile -t _names < <(grep -v '^[[:space:]]*$' "$TASK_LIST" | head -n "$SMOKE")
  echo "SMOKE=$SMOKE: the first ${#_names[@]} task(s) of $(basename "$TASK_LIST")"
  echo "  (the head of a sorted list, so expect them to share a repository -- use SAMPLE to measure)"
elif [ -n "${SAMPLE:-}" ]; then
  # Named explicitly: without it the draw fails inside a process substitution, which
  # surfaces as "selected 0 tasks" several lines later and reads like a bad list.
  [ -x "${PART:-}/.venv/bin/python" ] || {
    echo "SAMPLE needs \$PART/.venv/bin/python; PART=${PART:-<unset>}" >&2
    echo "source this from a run_eval_* script, or run 'uv sync' in part3-evaluate" >&2
    exit 1
  }
  mapfile -t _names < <("$PART/.venv/bin/python" - "$TASK_LIST" "$SAMPLE" "${SAMPLE_SEED:-0}" <<'PY'
import random, sys
names = sorted({line.strip() for line in open(sys.argv[1]) if line.strip()})
random.Random(int(sys.argv[3])).shuffle(names)
print("\n".join(sorted(names[: int(sys.argv[2])])))
PY
)
  echo "SAMPLE=$SAMPLE seed=${SAMPLE_SEED:-0}: ${#_names[@]} of $(grep -c . "$TASK_LIST") task(s), drawn across the list"
elif [ -n "${TASK_LIST:-}" ]; then
  # A TASK_LIST that cannot be read is a mistake, not a request for everything.
  if [ ! -r "$TASK_LIST" ] || [ ! -s "$TASK_LIST" ]; then
    echo "TASK_LIST is set but not a readable non-empty file: $TASK_LIST" >&2
    echo "to run the whole task set on purpose, pass TASK_LIST=''" >&2
    exit 1
  fi
  mapfile -t _names < <(grep -v '^[[:space:]]*$' "$TASK_LIST")
fi

include_flags=()
_argv_bytes=0
for _name in ${_names[@]+"${_names[@]}"}; do
  include_flags+=(-i "*$_name")
  _argv_bytes=$((_argv_bytes + ${#_name} + 5))
done

if [ "$_argv_bytes" -gt "$ARGV_BUDGET_BYTES" ]; then
  echo "${#_names[@]} task names is ~$((_argv_bytes / 1000)) kB of argv, over the" >&2
  echo "${ARGV_BUDGET_BYTES} byte budget (ARG_MAX here is $(getconf ARG_MAX)). Harbor has no" >&2
  echo "file form for -i, so this cannot be filtered on the command line. Either:" >&2
  echo "  SAMPLE=n   measure a random subset of it, or" >&2
  echo "  TASKS=<dir> point at a directory holding only the tasks you want" >&2
  echo "             (part4-train/scripts/make_train_set.sh builds one from this list)" >&2
  exit 1
fi

# Selecting nothing must never mean "select everything". Only TASK_LIST='' asks for
# the whole set; anything else reaching zero is a filter that silently failed, and
# the cost of guessing wrong is a 45,844-task run instead of a 50-task one. This is
# the same trap the TASK_LIST check above closes, one step further along -- it also
# catches `mapfile: command not found` from sourcing this into a shell that is not
# bash.
if [ "${#include_flags[@]}" -eq 0 ]; then
  if [ -n "${TASK_LIST:-}${SMOKE:-}${SAMPLE:-}" ]; then
    echo "the filter selected 0 tasks, which is not a request to run all of them." >&2
    echo "  TASK_LIST=${TASK_LIST:-<unset>} SMOKE=${SMOKE:-} SAMPLE=${SAMPLE:-}" >&2
    echo "to run every task under \$TASKS on purpose, pass TASK_LIST='' with no SMOKE/SAMPLE" >&2
    exit 1
  fi
  echo "no instance filter: running every task under $TASKS" >&2
fi
