#!/usr/bin/env bash
# Clone SkyRL and repoint its Harbor pin at the kit branch.
#
#   scripts/setup_skyrl.sh
#
# This one edit is the difference between a training run that works and one that
# silently produces nothing, so it gets its own script rather than a README step.
#
# SkyRL pins Harbor to its own revision in pyproject.toml -- a git rev these days
# (laude-institute/harbor@3de07a0e as of 2026-09-28), not a version -- and that
# revision's EnvironmentType enum has no `agentcore`. Every rollout then fails
# validation *before* it creates a sandbox:
#
#   1 validation error for TrialConfig
#   environment.type
#     Input should be 'docker', 'daytona', 'e2b', ... [input_value='agentcore']
#
# and the training loop completes the step anyway, writes a checkpoint, and
# reports:
#
#   avg_raw_reward: 0.0     response_length: 1.0     policy_loss: 0.0
#
# `response_length: 1.0` is the only tell. Nothing errors out.
#
# The pin must be changed in pyproject.toml, not with `uv pip install`: SkyRL is
# launched via `uv run --extra fsdp --extra harbor`, which re-syncs from the lock
# and would restore SkyRL's own pin.
#
# SkyRL's contact surface with Harbor is three imports (TrialConfig,
# Trial.create/run, RolloutDetail) whose signatures are unchanged in 0.23.0, so
# swapping the pin is safe.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART="$(cd "$HERE/.." && pwd)"
set -a; . "$PART/../config.env"; set +a

SKYRL_DIR="${SKYRL_DIR:-$KIT_WORK_DIR/skyrl}"
SKYRL_REPO="${SKYRL_REPO:-https://github.com/NovaSky-AI/SkyRL.git}"

if [ ! -d "$SKYRL_DIR/.git" ]; then
  echo "cloning SkyRL into $SKYRL_DIR"
  git clone --depth 1 "$SKYRL_REPO" "$SKYRL_DIR"
fi

cd "$SKYRL_DIR"
[ -f pyproject.toml ] || { echo "no pyproject.toml in $SKYRL_DIR" >&2; exit 1; }

cp pyproject.toml pyproject.toml.orig-harbor-pin

"$PART/.venv/bin/python" - "$HARBOR_GIT_REF" <<'PY'
import re
import sys
import tomllib
from pathlib import Path

ref = sys.argv[1]
path = Path("pyproject.toml")
text = path.read_text()
before = text

# 1. Point the harbor source at the kit ref. This one is a single-line inline
#    table in [tool.uv.sources], so matching to the closing brace is safe.
text = re.sub(
    r'^harbor\s*=\s*\{[^}]*\}',
    f'harbor = {{ git = "{ref.split("git+")[-1].split("@")[0]}", rev = "{ref.rsplit("@", 1)[-1]}" }}',
    text,
    flags=re.M,
)

# 2. Request the agentcore extra, by editing the *requirement string* rather than
#    the list that holds it.
#
#    Do not be tempted back into rewriting the list. `^harbor\s*=\s*\[[^\]]*\]`
#    looks right and is not: `[^\]]` matches newlines, so it runs past
#
#        harbor = [
#            "harbor[daytona,modal]",
#        ]
#
#    and stops at the first `]` it finds -- the one closing the *extras* in
#    `harbor[daytona,modal]`. The replacement then leaves a stray `",` and an
#    orphan `]` behind, and `uv sync` fails with a TOML parse error 200 lines
#    from anything that mentions harbor. It only ever worked because upstream
#    happened to keep the list on one line; it is wrong for either shape.
#    Neither pattern requires the string to *end* after the extras, because a
#    requirement may carry a version specifier ("harbor[daytona,modal]==0.13.1").
text = re.sub(
    r'"harbor\[(?!agentcore)([^"\]]*)\]',
    r'"harbor[agentcore,\1]',
    text,
)
text = re.sub(r'"harbor(?=["=<>~!])', '"harbor[agentcore]', text)

if text == before:
    print("WARNING: no harbor pin matched; inspect pyproject.toml by hand", file=sys.stderr)
    raise SystemExit(1)

# The check that would have caught the bug above: a rewrite that produces invalid
# TOML must fail here, not inside uv several steps later. The backup this script
# already made is the way back.
try:
    tomllib.loads(text)
except tomllib.TOMLDecodeError as err:
    print(f"the rewrite produced invalid TOML and was not written: {err}", file=sys.stderr)
    print("pyproject.toml is untouched; pyproject.toml.orig-harbor-pin is the backup",
          file=sys.stderr)
    raise SystemExit(1) from err

path.write_text(text)
for n, line in enumerate(text.splitlines(), 1):
    if "harbor[" in line or line.startswith("harbor "):
        print(f"  {n}: {line.strip()}")
PY

echo
echo "syncing SkyRL (this pulls torch and vllm; expect tens of minutes cold)"
UV_CACHE_DIR="$UV_CACHE_DIR" uv sync --extra fsdp --extra harbor

echo
echo "verifying the pin took:"
uv run --extra fsdp --extra harbor python -c "
from harbor.models.environment_type import EnvironmentType
values = [e.value for e in EnvironmentType]
assert 'agentcore' in values, f'agentcore missing -- pin did not take: {values}'
import harbor; print('  harbor', getattr(harbor, '__version__', '?'), 'with agentcore: OK')
"
echo
echo "SKYRL_DIR=$SKYRL_DIR"
