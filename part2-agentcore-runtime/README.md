# Part 2 · Build AgentCore runtimes, and drive a session

This part deploys one runtime per task image and shows you what a session actually is.
It is the join between part 1 and parts 3–4: both of those assume the runtimes already
exist, which is what turns a rollout's environment start from ~37 s into ~3 s.

```bash
cd part2-agentcore-runtime
uv sync
```

## What a session is

Read this before debugging anything. **\[measured\]** with `scripts/probe_session.py`:

| | |
|---|---|
| Architecture | `aarch64` — the constraint behind all of part 1 |
| Shape | 2 vCPU / 8 GB RAM / 8.8 GB disk, **fixed** |
| Isolation | one microVM per session |
| Identity | the microVM has IMDS; `.../iam/security-credentials/` returns `execution_role` |
| Egress | open; no per-session network control |
| Hot session open | **0.4 s** |
| Command round trip | **0.15–0.3 s** |
| State within a session | persists — files and background processes survive between commands |

Two behaviours will bite you if you assume otherwise:

- **Every command is a new shell.** `cd /foo` then `ls` does *not* list `/foo`. Harbor
  wraps each command, so this only matters when you drive the client yourself.
- **A command must be shell-wrapped.** The exec primitive is not a shell; pipes,
  redirects and `&&` need an explicit `bash -lc`.

The IMDS row is a security boundary, not a convenience: **anything running in the
sandbox can read the execution role's credentials** — the agent under test, and any
script the task ships. Scope that role to pulling ECR and writing logs, and grant
`bedrock:InvokeModel` only when you actually want the sandbox to call a model. Handing
the sandbox static host credentials instead (the other auth path) is strictly worse: it
exposes long-lived account credentials to code you are evaluating.

## Drive a session by hand

Point it at any deployed runtime:

```bash
uv run scripts/probe_session.py hb_a3c38cb06d80
uv run scripts/probe_session.py arn:aws:bedrock-agentcore:us-west-2:...:runtime/...
```

It creates the session, runs ~15 probes, round-trips a 1 MB and an 8 MB file, and tears
the session down. The minimal shape, if you want to embed it:

```python
from harbor.environments.agentcore.client import AgentCoreSandbox, new_session_id

sandbox = AgentCoreSandbox(
    data_client=boto3.client("bedrock-agentcore", region_name="us-west-2"),
    runtime_arn=arn,
    session_id=new_session_id("probe"),
)
await sandbox.start()
result = await sandbox.exec("uname -m", timeout_sec=60)   # -> aarch64
await sandbox.write_file("/tmp/x", b"...")
await sandbox.stop()
```

## Deploy the runtimes

```bash
# SWE-smith: 119 runtimes for 44,489 tasks
scripts/deploy_runtimes.sh "$HARBOR_DATASETS/swesmith-arm64" 20

# SWE-bench Verified arm64 eval slice: 73 runtimes
scripts/deploy_runtimes.sh "$HARBOR_DATASETS/swebv-arm64" 16
```

**\[measured\]** 119 runtimes in **19 minutes** at `-n 20`.

This runs Harbor's `oracle` agent over one task per image, which does three things in
one pass:

1. builds and pushes the sandbox-wrapped image — **once per image**, not per task,
   which is what `--ek share_by_content=true` buys;
2. creates the runtime and leaves it deployed (`--ek delete_runtime=false`);
3. **gates the reward path.** The oracle applies the golden patch, so anything scoring
   below 1.0 means that image's tests fail even when they should pass.

Step 3 is the point. In RL a dead reward path is invisible: the loop runs, checkpoints
land, metrics look plausible, and every reward is 0. Both silent failures this kit hit
would have gone unnoticed without it:

- **`import numpy` SIGILLs in an ACR session** on images built with an OpenBLAS that
  probes for CPU features the microVM does not expose. Every reward became 0. Fixed by
  pinning `OPENBLAS_CORETYPE` in generated tasks — one of the five commits on the kit
  branch.
- **Truncated golden patches.** See part 1; 16 repositories looked entirely broken
  because one sampled task's patch had been mangled.

### Read the gate

```bash
uv run scripts/gate_report.py "$HARBOR_JOBS/gate-<timestamp>"
```

Every oracle failure looks identical at the top level (`test_patch_resolved` asserting
that not all FAIL_TO_PASS passed), so the verifier log tells you nothing. `gate_report.py`
reads `agent/oracle.txt` instead and splits them by cause. **\[measured\]** on the
SWE-smith gate:

```
instances     : 119
reward 1.0    : 91
not 1.0       : 28

failures by cause:
    16  truncated patch (tooling bug -- regenerate tasks)
     8  patch applied cleanly, tests still failed
     3  patch target missing from the tree
     1  EventStreamError (infra -- retry before believing it)
```

That split is the difference between "27 dead repositories" and "one tooling bug plus
11 real unknowns". After fixing the patch bug and re-gating only the failures:

```bash
REPO_PREFIXES="$(paste -sd, failing-repos.txt)" JOB_NAME=regate \
  scripts/deploy_runtimes.sh "$HARBOR_DATASETS/swesmith-arm64" 20

uv run scripts/gate_report.py "$HARBOR_JOBS"/gate-* "$HARBOR_JOBS"/regate \
  --task-root "$HARBOR_DATASETS/swesmith-arm64" \
  --allowlist-out "$KIT_STATE_DIR"
```

**\[measured\]** **108 of 119 clean, 38,815 of 44,489 tasks** — up from 91 / 34,908.
Job dirs merge left to right, so the re-gate supersedes the original for anything it
re-ran. The allowlist it writes is what parts 3 and 4 restrict themselves to.

### Turn the gate into what parts 3 and 4 consume

For SWE-bench the gate is 1:1 with tasks, and its result feeds two different shapes:

```bash
# the lists (part 3 filters with these)
cd ../part1-arm64-images
uv run swebench/make_lists.py --from-gate "$HARBOR_JOBS"/gate-<timestamp> --runnable

# a directory of only the gate-clean tasks (part 4 needs this shape)
swebench/tasks.sh swebench/data/swebv-arm64-gated.txt "$HARBOR_DATASETS/swebv-arm64-gated"
```

Both, because the two parts filter differently and only one of them *can* filter. Part 3
takes `TASK_LIST` and excludes by instance id, so it can point at the full task directory.
**Part 4 cannot**: SkyRL takes a directory and treats everything in it as the validation
set, so a task that never passes the gate contributes an exception or a permanent reward
of 0 to every epoch — indistinguishable from a model that cannot solve it. Hence the
second directory; `run_train.sh` defaults to it and refuses to start without it.

**\[measured\]** the first full pass over SWE-bench Verified's arm64-runnable set, 438
tasks at concurrency 8: **397 clean (90.6%)**, 13 with a broken reward path, 27 refused a
runtime for exceeding the 2048 MB image ceiling, 1 transient. django 229/229, sympy 74/74,
pytest 19/19, astropy 16/16; sphinx 36/44, pylint 7/10, requests 5/8; matplotlib 4/31,
which is the size ceiling rather than anything about the tasks.

The 11 that still fail are two groups, and only the second might be a real reward
problem:

| Group | n | Next probe |
|---|---|---|
| patch target missing from the tree | 3 | `ls /testbed` and `git -C /testbed log --oneline -2` in a session |
| patch applied cleanly, tests still failed | 8 | capture `/logs/test_output.log`; the trial does not download it |

## Naming, and why runtime names look like that

With `share_by_content=true` both the image tag and the runtime name key purely on the
environment content hash:

```
off (default):  swesmith-addict-09vlzwgc-a1b2c3d4e5f6-arm64
                swesmith-addict-0fceycuu-a1b2c3d4e5f6-arm64   -> 2 images
on:             a1b2c3d4e5f6-arm64                            -> 1 image
                runtime hb_a3c38cb06d80
```

`hb_<12 hex>` is 15 characters, which fits the service's
`[a-zA-Z][a-zA-Z0-9_]{0,47}` constraint. The flag defaults to **off**, so
one-image-per-task datasets (terminal-bench) are unaffected.

## Clean up

```bash
uv run scripts/cleanup_runtimes.py            # list what is deployed
uv run scripts/cleanup_runtimes.py --delete   # remove them
```

Runtimes cost nothing idle but they consume quota against the 1,000/account default,
and they accumulate silently across runs. Check before a large deploy, not after it
fails.

## What parts 3 and 4 expect

- runtimes deployed for the task set they will use (`delete_runtime: false`)
- `$KIT_STATE_DIR/gate_passing_tasks.txt` if you want to restrict to clean images
- the same `share_by_content: true` in their environment kwargs — **without it every
  rollout deploys its own runtime**, which is the single easiest way to burn the quota
