# Design: make this kit a method, not an example

**Status: reviewed, decided, and implemented. All nine changes are marked \[shipped\]
below. Change 4's build path has now been exercised for real — one instance built from
scratch and gated 1.000 on AgentCore; change 6 ships as a capability that has
deliberately not been run. See "What this host could and could not run".**

Today the kit reads as *"here is how we ran SWE-smith and SWE-bench Verified with
terminus-2 against Qwen3.5 and Sonnet 5."* It should read as *"here is the method; here
is how you substitute your dataset, your harness, your model."*

A customer substitutes along exactly three axes. Each has a general part worth teaching
and a specific part they have to supply, and the kit currently conflates the two.

| Axis | The general question | Where the kit hardcodes our answer |
|---|---|---|
| Dataset | How many *distinct environments* does it have? | ~~`share_tasks_by_repo()` is SWE-smith-shaped~~ → `share_tasks()` |
| Harness | Does it call the model itself, or through Harbor? | terminus-2 assumed throughout part 4 |
| Model | Does the call originate on the host or in the sandbox? | two configs, no stated rule |

---

## Axis 1 · Dataset

### The one question that determines cost

**How many distinct environments does the dataset have?** That number is the runtime
count, and on AgentCore it is the binding constraint (1,000/account default). Everything
else in part 1 follows from it.

The method is: **find the coarsest grouping over tasks whose *installed environment* is
byte-identical, then make the per-task difference a cheap in-session step.**

Applied to the two datasets in the kit — both **\[measured\]**:

| Dataset | Grouping | Groups | Per-task delta | Cost of the delta |
|---|---|---|---|---|
| SWE-smith | repository | **119** for 44,489 tasks | `git checkout <branch>` | local, ~0 s |
| SWE-bench Verified | `env_image_key` | **40** for 500 instances | clone + reset + history scrub + `pip install -e .[test]` | **minutes** |

SWE-smith collapses so well because every task of a repository is a branch off the *same*
commit, so one installed conda environment serves all of them. SWE-bench instances of the
same repo span years and different dependency versions; `env_image_key` is precisely the
"dependencies identical" boundary, and *within* a group the commit still changes, so the
editable install has to be redone.

For the record, from `make_test_spec` over all 500: **1** `base_image_key`, **40**
`env_image_key`, 12 repos, 500 instances. Group sizes: min 1, max 75, median 4.

### The mechanism is already general; only the policy is not

What makes sharing work is three edits to a generated task, and none of them is
SWE-smith-specific:

1. delete `environment/Dockerfile`, so the directory is empty and
   `environment_content_hash` falls back to `sha256(docker_image)`;
2. set `[environment].docker_image` to the shared image;
3. put the per-task setup in `[environment.healthcheck]`, which `task.toml` holds and
   which is therefore **not** part of the content hash.

Then `--ek share_by_content=true` drops the task name from the image tag and runtime
name so identical hashes converge.

`run_healthcheck` returns as soon as the command exits 0 (`environments/base.py:1400`),
so the setup runs **exactly once**, and a failure aborts the trial rather than handing
the agent a wrongly-prepared tree. (Note: `HealthcheckConfig`'s docstring claims "all
retries must pass" — that is wrong; worth an upstream doc fix.)

What *is* dataset-specific is only: the grouping function, the image name per group, and
the setup command per task.

**Change 1 \[shipped\] — generalise the sharing helper.** Replaced
`share_tasks_by_repo(output_dir, prepared)` with

```python
share_tasks(
    task_dir,
    group_of:  Callable[[TaskDir], str],        # task -> group key
    image_of:  Callable[[str], str],            # group key -> image reference
    setup_cmd: Callable[[TaskDir], str],        # task -> healthcheck command
    timeout_sec: float = 120.0,
)
```

and document the signature as the extension point. Effort: small; the body already
exists.

**As shipped**, the helper lives in `part1-arm64-images/shared/task_sharing.py` and
carries only **one** adapter — `share_swesmith_tasks()` in `swesmith_tasks.py`, group =
repo, setup = `git checkout`. The `swebench.py` adapter this change originally proposed
is *not* shipped, because Change 3's review decided against grouping SWE-bench Verified
at all; its grouping is documented in `part1/README.md` instead. The shipped helper also
gained one thing the original did not have: it reports tasks whose `environment/` is
still non-empty after the Dockerfile is removed, since those silently do not collapse.

### Two traps a customer will hit, and must be told about

**Baking full git history leaks the future.** SWE-bench's instance script deliberately
deletes tags newer than the base commit, expires the reflog and runs
`git gc --prune=now --aggressive`. If you bake one clone with all history and only check
out per task, **the agent can read future commits and tags** — a benchmark-validity
failure, not a performance one. SWE-smith is immune because its bug branches are
synthetic. Any customer grouping a *real-history* dataset has to keep the scrub in the
per-task step, and pay for it.

**Three hard ceilings bound every grouping choice**, and they pull in opposite
directions:

| Ceiling | Value | Pushes you toward |
|---|---|---|
| ACR image size | 2048 MB **compressed** | fewer things baked in |
| microVM disk | 8.8 GB | fewer things baked in |
| Runtimes per account | 1,000 | coarser grouping |

**Change 2 [shipped] — `part1/README.md` gets a decision tree**, replacing the current
two-dataset narrative:

```
Does your dataset publish arm64 images?
├── yes, for all instances    -> generate tasks, prepull, done  (~10 min)
├── yes, for some             -> split by coverage; state the subset in every result
└── no                        -> build. Then:
    What is the coarsest grouping with an identical installed environment?
    ├── one commit per group (synthetic bugs)  -> per-task = git checkout       (~0 s)
    ├── same deps, different commits           -> per-task = checkout + install (minutes)
    └── nothing shared                         -> one image per task; check the 1,000 quota
```

### Decision: SWE-bench Verified stays per-task. Document the alternative, do not ship it.

**\[decided\]** Build one image per instance. The 40-runtime layout trades a one-time
build cost for a recurring per-session cost, and that is the wrong trade for a benchmark
you run repeatedly:

| | per-task images (**chosen**) | group by `env_image_key` |
|---|---|---|
| Images to build | 500 | 40 |
| Runtimes | 500 | **40** |
| Build cost | one-off, parallel, cacheable | one-off, 12× smaller |
| **Session start** | **~3 s** | clone + reset + scrub + `pip install -e .[test]` — **minutes, every trial, forever** |
| Large env groups (75/44/43 instances) | fine | shared image cannot hold every tree anyway → hybrid needed |
| Benchmark validity | scrub is baked in, done once | scrub must be redone per session or the agent sees future commits |

An eval pass is 70–500 trials and gets repeated across models and checkpoints; RL
multiplies it by epochs. Paying minutes per trial to save a one-time build is a bad
exchange, and it also erodes the property this whole substrate is chosen for — the
measured 3 s warm start.

**Change 3 (revised) [shipped] — do not implement grouping for SWE-bench.** Instead, document the
comparison above in `part1/README.md` as the worked example of the sharing-granularity
method, and state the rule it illustrates:

> Share when the per-task delta is *local and constant* (SWE-smith: `git checkout`, ~0 s).
> Do not share when it is *work* (SWE-bench: an editable install). The grouping that
> minimises runtime count is not automatically the grouping you want.

Keep `share_tasks()` from Change 1 general anyway — a customer whose dataset has a cheap
per-task delta needs it, and SWE-smith already does.

### Completing SWE-bench Verified to 500

**\[measured\]** 281 of 500 have a published `swebench/sweb.eval.arm64.*` image; the other
219 have none. Unlike SWE-smith this needs no reverse engineering — `sweb.base.*` and
`sweb.env.*` are built locally by the `swebench` harness by design, which is why they were
never published. So "build it yourself" means driving its own build chain under qemu.

**Change 4 — `part1/swebench/build_images.py`**, mirroring
`build_swesmith_images.py`: drive `swebench`'s builder with `--arch arm64`, same
concurrency/skip/report shape. Then `swebench/data/swebv-arm64-instances.txt` becomes an *output*
of coverage probing rather than a fixed 281-line fact.

---

## Axis 2 · Harness

### The matrix a customer needs, and does not have today

| | Model call originates | Eval | Train (needs token ids + logprobs) |
|---|---|---|---|
| `terminus-2`, `computer-1` | host, via Harbor's LLM layer | yes | **yes** |
| `claude-code`, `mini-swe-agent`, `swe-agent`, … (`InstalledAgentOptions`) | inside the sandbox, own client | yes | **no, today** |

**Change 5 [shipped] — publish this table in the top-level README**, with the rule stated plainly:
*any harness can be evaluated; only a harness whose calls flow through Harbor's LLM layer
can be trained today.* The existing "Open limitation" section explains the why; what is
missing is the general statement and how to test your own harness against it.

### Pre-baking the harness

**\[measured\]** with `claude-code` on this substrate, medians:

| Stage | claude-code | terminus-2 |
|---|---|---|
| environment start | 3.2 s (0.9%) | 3.3 s (3.6%) |
| **agent install** | **51.3 s (13.5%)** | 12.0 s (13.1%) |
| agent execution | 247.6 s (65.4%) | 63.3 s (69.1%) |
| verifier | 76.6 s (20.2%) | 13.0 s (14.1%) |
| total | 379 s | 92 s |

51 s × 70 trials ≈ 1 hour of pure `npm install` per eval pass. **And terminus-2 is not
free either — 12.0 s × 70 ≈ 14 minutes**, which is worth reclaiming even though it is the
demo harness.

> **Correction (raised in review, verified in the 0.23.0 source).** An earlier draft of
> this section called terminus-2's 12.0 s an "agent install" and said baking would make
> its setup "upload-only". That was wrong twice over, and the distinction matters:
>
> - **terminus-2 installs no agent into the sandbox.** It is host-internal: it runs in
>   the harbor process, and the only thing crossing into the sandbox is shell commands.
>   There is no CLI to bake.
> - **The 12.0 s is nevertheless real, and it is a package install** —
>   `Terminus2.setup()` constructs a `TmuxSession` and `TmuxSession.start()` calls
>   `_attempt_tmux_installation()`, which checks for `tmux` and (because
>   `record_terminal_session` defaults to **True**) `asciinema`, and on a miss runs
>   `DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y tmux asciinema`
>   in the sandbox as root, with a source build of tmux as fallback.
>
> So what `--bake-harness terminus-2` reclaims is **tmux + asciinema**, not an agent
> install — and only the *install* part of the 12.0 s, not the tmux session start, the
> pane setup or the script upload that follow it. Treat "up to 12.0 s" as the ceiling and
> measure the remainder rather than assuming it goes to zero. `record_terminal_session:
> false` is the other way to cut the asciinema half, at the cost of the recordings.
>
> **\[measured\]**, the 70-task Sonnet 5 pass: `agent_setup` median **12.0 s**, min 9.0 s,
> max 53.6 s over 70/70 trials — so the floor is ~9 s even on a warm image, which is
> consistent with an apt install rather than a file upload.

**Baking works because Harbor's setup is idempotent — for both classes.** For installed
CLIs it is a version check: `claude_code.py:410`
`_installed_claude_satisfies_version()` short-circuits with *"Claude Code is already
available at the requested version"*. For terminus-2 it is a presence check: with both
tools already on the image, `_install_recording_tools()` logs *"Both tmux and asciinema
are already installed"* and returns without exec'ing a package manager.

**\[decided\]** terminus-2 remains the harness for the demo paths (parts 3 and 4), since
it is the only one that is both evaluable and trainable. Baking is an **option**, not the
default, so the un-baked path stays the reference.

**Change 6 (revised) — `--bake-harness <name>` in the image prepare step, covering
terminus-2 as well as installed harnesses.** `prepare_swesmith_images.py` already bakes
git, uv, `/logs` and every task branch; this appends a harness layer to the same
Dockerfile. Ship a recipe per harness and document the shape so a customer can add
theirs:

| Harness | What the layer installs | What Harbor then skips | Reclaimed per trial |
|---|---|---|---|
| `terminus-2` | `tmux` + `asciinema` (apt) | `TmuxSession._install_recording_tools()` | **up to 12.0 s** (**\[measured\]** `agent_setup` median; the session start remains) |
| `claude-code` | node + `npm install -g @anthropic-ai/claude-code@<version>` | `_installed_claude_satisfies_version()` | **51.3 s** (**\[measured\]** median) |
| `mini-swe-agent` | `uv tool install mini-swe-agent` + curl/bash/git/build tools | its `install()`, via `get_version_command()` | not yet measured |

Four caveats to document with it:

- the baked version must satisfy the config's `version` or Harbor reinstalls anyway,
  silently undoing the saving. For `mini-swe-agent` the check is
  `uv tool list | grep mini-swe-agent`, so the bake has to install it *as a uv tool
  under the agent user*, not as a plain `pip install`;
- terminus-2's saving is bounded by what `_install_recording_tools()` would have done, so
  it is smaller than the full 12.0 s and shrinks to nothing if the base image already
  carries tmux;
- each baked harness costs image size against the 2048 MB compressed ceiling, so baking
  several into one image is not free;
- a baked image is harness-specific, which cuts against sharing one image set across
  experiments — so it gets its own tag suffix rather than replacing the base one.

**\[shipped, not run\]** `prepare_swesmith_images.py --bake-harness <name>` exists with
the three recipes above and writes `<key>-prepared-<harness>-arm64`. No image has been
built with it: this host cannot build (see the note at the end of this document), so the
recipes are validated by rendering the Dockerfile, not by running it.

---

## Axis 3 · Model

### The rule, which the kit currently only demonstrates

Where the model call originates decides the security posture, and it is a property of the
*harness*, not a setting:

| | terminus-2 (internal) | installed harness |
|---|---|---|
| Call originates | host process | inside the sandbox |
| Sandbox needs credentials | **no** | yes — reads execution role off IMDS |
| Execution role needs `bedrock:InvokeModel` | no | **yes** |
| Sandbox needs a path to your vLLM | no | **yes** |
| What is exposed to the code under test | nothing | the execution role |

**Change 7 [shipped] — state this as a rule in the top-level README**, next to the harness matrix.
It is the single most consequential thing a customer gets wrong, and both existing eval
configs merely *demonstrate* it in comments.

### Serving everything from the host

The goal — all model calls from host-side vLLM, sandbox supplies only reward, training no
longer harness-dependent — is the right architecture. Capturing at the API boundary is
*more* trustworthy than capturing inside the agent, because what you record is what the
engine emitted.

**Change 8 — a recording proxy.** Sits at the address Harbor already injects into
installed harnesses, forwards to vLLM, adds the two parameters Harbor's own LLM layer
adds (`logprobs=True`, `extra_body["return_token_ids"]=True` — `llms/lite_llm.py:313`),
and accumulates `RolloutDetail`. No agent changes, no SkyRL changes: SkyRL already
consumes `RolloutDetail`.

The injection points already exist per harness:

```
mini_swe_agent.py : base_url_envs=("OPENAI_BASE_URL", "OPENAI_API_BASE")
claude_code.py    : base_url_envs=("ANTHROPIC_BASE_URL",)
gemini_cli.py     : base_url_envs=("GOOGLE_GEMINI_BASE_URL",)
```

**Target `mini-swe-agent` first, not `claude-code`.** Four obstacles, and mini-swe-agent
has only the first:

| Obstacle | mini-swe-agent | claude-code |
|---|---|---|
| Network path sandbox → host | yes, both | yes, both |
| Protocol translation | **no** — speaks OpenAI, as vLLM does | **yes** — Anthropic only (`MODEL_CONNECTION` has no OpenAI entry) |
| Subagents break the linear trajectory | no, by design | **yes** — the Task tool spawns them |
| Client-side context rewriting | no | **yes** — it compacts its own history |

The last two are semantic, not engineering: a proxy sees parallel requests and cannot
decide which tokens belong to which trajectory's advantage. `RolloutDetail`'s own
docstring concedes the class ("agents with subagents, summarization, or other non-linear
chat histories").

### The network path needs no new code — VPC mode already exists on both sides

This is the part of the original proposal that was wrong, and it makes Change 8 much
smaller. **\[verified\]** AgentCore Runtime takes a network configuration:

```
CreateAgentRuntime.networkConfiguration
  networkMode      enum: ['PUBLIC', 'VPC']
  networkModeConfig  { securityGroups: list, subnets: list, requireServiceS3Endpoint: bool }
```

and **Harbor's provider already exposes it** — `environment.py:171-172` accepts
`network_mode`, `subnets` and `security_groups`, defaulting to `PUBLIC`. So placing the
sandbox in the same VPC as the training host is configuration, not development:

```yaml
environment:
  type: agentcore
  kwargs:
    network_mode: VPC
    subnets: subnet-...          # the host's subnet, or any in its VPC
    security_groups: sg-...      # must allow inbound to the vLLM port from the sandbox
```

The sandbox then reaches the host on its **private** address, and nothing is exposed to the
internet — strictly better than the public-IP route the first draft assumed. Two things to
document:

- **This reverses the current security posture.** Today the sandbox needs no inbound path
  and no credentials; VPC mode plus a listening vLLM means the code under test can reach a
  host service. Scope the security group to the one port, and remember the sandbox also
  still holds the execution role.
- **`requireServiceS3Endpoint`** exists because a VPC-mode sandbox may lose the default
  route to AWS services. If the sandbox needs ECR or S3, the VPC needs the corresponding
  endpoints, or image pulls start failing in a way that looks like a build error.

**Change 8 (revised) — the proxy is the only new component; the transport is config.**
Target `mini-swe-agent`, point its `OPENAI_BASE_URL` at the proxy, run the proxy on the
host beside vLLM, and connect the two with VPC mode. Scope:

| Piece | Status |
|---|---|
| Sandbox → host transport | **exists** (`network_mode: VPC`, no code) — verified as a real provider kwarg, `network_mode` defaulting to `PUBLIC` |
| Pointing the harness at an endpoint | **exists** (`base_url_envs` injection, no code) |
| Recording proxy: forward, add `logprobs`/`return_token_ids`, accumulate `RolloutDetail` | **\[shipped\]** `part5-record-proxy/scripts/record_proxy.py` |
| Attaching `RolloutDetail` to the trial result so SkyRL sees it | **\[shipped, prototype\]** `scripts/attach_rollouts.py` — post-hoc, not in-trial |

**\[shipped\] Change 8 landed as `part5-record-proxy/`.** What the prototype settled:

- **The two parameters are wire-level, not client-level.** `extra_body` is a LiteLLM
  concept; on the HTTP body `logprobs` and `return_token_ids` are plain top-level fields,
  which is *why* a proxy can add them without touching the agent.
- **\[measured\]** vLLM 0.28 (Qwen3-1.7B, one L40S) answers with `prompt_token_ids` at the
  response root, `token_ids` on the choice, and `logprobs.content[*].logprob` — the three
  places `lite_llm.py` reads. The same request without those parameters returns no token
  ids at all, so the proxy is load-bearing rather than decorative.
- **Grouping needs no client cooperation.** `X-Session-ID` (which `mini-swe-agent` sends
  when Harbor sets a `session_id`, i.e. SkyRL's case) is the authoritative key, and
  prefix-chaining over the message list covers everything else.
- **The non-linear case is now detected rather than assumed away.** A request that
  extends a conversation somewhere other than its tip is a fork — subagent, or
  client-side compaction — and is recorded separately with `forked: true` and refused by
  the attach step. That is the honest version of the "a proxy cannot segment this"
  caveat: it still cannot, but it no longer corrupts the data quietly.
- **The seam is the part that stayed a prototype.** Attaching after the job runs needs no
  Harbor change and makes every match auditable, but the data exists *during* the trial,
  so the end state is an installed agent accepting a rollout sink and populating
  `agent_result.rollout_details` itself. That is the RFC, and CONTRIBUTING wants a human
  to draft it.

Untested, and needing only configuration: an installed harness **inside** an AgentCore
session reaching the proxy over VPC mode, and SkyRL training a step from proxy-sourced
details. See part 5's README for the full verified/unverified split.

**Change 9 [shipped] — part 3 and part 4 grow a "bring your own model" section**: the provider
prefix rule (`bedrock/`, `hosted_vllm/`, …), that `CLAUDE_CODE_USE_BEDROCK` is an
*environment variable* rather than a model prefix, and how to check a model id is live
(`aws bedrock list-inference-profiles`).

---

## Change list, ordered by value per unit of risk

| # | Change | Where | Effort | Risk | Status | Unblocks |
|---|---|---|---|---|---|---|
| 1 | Generalise `share_tasks()`, keep the SWE-smith adapter | part 1 | S | low | **shipped** (`shared/task_sharing.py`) | any dataset with a cheap per-task delta |
| 2 | Dataset decision tree | part 1 README | S | none | **shipped** | any dataset |
| 3 | Document per-task vs `env_image_key` comparison; **ship neither grouping for SWE-bench** | part 1 README | S | none | **shipped** | the method, taught |
| 5 | Harness capability matrix | top README | S | none | **shipped** | any harness |
| 7 | Where-the-call-originates rule | top README | S | none | **shipped** | any harness |
| 9 | "Bring your own model" sections | parts 3, 4 | S | none | **shipped** | any model |
| 6 | `--bake-harness`, terminus-2 included | part 1 | M | low | **shipped, not run** (recipes render; no image built) | up to 12 s/trial now, 51 s for installed harnesses |
| 4 | Build the missing 219 arm64 images | part 1 | L | medium | **shipped and run** — 24 built, 16 gate-clean; the other 195 are qemu hours | full SWE-bench Verified (500) |
| 8 | Recording proxy for `mini-swe-agent` (transport is config) | new part | M–L | medium | **shipped** (`part5-record-proxy/`, measured against real vLLM) | harness-independent training |

**Batch A — 1, 2, 3, 5, 7, 9. Done.** Documentation and one refactor, no behavioural
risk, and they carry most of the generality benefit. What landed, beyond the text: the
sharing helper is now `share_tasks()` over three callables with the SWE-smith policy as
its only adapter, and the harness/model rules are stated with a *measured* self-test —
`agent_result.rollout_details` is a non-empty list for a trainable harness, `[]` when
the details were not requested, and `null` when the harness called the model itself.

**Batch B — 6. Shipped as a capability, deliberately not run.** The three recipes exist
behind `--bake-harness` and are reviewable with `--render-only`; no image has been built
from them, so the *saving* is still a projection while the *contracts* they satisfy are
read out of the source. The correction above is part of this batch's outcome: terminus-2's
12.0 s was never an agent install.

**Batch C — 4, then 8. Both shipped and both exercised.** 8 is `part5-record-proxy/`:
21 stub checks plus a real vLLM. 4 is
`part1-arm64-images/swebench/build_images.py`, and it was run for real:

| | |
|---|---|
| Selection | **\[measured\]** exactly the 219 unpublished instances, resolving to **1 base + 34 env images** |
| Built | **24 instance images** (requests, seaborn, pylint ×3, pytest ×9, sphinx ×10) — 23/23 in one ~1 h 50 min batch on 8 vCPU under qemu |
| Per stage | base ~7 min · env ~9 min · instance 65 s (requests) to 740 s (sphinx) |
| Sizes | 873 MB compressed / 3.28 GB on disk — inside both ACR ceilings |
| Durability | all 24 pushed to ECR `swebv-arm64` |
| **Oracle gate** | **15/24 clean** first pass, **16/24** after one pin |

Running it surfaced four things no amount of reading would have:

1. **The build's tag is not the tag the task asks for.** A local build is
   `sweb.eval.arm64.<id>`; a generated task dir asks for
   `swebench/sweb.eval.arm64.<id with __ → _1776_>`. The script now applies that second
   tag itself, otherwise every trial dies in `ImageBuildError` minutes in.
2. **The gate failures are dependency drift, not architecture.** Six sphinx instances
   failed with `No module named 'roman'`: docutils **0.23** resolved against Sphinx
   **3.1.0**, which needs `<0.18`. Pinning `docutils<0.17` turned 0 passing tests into 14
   — the exact 2 F2P + 12 P2P that failed — and a rebuild re-gated **1.000**. Same class
   as the `flit_core` failure in part 1's SWE-smith notes, and it would happen on amd64
   too. Hence `--pin repo=requirement`, opt-in because the right pin depends on the repo
   version.
3. **A re-pushed image never reaches an existing runtime.** AgentCore resolves
   `containerUri` at *runtime creation*, so with `delete_runtime: false` the deployed
   runtime is frozen; `--force-build` rebuilds and re-pushes and changes nothing.
   **\[measured\]** the same pinned image gated 0.000 through the old runtime and 1.000
   through a fresh one. Delete the runtime or change the task's content hash.
4. **The host's role can create runtimes but not delete or read them** — `CreateAgentRuntime`
   and `ListAgentRuntimes` are allowed, `GetAgentRuntime` and `DeleteAgentRuntime` are
   denied. Worth knowing before planning a cleanup.

### What this host could and could not run

The kit was written on an 8× H100 box that has since been reclaimed; Batches B and C were
implemented on its replacement, which is a smaller machine:

| | Original host | This host |
|---|---|---|
| GPUs | 8× H100 80GB | **1× L40S 46GB** |
| vCPU / RAM | 192 / 2 TB | **8 / 61 GB** |
| Docker | working | corrupt containerd store, **repaired** (see below) |
| qemu-aarch64 handler | registered | **registered** |
| AWS | full | full — ECR push and `CreateAgentRuntime` both verified working |

The docker store arrived unusable: its metadata referenced blobs that no longer existed
(`blob not found`), left over from work done on another instance, so even
`docker system prune` failed. Stopping docker and containerd, deleting the content store,
metadata DB and snapshotter state, and restarting fixed it **and freed 30 GB** — which
mattered, because the disk was at 97%. `tonistiigi/binfmt --install arm64` then registered
the emulator.

With that, change 4 was actually run: 24 instances built, pushed to ECR, and gated on
AgentCore. Change 6 is still unrun by choice, not by constraint — it was scoped as "keep
the capability, do not run it".

The one thing this host still cannot do is the *whole* of change 4. Two limits, both
measured rather than guessed: **time** — 195 instances remain, and the expensive repos
(django ×90, matplotlib ×32, scikit-learn ×28, xarray ×22, astropy ×15, sympy ×8) are the
ones whose env images compile C extensions under emulation rather than installing aarch64
wheels; and **disk** — an instance image is ~3.3 GB on disk, so 219 of them need ~700 GB
against the 38 GB free here. `--push --prune-after-push` is the answer to the second: push
each image to ECR and drop the local copy, keeping only the 34 env images resident.

### Decisions taken in review

| Question | Decision |
|---|---|
| Group SWE-bench by `env_image_key`? | **No** — per-task images. Session cost is recurring, build cost is not. Document the comparison. |
| Which harness for the demo? | **terminus-2**, the only one both evaluable and trainable. |
| Bake the harness into images? | **Optional flag**, and it should cover terminus-2 too — its 12.0 s is not free. |
| Sandbox → host transport | **VPC mode**, private addressing. No public exposure, no Harbor change. |
| First harness for the proxy | **`mini-swe-agent`**, not `claude-code`. |

## Non-goals

- **Making every harness trainable.** Subagent-spawning and self-compacting harnesses need
  a trajectory-segmentation decision that no proxy can make for you.
- **Publishing images.** The manifests describe what to build in *your* registry; there is
  no public arm64 dataset to pull.
- **Hiding the ceilings.** 2048 MB compressed, 8.8 GB disk, 1,000 runtimes, arm64 only —
  a customer who does not internalise these will design something that cannot run.
