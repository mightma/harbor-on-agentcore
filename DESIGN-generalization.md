# Design: make this kit a method, not an example

**Status: proposal, for review. Nothing here is implemented.**

Today the kit reads as *"here is how we ran SWE-smith and SWE-bench Verified with
terminus-2 against Qwen3.5 and Sonnet 5."* It should read as *"here is the method; here
is how you substitute your dataset, your harness, your model."*

A customer substitutes along exactly three axes. Each has a general part worth teaching
and a specific part they have to supply, and the kit currently conflates the two.

| Axis | The general question | Where the kit hardcodes our answer |
|---|---|---|
| Dataset | How many *distinct environments* does it have? | `share_tasks_by_repo()` is SWE-smith-shaped |
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

**Change 1 — generalise the sharing helper.** Replace `share_tasks_by_repo(output_dir,
prepared)` with

```python
share_tasks(
    task_dir,
    group_of:  Callable[[TaskDir], str],        # task -> group key
    image_of:  Callable[[str], str],            # group key -> image reference
    setup_cmd: Callable[[TaskDir], str],        # task -> healthcheck command
    timeout_sec: float = 120.0,
)
```

Ship two adapters over it — `swesmith.py` (group = repo, setup = `git checkout`) and
`swebench.py` (group = `env_image_key`, setup = clone/reset/scrub/install) — and document
the signature as the extension point. Effort: small; the body already exists.

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

**Change 2 — `part1/README.md` gets a decision tree**, replacing the current
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

**Change 3 — a `--group-by` mode for SWE-bench task generation**, so the 40-runtime
layout is a flag rather than a rewrite, and a documented "hybrid" recommendation: share
the small env groups (median size 4), keep per-instance images for the large ones (75,
44, 43) where the shared image cannot hold every tree anyway.

### Completing SWE-bench Verified to 500

**\[measured\]** 281 of 500 have a published `swebench/sweb.eval.arm64.*` image; the other
219 have none. Unlike SWE-smith this needs no reverse engineering — `sweb.base.*` and
`sweb.env.*` are built locally by the `swebench` harness by design, which is why they were
never published. So "build it yourself" means driving its own build chain under qemu.

**Change 4 — `part1/scripts/build_swebench_images.py`**, mirroring
`build_swesmith_images.py`: drive `swebench`'s builder with `--arch arm64`, same
concurrency/skip/report shape. Then `data/swebv-arm64-instances.txt` becomes an *output*
of coverage probing rather than a fixed 281-line fact.

---

## Axis 2 · Harness

### The matrix a customer needs, and does not have today

| | Model call originates | Eval | Train (needs token ids + logprobs) |
|---|---|---|---|
| `terminus-2`, `computer-1` | host, via Harbor's LLM layer | yes | **yes** |
| `claude-code`, `mini-swe-agent`, `swe-agent`, … (`InstalledAgentOptions`) | inside the sandbox, own client | yes | **no, today** |

**Change 5 — publish this table in the top-level README**, with the rule stated plainly:
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

51 s × 70 trials ≈ 1 hour of pure `npm install` per eval pass. For a customer who
evaluates on a *fixed* harness, that is pure waste.

**It works, because Harbor's install is idempotent.**
`claude_code.py:410` `_installed_claude_satisfies_version()` short-circuits with
*"Claude Code is already available at the requested version"*. Bake the CLI into the
prepared image and Harbor skips the install.

**Change 6 — an optional harness layer in the image prepare step.**
`prepare_swesmith_images.py` already bakes git, uv, `/logs` and every task branch; add an
opt-in `--bake-harness <spec>` that appends the harness install to the same Dockerfile.
Ship one recipe per harness and document how to add another.

Two caveats to document with it: the baked version must satisfy the config's `version`
or Harbor reinstalls anyway; and every baked harness costs image size against the
2048 MB compressed ceiling.

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

**Change 7 — state this as a rule in the top-level README**, next to the harness matrix.
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

On the network path: this host has a public IP, so it is *possible*, but it means exposing
the vLLM endpoint. Prefer the proxy at an AWS-internal address over opening a host port,
and document that this reverses the current posture where the sandbox needs no inbound
path at all.

**Change 9 — part 3 and part 4 grow a "bring your own model" section**: the provider
prefix rule (`bedrock/`, `hosted_vllm/`, …), that `CLAUDE_CODE_USE_BEDROCK` is an
*environment variable* rather than a model prefix, and how to check a model id is live
(`aws bedrock list-inference-profiles`).

---

## Change list, ordered by value per unit of risk

| # | Change | Where | Effort | Risk | Unblocks |
|---|---|---|---|---|---|
| 1 | Generalise `share_tasks()` + two adapters | part 1 | S | low | any dataset |
| 2 | Dataset decision tree | part 1 README | S | none | any dataset |
| 5 | Harness capability matrix | top README | S | none | any harness |
| 7 | Where-the-call-originates rule | top README | S | none | any harness |
| 6 | `--bake-harness` in image prepare | part 1 | M | low | fixed-harness customers |
| 3 | `--group-by env` for SWE-bench + hybrid guidance | part 1 | M | medium | 500 → 40 runtimes |
| 9 | "Bring your own model" sections | parts 3, 4 | S | none | any model |
| 4 | Build the missing 219 arm64 images | part 1 | L | medium | full SWE-bench Verified |
| 8 | Recording proxy, `mini-swe-agent` first | new part | L | high | harness-independent training |

Changes 1, 2, 5, 7, 9 are documentation and refactoring with no behavioural risk and they
carry most of the generality benefit — worth doing first and together. 6 and 3 are real
features with measurable payoff. 4 is mechanical but long (qemu). 8 is a research-shaped
task and, if it works, an upstream RFC rather than a kit feature.

## Non-goals

- **Making every harness trainable.** Subagent-spawning and self-compacting harnesses need
  a trajectory-segmentation decision that no proxy can make for you.
- **Publishing images.** The manifests describe what to build in *your* registry; there is
  no public arm64 dataset to pull.
- **Hiding the ceilings.** 2048 MB compressed, 8.8 GB disk, 1,000 runtimes, arm64 only —
  a customer who does not internalise these will design something that cannot run.
