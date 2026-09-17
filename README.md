# Harbor on AgentCore Runtime

Amazon Bedrock AgentCore Runtime (ACR) as the sandbox provider for
[Harbor](https://github.com/harbor-framework/harbor): build the task images, deploy
the runtimes, evaluate a model, and run RL against it. Five parts, each a
self-contained directory with its own `uv` environment.

| Part | What it produces |
|---|---|
| [1 · arm64 task images](part1-arm64-images/) | arm64 images + Harbor task dirs for SWE-smith and SWE-bench Verified, pushed to ECR — one directory per dataset, over Harbor's own adapters |
| [2 · AgentCore runtimes](part2-agentcore-runtime/) | one deployed runtime per task image, and a session you can drive by hand |
| [3 · Evaluate](part3-evaluate/) | SWE-bench Verified scores for Claude Sonnet 5 (Bedrock) and a vLLM-served Qwen3.5 |
| [4 · Train](part4-train/) | GRPO on SWE-smith with SkyRL, held-out SWE-bench Verified as the eval set |
| [5 · Record proxy](part5-record-proxy/) | token ids and logprobs for an *installed* harness, so `mini-swe-agent` becomes trainable |

The first four chain, and part 2 is the join: parts 3 and 4 both consume the runtimes
it deploys. Part 1 is the only expensive one (hours of qemu); parts 2–4 are minutes to
hours depending on how much you evaluate. Part 5 is optional and sits beside part 4 —
it exists only if you want to train a harness other than `terminus-2`.

## Start here

```bash
cp config.env.example config.env && $EDITOR config.env
```

That file holds every account id, role ARN and absolute path the kit needs. Nothing
else hardcodes them.

## The one constraint that shapes everything

**ACR only accepts arm64 images.** Almost every published SWE-* task image is
amd64-only, and that single fact determines the structure of the whole kit.

Measured by probing registry manifests one by one:

| Dataset | Namespace | arm64 images |
|---|---|---|
| SWE-bench Verified (500) | `swebench/sweb.eval.arm64.*` | **281 / 500** |
| SWE-bench test, non-Verified | — | none |
| SWE-bench train split (19,008) | — | none |
| SWE-Gym-Lite (230) | `xingyaoww/*` | none |
| SWE-smith (~59k over 222 repos) | `jyangballin/swesmith.x86_64.*` | none |

Every ready-made arm64 SWE image in existence is those 281. So:

(Note what this is *not* about: converting a dataset into Harbor tasks is
[`adapters/`](https://github.com/harbor-framework/harbor/tree/main/adapters)' job and
there are 85 of them, two of which this kit installs as packages. Part 1 is about the
images those generated tasks point at, which is where arm64 bites.)

- **SWE-bench Verified** works from published images, but only 281 of 500, and part 1
  splits them **by repository** into 208 train / 73 eval. The other 219 can now be
  built (`swebench/build_images.py`, **\[measured\]** one instance end to end),
  which is qemu hours rather than a blocker. The eval number this kit
  produces is therefore *not* "SWE-bench Verified" — it is a 73-instance held-out-repo
  subset whose repo distribution differs sharply from the full 500 (46% of which is
  django). Say so whenever you quote it.
- **SWE-smith** has no arm64 images at all, so part 1 builds them: 119 repository
  images under qemu, ~2.5 h at concurrency 12. That is the price of a training set
  that is disjoint from the eval benchmark.

Other hard service limits, all of them design constraints rather than bugs:

| Limit | Value | Consequence |
|---|---|---|
| Architecture | arm64 only | the whole table above |
| Image size | 2048 MB **compressed** | prepared SWE-smith images land at ~907 MB |
| Session shape | 2 vCPU / 8 GB / 8.8 GB disk | fixed; `override_cpus` is ignored |
| Max single command | 1 hour | a long pytest suite needs `exec_timeout_sec: 3600` |
| Runtimes per account | 1,000 (default) | see "Runtime budget" below |
| Networking | single container, no egress control | cannot air-gap a task |

## The Harbor build this uses

Parts 3 and 4 install Harbor as a package from a branch, not from PyPI:

```
harbor[agentcore] @ git+https://github.com/mightma/harbor@acr-kit
```

That branch is upstream `main` plus five commits. Two are in review upstream; three
are not yet submitted:

| Commit | What | Upstream status |
|---|---|---|
| Add an Amazon Bedrock AgentCore Runtime environment | the provider itself | in PR |
| Share images and runtimes by environment content hash | `share_by_content` | in PR (same branch) |
| Stop the SWE-smith adapter from truncating golden patches | `.strip()` → `.strip("\n")` | in PR (separate) |
| Let the SWEBench adapter target arm64 task images | `--arch` flag | not submitted |
| Pin OpenBLAS to baseline ARMv8 in generated arm64 tasks | `OPENBLAS_CORETYPE` | not submitted |

Two of them are worth understanding before you trust a number out of this kit,
because both failed *silently* — the mechanism ran, the reward was 0, and nothing
looked broken:

- **`share_by_content`.** Harbor keys a runtime on
  `environment_content_hash(environment_dir, docker_image)`. The upstream SWE-smith
  adapter writes the per-task `git checkout` into the task's own Dockerfile, so all
  44,489 tasks hash differently and each would deploy its own runtime. This flag keys
  purely on content, collapsing them to **119 runtimes**. Without it you hit the
  1,000-runtime quota almost immediately.
- **`OPENBLAS_CORETYPE`.** `import numpy` SIGILLs inside an ACR session on some
  builds, so every reward was 0 while the training loop happily reported steps. A
  dead reward path looks exactly like a hard benchmark.

## Measured vs untested

Everything in the per-part READMEs marked **\[measured\]** was actually run. Most of it
on the host this kit came from — 8× H100 80GB, 192 vCPU, 2 TB RAM, `us-west-2` — which
has since been reclaimed; part 5's numbers come from its replacement, a 1× L40S box that
cannot build images at all. Anything not marked is a projection. In particular:

| Claim | Status |
|---|---|
| SWE-bench Verified arm64 eval, Qwen3.5-**4B**, 7/70 = 10.0% | **\[measured\]** |
| SWE-smith 44,489 tasks → 119 runtimes | **\[measured\]** |
| SWE-smith oracle gate: 108 / 119 repos clean | **\[measured\]** |
| Warm session start 3 s median; hot session open 0.4 s | **\[measured\]** |
| GRPO on 8× H100 with ACR rollouts, non-zero gradient path | **\[measured\]**, 4B |
| Recording proxy: vLLM returns token ids + logprobs, turns stitched, attached to `result.json` | **\[measured\]** on the replacement host (1× L40S, vLLM 0.28) — part 5 |
| Self-built arm64 SWE-bench images: **160 built, 130 gate-clean**; 27 exceed ACR's 2048 MB image ceiling and cannot be deployed at all | **\[measured\]** — part 1 |
| Claude Sonnet 5 on those 130 self-built instances: **58/130 = 44.6%**, $9.29 | **\[measured\]** — part 3 (68% django; not comparable to the 70-task held-out-repo set) |
| Native arm64 is 16× faster per image than qemu (131 s vs 2112 s for a django instance) | **\[measured\]** c7gd.8xlarge vs 8-vCPU x86 |
| Proxy reached from *inside* a sandbox over `network_mode: VPC` | **untested.** Configuration only, but nobody has run it |
| **Qwen3.5-9B anywhere in this kit** | **untested.** `config.env` defaults to it because that is what was asked for, but no run in this repo used a 9B policy. 4B is the largest measured. Expect to retune `MICRO_TRAIN`/`GPU_MEM_UTIL` in part 4. |
| Claude Sonnet 5 on this eval set | **untested.** The Bedrock *path* is measured (with Opus 5 on terminal-bench), the Sonnet-5-on-SWE-bench number is not. |

This kit is a working pipeline with two known-good end-to-end runs behind it. It is
not a benchmark report.

## Substituting the harness: what any harness can and cannot do here

Harbor ships a lot of harnesses — 45 names wired into `AgentFactory` in the 0.23.0
build this kit uses, `harbor run --agent <name>` or `agents[].name` in a config. They
fall into three classes, and the class decides what you can do with one:

| Class | Examples | Model call originates | Eval | Train (needs per-turn token ids + logprobs) |
|---|---|---|---|---|
| internal, through Harbor's LLM layer | `terminus-2`, `computer-1` | host process | yes | **yes** |
| internal, own client | `dspy-rlm` (calls `dspy.LM` itself) | host process | yes | no |
| installed CLI (`harbor/agents/installed/*`, 40 of the 45) | `claude-code`, `mini-swe-agent`, `swe-agent`, `codex`, `opencode`, … | **inside the sandbox** | yes | **no, today** |

The remaining two, `oracle` and `nop`, call no model at all — `oracle` applies the golden
patch, which is what part 2's gate uses to prove a task's reward path works.

**The rule: any harness can be evaluated; only a harness whose model calls flow through
Harbor's own LLM layer can be trained.** Being in-process is not enough — `dspy-rlm`
runs on the host and still records nothing, because it brings its own client.

### Testing your own harness against the rule

Three checks, cheapest first:

1. **Where does the class live?** `harbor/agents/installed/…` means Harbor installs a
   CLI into the sandbox and that CLI calls the model. Nothing on the host sees the
   traffic.
2. **Does its options model accept `collect_rollout_details`?** In 0.23.0 exactly two
   do: `terminus_2` and `computer_1`.
3. **Run one trial and read `agent_result.rollout_details` in its `result.json`.** This
   is the check that cannot be argued with, and both outcomes below are **\[measured\]**
   in this kit, same task, same model id:

   ```
   terminus-2   "model_info": {"name": "us.anthropic.claude-sonnet-5", "provider": "bedrock"}
                "rollout_details": []          # layer is there; details not requested
   claude-code  "model_info": {"name": "us.anthropic.claude-sonnet-5", "provider": null}
                "rollout_details": null        # field never applied; the CLI called out itself
   ```

   `[]` with `collect_rollout_details: true` set would mean a broken engine (vLLM must
   return `logprobs` and `return_token_ids`); `null` means the harness is in the third
   class and no engine flag will change it. `provider: null` is not a bug either — it is
   Harbor recording that it does not own the routing for this call.

## Where the model call originates, and why it is not a setting

This is the single most consequential thing to get right, and it follows from the
harness class above rather than from any option you can set:

| | internal harness (`terminus-2`) | installed harness (`claude-code`, `mini-swe-agent`, …) |
|---|---|---|
| Call originates | host process | inside the sandbox |
| Sandbox needs credentials | **no** | yes — reads the execution role off IMDS |
| Execution role needs `bedrock:InvokeModel` | no | **yes** |
| Sandbox needs a network path to your vLLM | no | **yes** |
| What is exposed to the code under test | nothing | the execution role |

Two consequences that catch people:

- **You cannot keep the sandbox credential-free and evaluate an installed CLI.** The
  only way to move the trust boundary is to change harness. `config.env.example` spells
  out what granting `bedrock:InvokeModel` exposes; part 3 has both paths side by side,
  and part 3's `configs/eval-claude-code.yaml` is the one that needs it.
- **A locally served policy is not reachable from an installed harness by default.** ACR
  sessions default to `PUBLIC` network mode, which gives the sandbox egress to the
  internet but no route to a vLLM bound to this host's loopback or private address.
  Harbor's provider does accept `network_mode: VPC` with `subnets` /
  `security_groups` (`environment.py:171`), which puts the session on your VPC's private
  addressing — but that reverses the posture in the table above, so scope the security
  group to the one port. `DESIGN-generalization.md` works this through.

## Open limitation: training is pinned to the `terminus-2` harness

**Status: still the default, but no longer a dead end. Parts 3 and 4 both use `terminus-2`; [part 5](part5-record-proxy/) is the prototype that lifts it for `mini-swe-agent`.**

The matrix above states the rule; this is the why, and what would lift it. Only a
harness whose model calls flow through Harbor's own LLM layer can produce the per-turn
token ids and logprobs that step-wise RL needs. Today that is `terminus-2` and
`computer-1`, and `computer-1` is a computer-use agent — so for SWE tasks the training
harness is effectively fixed.

The mechanism is small (`llms/lite_llm.py`): when `collect_rollout_details` is on it
asks the engine for two extra things,

```python
completion_kwargs["logprobs"] = True
extra_body["return_token_ids"] = True
```

and assembles the responses into `RolloutDetail` (`prompt_token_ids`,
`completion_token_ids`, `logprobs` per turn). That is an *engine* capability, not an
agent capability — vLLM's OpenAI endpoint returns both.

The SWE-specialised harnesses are all `InstalledAgentOptions` — `mini-swe-agent`,
`swe-agent`, `claude-code` — meaning Harbor installs the CLI into the sandbox and the
CLI calls the model with its own client. Nothing records those calls.

**This is a gap, not a wall.** Harbor already decides where those CLIs send their
traffic: it injects the endpoint as an env var per harness
(`mini_swe_agent.py` declares `base_url_envs=("OPENAI_BASE_URL", "OPENAI_API_BASE")`,
`claude_code.py` declares `ANTHROPIC_BASE_URL`). The missing piece is a **recording
proxy** at that address: forward to vLLM, add the two parameters above, accumulate
`RolloutDetail`. No agent changes, and no SkyRL changes — it already consumes
`RolloutDetail`. Capturing at the API boundary is also more trustworthy than capturing
inside the agent, because what you record is what the engine actually emitted.

Two classes of harness stay hard even with a proxy, and `RolloutDetail`'s own docstring
says as much ("agents with subagents, summarization, or other non-linear chat
histories"):

- **subagents** (`claude-code` spawns Task) — the trajectory is not one line, and which
  tokens belong to which trajectory's advantage is a semantic question a proxy cannot
  answer;
- **client-side context rewriting** — if the harness compacts its own history, turn *N*'s
  prompt is not the concatenation of turns 1..*N*−1. This is also why parts 3 and 4 set
  `enable_summarize: false`; SkyRL's step-wise path rejects summarisation outright.

`mini-swe-agent` is deliberately minimal (single linear conversation, no subagents), so
it is the most likely third-party harness to become trainable through a proxy.

**That proxy now exists: [part 5](part5-record-proxy/).** It adds the two parameters,
records what vLLM returns (**\[measured\]** against vLLM 0.28 — `prompt_token_ids` at the
response root, `token_ids` on the choice, one logprob per completion token), stitches
turns into rollouts without the client's help, and attaches them to
`agent_result.rollout_details`. Two things it does *not* yet prove: that an installed
harness inside an AgentCore session reaches it over `network_mode: VPC` (configuration,
untested), and that SkyRL trains a step from proxy-sourced details. It also does not
solve the two hard classes above — it *detects* them, flags the turn as forked, and
refuses to attach, which is the most a proxy can honestly do.

So the default stands: **train with `terminus-2`**, and reach for part 5 when the harness
itself is what you need to train. Evaluating with a different harness is fine and cheap,
but be explicit that the training and evaluation scaffolds then differ — which is exactly
the comparison this kit otherwise works to keep clean.

## Runtime budget

One runtime per distinct task image, and they are kept deployed
(`delete_runtime: false`) because redeploying costs an ECR push and a wait on every
epoch. Budget against the 1,000/account default:

| Dataset | Runtimes |
|---|---|
| SWE-smith, all 119 repos | 119 |
| SWE-smith, 108 oracle-clean repos | 108 |
| SWE-bench Verified arm64, 73 eval | 73 |
| SWE-bench Verified arm64, 208 train | 208 |

Runtimes cost nothing idle but they do consume quota, and they accumulate: leftovers
from earlier runs are easy to forget. `part2-agentcore-runtime/scripts/cleanup_runtimes.py --delete`
is the broom.

## Cost

Sandbox time is cheap; model tokens are not, and they are dominated by outliers.
**\[measured\]**, terminal-bench with Opus 5 as an installed agent: median ~$0.40 per
task, but one task cost $10.09 because the agent explored for 3,020 seconds. Estimate
from the tail, not the mean. The Qwen3.5 paths cost nothing per token beyond the GPUs
you are already holding.

## Prerequisites

- An AWS account with AgentCore Runtime available in your region, plus an execution
  role (see `config.env.example` for exactly what it must allow, and what granting
  `bedrock:InvokeModel` exposes).
- Docker with the arm64 qemu handler registered — **re-assert this before every build
  session**, it disappears on this host:
  ```bash
  [ -e /proc/sys/fs/binfmt_misc/qemu-aarch64 ] || \
    docker run --privileged --rm tonistiigi/binfmt --install arm64
  ```
- `uv`, and Python 3.12+.
- For parts 3–4: NVIDIA GPUs. Part 4 as written assumes 8; part 3's vLLM path needs 1
  for a 4B.
- Several hundred GB of fast scratch, and a Docker Hub account if you build much —
  anonymous pulls are capped at 100/hour/IP and part 1 makes ~134 of them.

## Reading order

Parts 1–4 are written to be read in order and each ends with what the next part
expects. If you only want to understand the substrate, read
[part 2](part2-agentcore-runtime/) — it is the shortest and it has the hands-on
session probe. [Part 5](part5-record-proxy/) reads on its own and only matters if the
harness is the thing you want to train; it is also the only part with a test you can
run in two seconds with no GPU (`python3 scripts/selftest.py`).
