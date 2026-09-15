# Harbor on AgentCore Runtime

Amazon Bedrock AgentCore Runtime (ACR) as the sandbox provider for
[Harbor](https://github.com/harbor-framework/harbor): build the task images, deploy
the runtimes, evaluate a model, and run RL against it. Four parts, each a
self-contained directory with its own `uv` environment.

| Part | What it produces |
|---|---|
| [1 · arm64 task images](part1-arm64-images/) | arm64 images + Harbor task dirs for SWE-smith and SWE-bench Verified, pushed to ECR |
| [2 · AgentCore runtimes](part2-agentcore-runtime/) | one deployed runtime per task image, and a session you can drive by hand |
| [3 · Evaluate](part3-evaluate/) | SWE-bench Verified scores for Claude Sonnet 5 (Bedrock) and a vLLM-served Qwen3.5 |
| [4 · Train](part4-train/) | GRPO on SWE-smith with SkyRL, held-out SWE-bench Verified as the eval set |

The parts chain, and part 2 is the join: parts 3 and 4 both consume the runtimes it
deploys. Part 1 is the only expensive one (hours of qemu); parts 2–4 are minutes to
hours depending on how much you evaluate.

## Start here

```bash
cp config.env.example config.env && $EDITOR config.env
```

That file holds every account id, role ARN and absolute path the kit needs. Nothing
else hardcodes them.

## The one constraint that shapes everything

**ACR only accepts arm64 images.** Almost every published SWE-* task image is
amd64-only, and that single fact determines the structure of all four parts.

Measured by probing registry manifests one by one:

| Dataset | Namespace | arm64 images |
|---|---|---|
| SWE-bench Verified (500) | `swebench/sweb.eval.arm64.*` | **281 / 500** |
| SWE-bench test, non-Verified | — | none |
| SWE-bench train split (19,008) | — | none |
| SWE-Gym-Lite (230) | `xingyaoww/*` | none |
| SWE-smith (~59k over 222 repos) | `jyangballin/swesmith.x86_64.*` | none |

Every ready-made arm64 SWE image in existence is those 281. So:

- **SWE-bench Verified** works from published images, but only 281 of 500, and part 1
  splits them **by repository** into 208 train / 73 eval. The eval number this kit
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

Everything in the per-part READMEs marked **\[measured\]** was actually run on the
host this kit came from: 8× H100 80GB, 192 vCPU, 2 TB RAM, account in `us-west-2`.
Anything else is a projection. In particular:

| Claim | Status |
|---|---|
| SWE-bench Verified arm64 eval, Qwen3.5-**4B**, 7/70 = 10.0% | **\[measured\]** |
| SWE-smith 44,489 tasks → 119 runtimes | **\[measured\]** |
| SWE-smith oracle gate: 108 / 119 repos clean | **\[measured\]** |
| Warm session start 3 s median; hot session open 0.4 s | **\[measured\]** |
| GRPO on 8× H100 with ACR rollouts, non-zero gradient path | **\[measured\]**, 4B |
| **Qwen3.5-9B anywhere in this kit** | **untested.** `config.env` defaults to it because that is what was asked for, but no run in this repo used a 9B policy. 4B is the largest measured. Expect to retune `MICRO_TRAIN`/`GPU_MEM_UTIL` in part 4. |
| Claude Sonnet 5 on this eval set | **untested.** The Bedrock *path* is measured (with Opus 5 on terminal-bench), the Sonnet-5-on-SWE-bench number is not. |

This kit is a working pipeline with two known-good end-to-end runs behind it. It is
not a benchmark report.

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

The per-part READMEs are written to be read in order and each ends with what the next
part expects. If you only want to understand the substrate, read
[part 2](part2-agentcore-runtime/) — it is the shortest and it has the hands-on
session probe.
