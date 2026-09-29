# Draft: upstream issue requesting AgentCore Runtime as a sandbox provider

**This is raw material, not something to paste.** Harbor's
[CONTRIBUTING](https://github.com/harbor-framework/harbor/blob/main/CONTRIBUTING.md)
says issues, PR descriptions and comments should be written by a human with AI
assistance, and that *you* own the first and last draft. An agent wrote everything below,
so rewrite it in your own words before posting, and disclose the agent use as CONTRIBUTING
asks. Every number in it is measured and traceable to this repo; the prose is not yours
yet.

Suggested title: **Feature request: Amazon Bedrock AgentCore Runtime as a sandbox
provider**

---

## Summary

I would like to contribute a Harbor `BaseEnvironment` implementation backed by
[Amazon Bedrock AgentCore Runtime](https://aws.amazon.com/bedrock/agentcore/), so that a
Harbor trial's sandbox is an AgentCore session instead of a local container. Before
opening PRs I want to confirm maintainers want this integration and agree with where the
seams are.

I have a working implementation and have used it for real work — a 397-task SWE-bench
Verified evaluation and a GRPO training run of 8,000+ rollouts — so this request comes with
measurements rather than a proposal to investigate. Details and the reproduction
instructions are at the end.

## What I need, concretely

I am doing RL on SWE tasks, where the sandbox is in the inner loop: every rollout is a
fresh environment, an agent loop of shell commands, and a test suite. Two things about
that workload are awkward with local Docker:

1. **Rollout concurrency is bounded by the trainer's own host.** A Harbor task container
   here is 2 vCPU, so a 192-vCPU box tops out near 96 concurrent rollouts — on the same
   machine that is running FSDP and vLLM. Rollouts and training compete for CPU and RAM.
2. **A training set of 45,844 tasks needs sandboxes to be cheap and disposable**, and I
   want per-rollout isolation stronger than a shared kernel while I run other people's
   test suites and model-generated shell commands.

AgentCore Runtime addresses both by moving the sandbox off the training host: each session
is its own microVM, created on demand, and the ceiling becomes a service rate rather than
my instance size.

For scale, from my own runs on this provider: **8,000+ trials in one async RL run at
rollout concurrency 128**, while the GPU host's own CPU stayed free for training. I do not
think that shape is reachable with local containers on the same box.

## What AgentCore Runtime is, and why it fits Harbor's environment interface

AgentCore Runtime is a serverless runtime for agent workloads. You give it a container
image and it gives you sessions; each session is an isolated microVM with its own
filesystem and process space, addressed by a session id you choose.

- [What is Amazon Bedrock AgentCore](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)
- [AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime.html)
- [Service contract a runtime image must satisfy](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-service-contract.html)
- [Sessions](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-sessions.html)
- [Bring your own container](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-getting-started-custom.html)
- [VPC networking](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-vpc.html)
- [Execution-role permissions](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-permissions.html)
- [Quotas](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/quotas.html)
- [Pricing](https://aws.amazon.com/bedrock/agentcore/pricing/)

Why it maps onto `BaseEnvironment` well:

| Harbor needs | AgentCore gives |
|---|---|
| start a sandbox per trial | create a session; measured **0.4 s** to open against a deployed runtime |
| run a command, get output | an invoke API; measured **0.15–0.3 s** round trip |
| read and write files | round-tripped 1 MB and 8 MB payloads in my probes |
| isolation between trials | one microVM per session, not a shared kernel |
| tear down | session ends; nothing to garbage-collect on my host |

What I think is genuinely attractive for Harbor specifically: **the concurrency ceiling
stops being the evaluator's machine.** Session creation is rate-limited by the service
(25/s in my account), not by local CPU, so a 400-task evaluation and a 40-task one put the
same load on the host running Harbor — during my runs the box's load average sat at 0.1.
For anyone doing RL, or evaluating a large suite from a laptop or a CI runner, that is the
whole difference.

### The constraints, stated up front

These are real and I would rather surface them than have a reviewer discover them:

| Constraint | Value | Consequence for Harbor |
|---|---|---|
| Architecture | **arm64 only** | task images must be arm64. This is the single biggest adoption cost: most published SWE-* images are amd64-only |
| Image size | **2048 MB compressed**, not adjustable | some task images cannot be deployed at all. 27 matplotlib images in SWE-bench Verified exceeded it even after slimming |
| Session shape | 2 vCPU / 8 GB RAM / 8.8 GB disk, fixed | `override_cpus` has no effect |
| Max single command | 1 hour | a long pytest suite needs an explicit `exec_timeout_sec` |
| Runtimes per account | 1,000 (default quota) | a dataset with one image per task needs care; see the open design question below |
| Networking | single container, no per-session egress control | cannot air-gap a task the way the Docker provider's egress sidecar can |
| Identity | the microVM has IMDS and can read the execution role's credentials | anything in the sandbox — the agent under test, any task-supplied script — can assume that role. It should be scoped to pulling ECR and writing logs |

The last row is a security boundary I think the docs for such a provider must state
plainly, so I have written it that way in mine.

## How this relates to #2788

[#2788 (Proposal: Strands Agent Integration)](https://github.com/harbor-framework/harbor/issues/2788)
is the only other issue that mentions AgentCore, and it is on the other axis: it asks for
an **agent** integration (a Strands factory driven through `BaseAgent`, running against
whatever environment Harbor provides), whereas this asks for an **environment** provider
(any Harbor agent, running inside an AgentCore session). They compose rather than compete.

Worth quoting, because it is the clearest statement of demand I can point to and it is not
mine — @jjbuck writes in #2788:

> An evaluator should be able to select a compatible task set, choose a harness, and
> eventually place the task in an AgentCore Runtime session without rebuilding the task
> image around that harness or changing the user factory. An AgentCore-backed Harbor
> BaseEnvironment is one possible implementation of that task placement. Again, that
> provider is way beyond the scope of this issue or the proposed PR, but I do want to
> surface the broader vision here to help ground the design choices.

That provider is exactly what this issue is about, and what I have built. @jjbuck — if
this is still something you want, a comment here would help: CONTRIBUTING asks that an
integration be one a user has asked for, and you named it before I did.

Related but distinct AWS-environment demand:
[#1049 (Feature request: Add AWS ECS/Fargate environment provider)](https://github.com/harbor-framework/harbor/issues/1049)
and the open PR [#2341 (feat(environments): add Amazon EKS environment)](https://github.com/harbor-framework/harbor/pull/2341).
Neither gives per-session microVM isolation without a cluster to operate, which is the
property I need.

## What already exists

An implementation, five commits on a fork of `main`, none submitted yet — I want this
conversation first:

| Commit | Scope |
|---|---|
| Add an Amazon Bedrock AgentCore Runtime environment | `src/harbor/environments/agentcore/`, docs, `pyproject.toml` |
| Share images and runtimes by environment content hash | the same directory, plus unit tests |
| Stop the SWE-smith adapter from truncating golden patches | `adapters/swesmith`, plus tests — an independent bug, see below |
| Let the SWEBench adapter target arm64 task images | `adapters/swebench` |
| Pin OpenBLAS to baseline ARMv8 in generated arm64 tasks | `adapters/swebench` |

Nothing touches a shared interface or core logic, so I do not think an RFC is required —
tell me if you disagree.

Measured with it, end to end:

- **SWE-bench Verified arm64, 397 gate-clean instances**: Claude Sonnet 5 with
  `terminus-2` scores 186/397 = 46.9%, 0 errored.
- **Two runs of that same set disagree on 55 of 397 tasks at `temperature: 0`** — about
  ±1.8 points of single-run noise, which is worth knowing before anyone quotes a delta.
- **GRPO on SWE-smith, 8 GPUs, rollouts in AgentCore sessions**: 8,160 completed trials,
  34.2% reward-1.0 rate, rollout concurrency 128. <!-- run still going; refresh with
  part4-train/scripts/check_run.sh before posting -->
- **Warm start 3 s** against ~37 s when the runtime has to be created first, which is why
  the provider keeps runtimes deployed by default.

Reproducible from an immutable tag (the branch will move as review proceeds, the tag will
not):

```
harbor[agentcore] @ git+https://github.com/mightma/harbor@acr-kit-v1
```

Everything around it — building arm64 task images, deploying runtimes, the evaluation and
the RL loop — is at <https://github.com/mightma/harbor-on-agentcore>.

## One design question I would like an opinion on before the PR

Harbor keys a runtime on `environment_content_hash(environment_dir, docker_image)`. That
is right for one-image-per-task datasets and wrong for datasets where many tasks share an
installed environment: the SWE-smith adapter writes a per-task `git checkout` into each
task's Dockerfile, so 45,844 tasks hash 45,844 different ways and would each deploy their
own runtime — against a 1,000-runtime quota.

My provider takes a `share_by_content` flag, default off, that keys the image tag and the
runtime name purely on content, collapsing those 45,844 tasks onto **122 runtimes**. It
is currently local to the AgentCore provider.

The question: is that the right place for it? It reads to me like a general property of
"environments that are expensive to instantiate" rather than an AgentCore detail, and
other providers may want it. I kept it local because a provider-scoped flag needed no
interface change, but I would rather be told now than after a PR.

## Disclosure

Per CONTRIBUTING: the implementation was written with heavy use of coding agents, and I
used an agent to assemble the measurements and the first pass of this text. The design
decisions, the numbers, and this issue as posted are mine, and I am able to discuss any
part of the implementation.
