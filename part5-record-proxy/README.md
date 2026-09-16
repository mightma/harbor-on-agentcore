# Part 5 · Make an installed harness trainable, with a recording proxy

Parts 3 and 4 both use `terminus-2` for one reason: it is the only SWE-capable
harness whose model calls flow through Harbor's own LLM layer, so it is the only one
that produces the per-turn token ids and logprobs step-wise RL needs. Every
SWE-specialised harness — `mini-swe-agent`, `swe-agent`, `claude-code` — is an
installed CLI that calls the model itself, and nothing records those calls.

This part closes that gap for `mini-swe-agent` without changing Harbor, the agent,
or SkyRL: it puts a recording proxy at the endpoint Harbor **already** injects into
the sandbox.

```
 host                                                   AgentCore sandbox (VPC mode)
 ┌────────────────────────────────┐                      ┌───────────────────────────┐
 │ vLLM :8000                     │                      │ mini-swe-agent (installed)│
 │   ▲                            │                      │   OPENAI_BASE_URL ────────┼──┐
 │   │ + logprobs                 │                      └───────────────────────────┘  │
 │   │ + return_token_ids         │                                                     │
 │ record_proxy :8010 ◄───────────┼─────────────────────── private VPC address ─────────┘
 │   │                            │
 │   └─► rollouts.jsonl ──► attach_rollouts.py ──► trial/result.json
 └────────────────────────────────┘                      agent_result.rollout_details
```

## Why this is smaller than it looks

Three of the four pieces already existed:

| Piece | Status |
|---|---|
| Pointing the harness at an endpoint | **exists** — `mini_swe_agent.py` declares `base_url_envs=("OPENAI_BASE_URL", "OPENAI_API_BASE")` and Harbor injects them |
| Sandbox → host transport | **exists** — the AgentCore provider takes `network_mode: VPC` with `subnets` / `security_groups` (`environments/agentcore/environment.py`), no code needed |
| Correlating turns to a trial | **exists** — Harbor sends its `session_id` as `X-Session-ID` when one is set, which is what SkyRL supplies per rollout |
| Recording, and getting it onto the trial result | **this part** — `scripts/record_proxy.py`, `scripts/attach_rollouts.py` |

## What the proxy does

It adds the two parameters Harbor's own LLM layer adds, and nothing else:

```python
completion_kwargs["logprobs"] = True          # llms/lite_llm.py
extra_body["return_token_ids"] = True
```

`extra_body` is a client-side LiteLLM concept — on the wire both are plain top-level
JSON fields, which is why a proxy can add them at all. The response is relayed to
the harness **byte for byte**, so the CLI sees an ordinary OpenAI response with two
extra fields it ignores.

**\[measured\]** against vLLM 0.28 + Qwen3-1.7B on one L40S, which is what
`scripts/e2e_vllm.sh` checks:

| Field | Where vLLM puts it | Harbor reads it from |
|---|---|---|
| `prompt_token_ids` (24 → 41 as the conversation grew) | response root | `response.prompt_token_ids` |
| `token_ids` (6 per completion) | `choices[0]` | `choice.provider_specific_fields["token_ids"]` |
| `logprobs` (6, one per completion token) | `choices[0].logprobs.content[*].logprob` | same |

The same request sent straight to vLLM, without those parameters, comes back with
**no token ids at all** — the test asserts that too, because it is what makes the
proxy load-bearing rather than decorative.

## Stitching turns into trajectories

A proxy sees interleaved requests from every concurrent trial, so it has to group
them. Two mechanisms, in order:

1. **`X-Session-ID`.** Authoritative, and what SkyRL gives you per rollout.
2. **Prefix chaining.** Turn *N*'s messages start with turn *N*−1's. A request that
   extends a known conversation continues it; one that extends nothing starts a new
   rollout. Requires no client cooperation at all.

Prefix chaining is also the honest detector for the cases a proxy **cannot** resolve.
A request that extends a conversation somewhere other than its tip is a *fork* — a
subagent, or a client that rewrote its own history — and it is recorded separately
with `forked: true` rather than concatenated into a trajectory it does not belong to.
`attach_rollouts.py` refuses to attach forked rollouts and says so.
`RolloutDetail`'s own docstring concedes this class ("agents with subagents,
summarization, or other non-linear chat histories"); the proxy's contribution is to
*detect* it instead of silently corrupting the data.

Two related refusals, both deliberate:

- **`stream: true` is rejected** (unless `--allow-unrecorded-stream`). A streamed
  turn cannot be recorded by this path, and a silently unrecorded turn inside a
  trajectory trains on a hole.
- **A rollout with a turn missing token ids** has that turn dropped, and the count
  is printed. If the upstream ignored `return_token_ids`, you learn it here rather
  than from a flat reward curve.

## Run it

```bash
cd part5-record-proxy
python3 scripts/selftest.py                       # stdlib only, no GPU, ~2 s
VLLM=/path/to/vllm scripts/e2e_vllm.sh            # one GPU, real vLLM, ~2 min
```

Against a real job:

```bash
# 1. serve the policy (part 3's script, or SkyRL's own engine in part 4)
# 2. proxy in front of it, on an address the sandbox can reach
scripts/run_proxy.sh                              # prints the env vars to export

# 3. run the job with an installed harness
export OPENAI_BASE_URL="http://<host private ip>:8010/v1"
export OPENAI_API_BASE="$OPENAI_BASE_URL"
export MSWEA_API_KEY=unused-but-required          # mini-swe-agent refuses None
harbor run -c configs/rollout-mini-swe-agent.yaml -p "$HARBOR_DATASETS/swesmith-arm64" ...

# 4. put the recording where a consumer looks
scripts/attach_rollouts.py "$HARBOR_JOBS/<job>" --record <rollouts.jsonl>
```

`configs/rollout-mini-swe-agent.yaml` carries the VPC-mode transport and documents
the three things that must be true outside the file.

## This reverses the kit's security posture

Say it out loud before using it. Everywhere else in this kit the sandbox has no
credentials and no inbound path, because `terminus-2` keeps the model call on the
host. Here:

| | terminus-2 (parts 3–4) | mini-swe-agent + proxy (here) |
|---|---|---|
| Model call originates | host | **inside the sandbox** |
| Sandbox → host network path | none | **VPC, to the proxy port** |
| What the code under test can reach | nothing | the proxy, and whatever else that SG allows |
| Credentials in the sandbox | none | the execution role (as always for installed harnesses), plus `MSWEA_API_KEY` |

VPC mode is still strictly better than the public-IP route: the address is private
and nothing is exposed to the internet. Scope the security group to the one port.

## What is verified, and what is not

| Claim | Status |
|---|---|
| vLLM returns `prompt_token_ids` / `token_ids` / `logprobs` for these parameters | **\[measured\]** vLLM 0.28, Qwen3-1.7B, one L40S |
| The proxy adds them, relays the response unchanged, chains turns, flags forks, refuses streams | **\[measured\]** `scripts/selftest.py`, 21 checks |
| `attach_rollouts.py` writes a well-formed `RolloutDetail` into `result.json` and reports what it cannot match | **\[measured\]**, against synthetic trial dirs and the real recording |
| An installed harness **inside an AgentCore sandbox** reaching the proxy over VPC mode | **untested.** Needs a VPC-mode runtime and a subnet/SG pair; the host this was written on could not build or deploy images (corrupt docker store, no qemu handler) |
| SkyRL consuming proxy-sourced `rollout_details` for a training step | **untested.** The field is populated in the shape SkyRL reads, but no training step has been run on it |

The second-to-last row is the one to close first, and it needs no new code — a
subnet id, a security group, and one trial.

## The upstream shape of this

Attaching after the fact is right for a prototype and wrong as an end state: the
data exists during the trial, so Harbor should be able to record it then. The
minimal upstream change is for an installed agent to accept a rollout sink (the
proxy's address plus the session id it already sends) and populate
`agent_result.rollout_details` itself — which is the RFC
`DESIGN-generalization.md` change 8 describes. Harbor's CONTRIBUTING requires a
human to write the first draft, so that is where it stops here.
