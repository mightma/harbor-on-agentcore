# Part 3 · Evaluate a model with Harbor + AgentCore Runtime

Score two policies on the same 397 gate-clean SWE-bench Verified arm64 tasks, with every
trial's sandbox an ACR session:

- **Claude Sonnet 5** via Bedrock
- **Qwen3.5** served locally with vLLM

```bash
cd part3-evaluate
uv sync                  # Bedrock path only
uv sync --extra serve    # adds vLLM (multi-GB); or set VLLM=<existing path>
```

## Run them

```bash
scripts/run_eval_bedrock.sh                    # $EVAL_BEDROCK_MODEL
scripts/run_eval_vllm.sh                       # $POLICY_MODEL, serves on GPU 0
```

Both default to `swebench/data/swebv-arm64-gated.txt` — the 397 test-set tasks of 411
runnable whose oracle ceiling is 1.0. The other 14 are excluded on purpose: their own
golden patch does not score 1.0, so including them subtracts a constant from every score
and teaches you nothing.

### Running SWE-smith here, when it is the training set

SWE-smith belongs to part 4, but a policy run over a sample of it is the cheapest check
that the tasks work as tasks — and it exercises something part 2's gate does not, since
that gate covers one task per *image*:

```bash
TASKS="$HARBOR_DATASETS/swesmith-arm64" \
TASK_LIST="$KIT_STATE_DIR/gate_passing_tasks.txt" \
SAMPLE=50 scripts/run_eval_bedrock.sh
```

Do not report the solve rate as a benchmark number: SWE-smith tasks are synthetic bug
injections, the sample is 50 of 39,686, and only the image each one sits on has a
verified oracle ceiling — not the task. If a sampled task scores 0 you cannot tell a
model failure from a task whose golden patch never applied. `gate_sample.sh` in part 2
is the run that separates them, and it is the one to do first.

## The comparison is only fair if the scaffold is identical

`configs/eval-bedrock.yaml` and `configs/eval-vllm.yaml` deliberately agree on
everything except the model: `terminus-2`, 8 turns, 32k window, `enable_summarize:
false`, verifier capped at 600 s. Change one, change both — otherwise you are comparing
scaffolds.

The one asymmetry that is *not* a scaffold difference, and is worth understanding:

| | Bedrock path | vLLM path |
|---|---|---|
| Where the model call originates | this host (harbor process) | this host (harbor process) |
| What crosses into the sandbox | shell commands only | shell commands only |
| Credentials in the sandbox | none | none |
| Execution role needs `bedrock:InvokeModel` | **no** | no |
| GPUs needed | 0 | 1+ |

Both use `terminus-2`, an *internal* agent, so both keep the sandbox credential-free.
Harbor's `bedrock/` prefix resolves to the `amazon-bedrock` provider and LiteLLM calls
it with this host's role.

There is a second, different way to evaluate a Bedrock model — `claude-code` as an
*installed* agent, where Harbor installs the CLI inside the sandbox and the model call
originates there. That path works (it is how the terminal-bench runs in this project
were done) but it needs `CLAUDE_CODE_USE_BEDROCK=1` plus an execution role allowed to
invoke Bedrock, and it changes both the scaffold and the trust boundary. It is the
right choice when the CLI *is* the thing under test, and the wrong choice for a
like-for-like model comparison.

### Why 8 turns and 32k, not 12 and 16k

Because 16k demonstrably does not work. **\[measured\]** at 16k with 12 turns, 15 of 17
attempts died on context length. The budget is roughly

```
max_model_len >= max_turns * (10 KB observation + output cap) + 2k
```

At 32k with 8 turns, **\[measured\]** exactly 1 of 70 trials hit
`ContextLengthExceededError`. Part 4 uses the same 8/32k so the training reward and this
eval number describe the same scaffold rather than two different ones.

### Qwen3.5 reasons by default, and it must be switched off

`configs/eval-vllm.yaml` sets:

```yaml
extra_body:
  chat_template_kwargs:
    enable_thinking: false
```

Without it an untouched "Reply with exactly: ok" spends its entire output budget on a
`Thinking Process:` preamble. Terminus-2 needs a parseable command every turn inside
`max_output_tokens`, so reasoning has to be off at the chat template. Harbor's LiteLLM
wrapper deep-merges `extra_body`, so this coexists with the `return_token_ids` it adds
for rollout details.

## Bring your own model

Two places to change, and neither is in a config file: `EVAL_BEDROCK_MODEL` /
`POLICY_MODEL` in `config.env`, or the first argument to the run scripts. The scripts
substitute the placeholder in the YAML, so the configs stay model-agnostic.

**The provider prefix rule.** For an *internal* harness (`terminus-2`) `model_name` goes
straight to LiteLLM, and the prefix — not a separate setting — selects the provider.
Harbor resolves credentials from the same prefix (`harbor/agents/model_connection.py`,
where `_PROVIDER_ALIASES` maps `bedrock` → its `amazon-bedrock` credential set):

| What you have | `model_name` | Also needs |
|---|---|---|
| Bedrock model or cross-region inference profile | `bedrock/us.anthropic.claude-sonnet-5` | **this host's** credentials to allow `bedrock:InvokeModel` — not the sandbox's |
| vLLM, SGLang, anything OpenAI-compatible | `hosted_vllm/<served-model-name>` | `api_base`, and the served name must match what the server advertises |
| Anthropic API directly | `anthropic/claude-…` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai/…` | `OPENAI_API_KEY` |

`run_eval_vllm.sh` derives the served name from the path (`SERVED_NAME`, default
`basename`) and passes it to both `vllm serve --served-model-name` and the
`hosted_vllm/` prefix. Overriding one and not the other is the standard way to get a
404 from a server that is running perfectly.

**The prefix is not a switch for an installed harness.** `claude-code` pins its provider
(`MODEL_CONNECTION.default_provider = "anthropic"`), so `bedrock/us.anthropic.claude-sonnet-5`
contributes *only the model id* — the routing comes from the environment variable
`CLAUDE_CODE_USE_BEDROCK=1` (or `AWS_BEARER_TOKEN_BEDROCK`), which is what blanks the
inferred Anthropic endpoint and makes Harbor export AWS credentials and region into the
sandbox. `run_eval_claude_code.sh` sets it. Without it the CLI takes an empty
`ANTHROPIC_API_KEY` to `api.anthropic.com` and fails auth while the model name looks
perfectly correct. Two follow-ons:

- Harbor defaults the sandbox's `AWS_REGION` to **us-east-1** when the host has none
  set. A `us.` inference profile is region-scoped, so the wrong region surfaces as a
  model-not-found rather than as a region problem.
- Not every installed harness behaves this way. `mini-swe-agent` declares no default
  provider and is `passthrough`, so its prefix *does* select the provider and Harbor
  injects `OPENAI_BASE_URL` / `OPENAI_API_BASE` into the sandbox for it. That injection
  point is why it, not `claude-code`, is the first target in
  `DESIGN-generalization.md`'s recording-proxy plan.

**Check the id is live before paying for a run.** Model ids are per-region and per
account:

```bash
aws bedrock list-inference-profiles --region "$AWS_REGION" \
  --query 'inferenceProfileSummaries[?status==`ACTIVE`].inferenceProfileId' --output table
aws bedrock list-foundation-models --region "$AWS_REGION" \
  --query 'modelSummaries[].modelId' --output text | tr '\t' '\n' | grep anthropic
```

Then `SMOKE=1 scripts/run_eval_bedrock.sh <id>` — one task, one trial, and an auth or
id mistake shows up in about a minute instead of after 397.

### Which subset flag: `SMOKE` proves a path, `SAMPLE` measures

Both narrow `TASK_LIST`, and choosing wrong produces a result that looks fine and means
nothing. Every list here is sorted, so `SMOKE=n` takes the head of an alphabet:
**\[measured\]** `SMOKE=50` on the SWE-smith allowlist selects 50 tasks from **one
repository** — one image, one runtime, one conda environment. `SAMPLE=50` on the same
list draws across it and hits **35** distinct images. So:

```bash
SMOKE=1   …   # does the path work at all: auth, sandbox, verifier
SAMPLE=50 …   # what does the model score, on a subset worth extrapolating from
SAMPLE=50 SAMPLE_SEED=7 …   # the same draw again
```

`SAMPLE` is also the only way to touch SWE-smith from here. Harbor takes task names as
repeated `-i` flags with no file form, and the 39,686-task allowlist is **2,228 kB** of
argv against this host's `ARG_MAX` of 2,097,152 — it cannot be passed at all. The
scripts check that budget and say so, rather than dying in `execve` with `E2BIG`. For a
whole-set run, point `TASKS` at a directory that already holds only those tasks
(`part4-train/scripts/make_train_set.sh` builds one from the same allowlist).

Selecting zero tasks is an error, never "so run all of them" — only `TASK_LIST=''` with
no `SMOKE`/`SAMPLE` asks for the whole set. The difference is a 50-task run and a
45,844-task one.

**What has to move with the model:**

| If you change | Also change |
|---|---|
| the model's context window | `model_info.max_input_tokens` **and** `MAX_MODEL_LEN` (vLLM path); the 8-turn budget assumes 32k |
| to a non-Qwen policy | drop `extra_body.chat_template_kwargs.enable_thinking` unless that model's chat template takes it — it is not a harmless no-op |
| to a paid API on the vLLM path | the `input_cost_per_token: 0.0` / `output_cost_per_token: 0.0` in `eval-vllm.yaml`, or `cost_usd` in the results is a lie |
| the model, for a comparison | nothing else — that is the point of `configs/eval-*.yaml` holding the scaffold |

## Results

**\[measured\]** Qwen3.5-**4B**, 70 tasks, 24 concurrent, 8 turns, 32k:

| | Value |
|---|---|
| **Solved** | **7 / 70 = 10.0%** |
| Scored | 69/70 (one `ContextLengthExceededError`) |
| Wall clock | **~11 minutes** |
| Tokens | in 2.88M / out 0.13M |
| Environment start, median | **3 s** |
| Agent execution, median | 25 s |
| Verifier, median | 12 s |

That 3 s is the entire point of part 2: the runtimes and images already existed, so a
trial only opens a session. The cold path — build, push, deploy, wait for READY, open —
was 37 s median.

### Claude Sonnet 5 on the whole arm64-runnable set

**\[measured\]** the 397 gate-clean instances — every SWE-bench Verified instance that has
an arm64 image ACR can deploy *and* a verified oracle ceiling, 79.4% of the benchmark.
terminus-2, 8 turns, 32k, temperature 0, 16 concurrent:

| | Value |
|---|---|
| **Solved** | **186 / 397 = 46.9%** |
| Scored | 397/397, **0 errors** |
| Cost | **$28.03** — in 16.50M tokens (13.06M cached), out 1.43M |
| Wall clock | **44.6 min**, achieved concurrency 14.9× of 16 |
| Environment start, median | **3 s** (max 6 s) |
| Agent execution, median | 62 s (max 332 s) |
| Verifier, median | 12 s |

| Repo | Solved | | Repo | Solved |
|---|---|---|---|---|
| django | 116/229 = 50.7% | | pylint | 2/7 = 28.6% |
| sympy | 36/74 = 48.6% | | requests | 1/5 = 20.0% |
| sphinx | 15/36 = 41.7% | | scikit-learn | 2/4 |
| pytest | 8/19 = 42.1% | | matplotlib | 0/4 |
| astropy | 5/16 = 31.2% | | seaborn / flask | 0/2, 1/1 |

**Read this as a scaffold number, not a model number.** 8 turns at 32k with no retries and
temperature 0 is chosen to match part 4's training scaffold, not to maximise a score;
published SWE-bench Verified results for frontier models use far larger budgets. What it is
good for is exactly what this kit needs: a baseline measured on the same scaffold the RL
rollouts use.

### Superseded: Sonnet 5 on 130 self-built instances

**\[measured\]** the other direction: part 1 built arm64 images for 160 instances SWE-bench
never published, 130 of which pass the oracle gate. Nobody had a number for those on this
substrate, because until now they could not be run at all.

| | Value |
|---|---|
| **Solved** | **58 / 130 = 44.6%** |
| Scored | 129/130 (one `RuntimeError`) |
| Cost | **$9.29** — in 5.50M tokens (4.38M cached), out 0.48M |
| Wall clock | ~16 min at 16 concurrent |
| Environment start, median | **3 s** — the runtimes were already deployed by part 2's gate |
| Agent execution, median | 60 s (max 337 s) |
| Verifier, median | 13 s |

| Repo | Solved |
|---|---|
| django | 43 / 88 = 48.9% |
| pytest | 6 / 9 = 66.7% |
| scikit-learn | 2 / 3 |
| sympy | 3 / 7 = 42.9% |
| astropy | 2 / 8 = 25.0% |
| sphinx | 1 / 10 = 10.0% |
| matplotlib / seaborn / requests | 1 / 2, 0 / 1, 0 / 1 |

**Do not compare 44.6% against a "SWE-bench Verified" number, including this kit's own
70-task one.** This set is **68% django** (88 of 130) because that is where the unpublished
instances are, while the 70-task set excludes django entirely — it is the held-out-repo
split. Different instance mix, different difficulty, no shared repositories with the eval
split. What the number does show is that the self-built images behave like real task
images: a spread of per-repo solve rates in a plausible range, not a suspiciously flat 0 or
1 that would mean the substrate was broken.

A different, older comparison, kept because it measures something the one above does not:
**terminal-bench with an *installed* agent** (claude-code), where the agent install is a
real cost and the trials are much longer. ACR against local Docker, **\[measured\]**
median/max:

| Stage | agentcore | docker |
|---|---|---|
| Environment start | 14 s / 600 s | 8 s / 140 s |
| Agent install | 67 s / 152 s | 93 s / 140 s |
| Agent execution | 169 s / 3020 s | 198 s / 2400 s |
| Verifier | 10 s / 277 s | 10 s / 771 s |
| **Per trial** | **322 s / 3139 s** | **341 s / 2570 s** |

ACR is *faster* at agent install because a session gets 2 dedicated vCPUs, while 40
local containers fight over 8 cores. The two 600 s entries are build timeouts, and that
14 s median *includes* first-time image push and runtime deployment.

## AgentCore or Docker: how to compare them, and why this kit has not

`SANDBOX=docker` flips the provider and changes nothing else:

```bash
scripts/run_eval_bedrock.sh                                  # AgentCore
SANDBOX=docker N_CONCURRENT=16 scripts/run_eval_bedrock.sh   # local containers
uv run scripts/compare_jobs.py "$HARBOR_JOBS/<job-a>" "$HARBOR_JOBS/<job-b>"
```

**There is no measured docker number here, and an earlier version of this file claimed
one.** Three jobs named `eval-docker-*` were run before `run_eval_bedrock.sh` had the
`-e "$SANDBOX"` flag, so `SANDBOX=docker` was read by nobody and all three ran on
AgentCore. Their trial logs say `Reusing AgentCore runtime hb_…` outright. The flag and
the write-up landed in the same commit, which is exactly the sort of thing the oracle
gate exists to catch and nothing was gating *this*.

`compare_jobs.py` now prints the sandbox each job actually used, read from the trial
logs, so a mislabelled comparison cannot be published from here again.

**Before you run the docker half, check the architecture.** The task images are arm64.
On an arm64 host (a Graviton) docker runs them natively; on x86 it needs qemu, which
this kit measured at **15.5–19× slower** — a comparison of the sandbox would instead be
a measurement of emulation. Confirm with `docker run --rm --platform linux/arm64
arm64v8/alpine uname -m`: `exec format error` means the host cannot run them at all.

What can still be said without that run, because it follows from what *was* measured:

- **The concurrency ceiling differs, whatever the per-trial speed turns out to be.** A
  docker container here is 2 vCPU of the host, so 32 vCPU tops out near 16 concurrent.
  AgentCore's ceiling is service-side (25 sessions/s) and was nowhere near reached — the
  measured runs achieved **14.9×** of a configured 16 while this box's load average sat
  at 0.1. Part 4 is where that bites: rollouts at concurrency 96 would need 192 vCPU of
  local containers, on the same box that is training.
- **Cost is model tokens either way.** $28 of Sonnet 5 dwarfs the sandbox: docker's is
  the instance you already rent (~$1 of c7gd.8xlarge for 45 minutes), AgentCore's is 397
  sessions × 91 s ≈ **10 hours of session time billed separately** — *not* in Harbor's
  `cost_usd`, which counts model tokens only. Check Cost Explorer for the real figure;
  the role on this host could not (`ce:GetCostAndUsage` denied).

### The number that should change how you read every other number

The two 397-task runs above were, it turns out, **the same configuration twice** — and
that makes them a cleaner measurement than the one they were mislabelled as.
`compare_jobs.py` compares per task, not just in aggregate, and they **disagreed on 55 of
397 tasks — they agreed on 86%**, while landing 0.8 points apart in aggregate (46.9% vs
46.1%). With the sandbox eliminated as a variable, the cause is unambiguous:
`temperature: 0` is not deterministic end to end — server-side batching, KV-cache reuse
and floating-point reduction order all move logits between otherwise identical requests.

A 13.9% flip rate, roughly symmetric, puts the standard deviation of a single 397-task run
at **about ±1.8 percentage points**. So:

- 46.9% versus 46.1% is the **same configuration disagreeing with itself**. Read it as
  the size of the noise, not as a result about anything.
- Comparing two models, or two checkpoints, on one run of this set cannot resolve a
  difference smaller than roughly **4 points**. Part 4's warning that "a 70-task eval set
  cannot resolve 10% from 13%" is the same fact with a smaller n.
- Report n and the scaffold with any number from here, and re-run before believing a small
  delta.

## Reading the output

```bash
uv run scripts/summarize.py "$HARBOR_JOBS/<job-name>"
uv run scripts/wandb_report.py "$HARBOR_JOBS/<job-name>" --project "$WANDB_PROJECT"
```

`summarize.py` prints per-task reward and stage timings; `wandb_report.py` is optional
and results stay on disk without it.

## A reward of 0 means two different things

This distinction matters more here than anywhere else in the kit, and the job output
does not make it for you:

- **the policy failed** — it produced a patch and the tests rejected it. A real signal.
- **the task never gave it a chance** — the environment died, the verifier hung, the
  context overflowed, `import numpy` SIGILLed. Not a signal at all.

Part 2's oracle gate is what separates them, and it is why part 3 evaluates only
oracle-verified tasks. If you see a suspiciously round 0.0 across many tasks, check
`environment_setup` and `agent_execution` in `result.json` before concluding the model
is bad.

## Cost

Sandbox time is negligible; model tokens dominate and are driven by outliers.
**\[measured\]**, Opus 5 on terminal-bench: median ~$0.40/task, but one task cost
$10.09 because the agent explored for 3,020 seconds — the same task cost $2.08 on local
Docker with the same model and prompt, purely from a different exploration path.
**Estimate from the tail, not the mean.** The Qwen paths cost nothing per token beyond
the GPU you are already holding.

## What part 4 reuses from here

- `configs/eval-vllm.yaml`'s scaffold, mirrored in `part4-train/configs/harbor_trial_acr.yaml`
- `$HARBOR_DATASETS/swebv-arm64` as the **validation** set
- the same deployed runtimes
