# Part 3 · Evaluate a model with Harbor + AgentCore Runtime

Score two policies on the same 70 SWE-bench Verified arm64 tasks, with every trial's
sandbox an ACR session:

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

Both default to `data/swebv-arm64-eval-verified.txt` — the 70 of 73 eval tasks whose
oracle ceiling is 1.0. The other three are excluded on purpose: two sphinx instances
have PASS_TO_PASS tests that fail regardless of the patch, and `psf__requests-2317`'s
verifier hangs on network calls. Including them subtracts a constant from every score
and teaches you nothing.

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

**Claude Sonnet 5 on this task set is untested.** The Bedrock path is exercised
(terminal-bench, Opus 5) but this specific number has not been produced. Run it before
quoting it.

Per-stage timings from the earlier terminal-bench comparison, ACR against local Docker,
**\[measured\]** median/max:

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
- `$HARBOR_DATASETS/swebv-arm64/eval` as the **validation** set
- the same deployed runtimes
