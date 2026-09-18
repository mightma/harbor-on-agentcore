# Part 4 · Train a model with Harbor + AgentCore Runtime

GRPO with SkyRL on 8× H100, every rollout's sandbox an ACR session.

| | |
|---|---|
| Train | SWE-smith arm64, restricted to the oracle-clean images from part 2 |
| Eval | SWE-bench Verified arm64 — the whole test set, never split |
| Policy | `$POLICY_MODEL` |
| Trainer | SkyRL, FSDP, GRPO |

SWE-smith is why the eval set can stay whole: it shares no instances **and no
repositories** with Verified, so nothing has to be held back. An earlier version of this
kit trained on 208 Verified instances and evaluated on the other 73 -- the only option
when just 281 instances had arm64 images at all. Part 1 building the missing images
retired that compromise, and the 4B numbers below come from it, so read them as the shape
of the output rather than as this configuration's result.

## Layers

Three processes, and knowing which one owns what saves hours of misdirected debugging:

```
SkyRL          owns the policy, the optimizer, the vLLM engines, weight sync
  └─ Harbor    owns one trial: sandbox lifecycle, agent loop, verifier, reward
       └─ ACR  owns one sandbox: a 2 vCPU / 8 GB arm64 microVM
```

A reward of 0 can originate in any layer, and only the bottom two produce it silently.

## Setup

```bash
cd part4-train
uv sync                      # helper env: setup + task-set scripts

scripts/setup_skyrl.sh       # clones SkyRL, repoints its Harbor pin, syncs
scripts/make_train_set.sh    # SWE-smith restricted to the gate allowlist
```

### `setup_skyrl.sh` exists because of one silent failure

SkyRL pins Harbor at **0.13.1**, whose `EnvironmentType` has no `agentcore`. Every
rollout then fails validation *before* creating a sandbox:

```
1 validation error for TrialConfig
environment.type
  Input should be 'docker', 'daytona', 'e2b', ... [input_value='agentcore']
```

and the training loop **completes the step anyway, writes a checkpoint, and reports**:

```
avg_raw_reward: 0.0     response_length: 1.0     policy_loss: 0.0
```

`response_length: 1.0` is the only tell. Nothing raises. The script rewrites the pin in
`pyproject.toml` — not via `uv pip install`, because `uv run --extra fsdp --extra
harbor` re-syncs from the lock and puts 0.13.1 back — then asserts `agentcore` is
present before returning.

SkyRL touches Harbor through three imports (`TrialConfig`, `Trial.create/run`,
`RolloutDetail`) whose signatures are unchanged in 0.23.0, so the swap is safe.

### `make_train_set.sh` and why the allowlist matters

An image whose oracle scores 0.0 cannot reward a correct patch, so every rollout on it
is a guaranteed 0. That is not a hard task, it is a dead signal — and in GRPO a group of
dead signals contributes no advantage while still costing a full rollout each.
**\[measured\]** the gate excludes 11 of 119 images, keeping 38,815 of 44,489 tasks.

## Verify in ascending cost — each step is an order of magnitude cheaper than the next

This ordering is the single most useful habit in this kit. Every step below actually
caught a distinct failure class in practice:

```bash
# 1. one oracle trial            (~1 min)   catches numpy SIGILL, truncated patches
cd ../part2-agentcore-runtime
REPO_PREFIXES=hukkin__tomli.443a0c1b scripts/deploy_runtimes.sh "$HARBOR_DATASETS/swesmith-arm64" 2

# 2. the full gate               (~20 min)  catches per-image ceilings
scripts/deploy_runtimes.sh "$HARBOR_DATASETS/swesmith-arm64" 20
uv run scripts/gate_report.py "$HARBOR_JOBS"/gate-* --task-root "$HARBOR_DATASETS/swesmith-arm64" --allowlist-out "$KIT_STATE_DIR"

# 3. one-step shakedown          (~10 min)  catches NVLS, TileLang, the harbor pin
cd ../part4-train
MAX_STEPS=1 BATCH=8 CONCURRENCY=16 scripts/run_train.sh

# 4. the real run
scripts/run_train.sh
```

Steps 1–3 are cheap precisely because the expensive failures in this stack all happen
*after* a full step of rollouts has been paid for.

## Three things that fail late and confusingly

`scripts/run_train.sh` sets all three; this is what they are.

| Variable | Symptom without it | Cause |
|---|---|---|
| `NCCL_NVLS_ENABLE=0` | `Failed to bind NVLink SHARP (NVLS) Multicast memory ... CUDA error 401`, dies in `FSDPPolicyWorkerBase.init_model()` | fabricmanager is active and NVSwitch devices exist, but `nvidia-smi -q` reports `GPU Fabric GUID: N/A` — inside a VM the GPUs are not in a fabric that can hand out multicast handles. Ring/tree over NV18 is unaffected. |
| `FLA_TILELANG=0` | `CUDA compiler and CUDA toolkit headers are incompatible` on the **first backward** | Qwen3.5's gated-delta-rule layers come from flash-linear-attention, which prefers a TileLang kernel and JITs it on first use. Its guard only checks that an `nvcc` binary exists, so the mismatch surfaces at codegen. Falls back to Triton. |
| `NCCL_NET_PLUGIN=none` | NCCL segfaults in `commAlloc()` on the first collective | No EFA device here, so the DLAMI's aws-ofi-nccl plugin fails to initialise. |

### The harness is not a free choice here

`configs/harbor_trial_acr.yaml` sets `agent.name: terminus-2` and
`collect_rollout_details: true`, and neither is swappable today: step-wise RL needs
per-turn token ids and logprobs, which only a harness routing its model calls through
Harbor's own LLM layer can supply. The SWE-specialised harnesses
(`mini-swe-agent`, `swe-agent`, `claude-code`) are installed CLIs that call the model
themselves, so nothing records those calls. See **Open limitation** in the top-level
README for why this is a gap rather than a wall, and what would close it.

Plus two Qwen3.5-specific flags, both taken from SkyRL's own
`examples/train/models/run_qwen3.5_0.8b.sh` rather than guessed:

- `trainer.remove_microbatch_padding=false` — Qwen3.5 is
  `Qwen3_5ForConditionalGeneration`, a multimodal shell over a hybrid
  linear/full-attention stack. FSDP's sample-packing path rejects it with
  `AssertionError: remove_microbatch_padding is not supported with VLM vision inputs`.
  This is not about GPU count; it bites identically on one L40S and on eight H100s, and
  only after a full step of rollouts.
- `language_model_only=true` — nothing here sends images, so the vision tower is dead
  weight in the policy, the reference model and the engine alike.

## Bring your own model

**Training needs weights you hold.** `bedrock/…` and any hosted API are eval-only —
part 3 scores them, part 4 cannot train them. The policy has to be a checkpoint SkyRL
can both shard with FSDP and serve with its own vLLM engines, which in practice means a
local or Hub HF model directory. Set `POLICY_MODEL`, or pass it as the first argument:

```bash
scripts/run_train.sh Qwen/Qwen3.5-4B                       # the measured config
scripts/run_train.sh /path/to/my-checkpoint                # a local directory
```

**There is no `model_name` in `configs/harbor_trial_acr.yaml` on purpose.** SkyRL owns
the policy and injects `agent.model_name` and `agent.kwargs.api_base` per rollout,
pointing them at its own engine — so the provider-prefix rule from part 3 applies to
part 3's eval, not here. To see what a rollout actually used, read `agent_info` out of
any trial's result:

```bash
f=$(ls "$KIT_WORK_DIR"/runs/<name>/trials/*/result.json | head -1)
python -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['agent_info']); \
  print('rollout turns:', len(d['agent_result']['rollout_details'] or []))" "$f"
```

`agent_info.model_info.provider` should name a provider and
`agent_result.rollout_details` should be a **non-empty** list. `[]` there means the
engine did not return `logprobs`/`return_token_ids`, and the trainer is about to train
on nothing — see the harness matrix in the top-level README for the full check.

**What has to move with the model:**

| If you change | Also change |
|---|---|
| the context window | `MAX_MODEL_LEN` only — `run_train.sh` drives `model_info.max_input_tokens`, `trainer.algorithm.max_seq_len` and the engine's `max_model_len` from that one value, so editing the YAML instead is how they drift apart. 8 turns assumes 32k |
| to a non-Qwen3.5 model | drop `extra_body.chat_template_kwargs.enable_thinking`; a template that does not take the kwarg is not a no-op |
| to a text-only, non-hybrid architecture | `REMOVE_MICROBATCH_PADDING=true LANGUAGE_MODEL_ONLY=false` — both defaults exist only because Qwen3.5 is a VLM shell over hybrid attention, and sample packing is worth having back |
| model size | `MICRO_TRAIN` / `GPU_MEM_UTIL` / `TP` / `ENGINES`, in the order in "Scaling notes" below |

**What does not change:** the harness, the reward, and the task set. Swapping the policy
does not touch the scaffold, which is why a checkpoint from here can be scored by part 3
against the same 70 tasks with no conversion (see "Compare a checkpoint" below).

## Results

**\[measured\]** on the **SWE-bench-Verified-arm64** training variant with
Qwen3.5-**4B** — the SWE-smith training set described here was prepared but a full run
on it has not been done, so treat these as the shape of the output rather than as this
configuration's numbers.

One-step shakedown, 8 prompts × 8 samples = 64 rollouts at concurrency 32:

| Metric | Value |
|---|---|
| `avg_raw_reward` | 0.0625 (4 of 64 solved) |
| `policy_loss` | −0.0049 |
| `grad_norm` | 0.280 |
| `policy_entropy` | 0.403 |

Main run, 16 prompts × 8 samples = 128 rollouts/step, concurrency 96, ~8 min/step,
stopped at 17 steps:

| | |
|---|---|
| `avg_raw_reward` | mean **0.144**, range 0.047 – 0.305 |
| `policy_loss` | non-zero at **every** one of 17 steps |
| `grad_norm` | 0.10 – 0.26, no explosion, no collapse to 0 |
| `policy_entropy` | 0.31 – 0.38, no entropy collapse at this scale |

Non-zero `policy_loss` and `grad_norm` mean the chain **advantage → gradient → parameter
update → sync back to the engine** is live, on a real dataset rather than synthetic smoke
tasks. That was the thing the single-GPU setup could never demonstrate.

### This is not a learning curve

**Do not read the reward column as "training is working."** Two reasons:

1. **16 prompts per step**, over tasks with enormous difficulty variance (django/sympy
   contains both three-line fixes and cross-file refactors). The 0.047 ↔ 0.305 swing
   mostly reflects *which tasks were sampled this step*, not the policy changing. Read as
   a curve, it supports whatever conclusion you want.
2. **17 steps × 128 rollouts cannot move a 4B model** at this batch size, and a 70-task
   eval set cannot resolve 10% from 13% — two tasks is ~3 points.

What this part demonstrates is that the pipeline is sound and the sandbox is no longer
the bottleneck. Claiming an RL-driven SWE-bench improvement needs a bigger batch, more
steps, and a larger eval set — none of which are now limited by the sandbox layer. That
is the actual result.

## Compare a checkpoint against the baseline

Checkpoints land in `$KIT_WORK_DIR/runs/<name>/exports/global_step_N` in HF format, so
part 3 serves them directly with no conversion — the same eval, the same task list:

```bash
cd ../part3-evaluate
scripts/run_eval_vllm.sh "$KIT_WORK_DIR/runs/swesmith-Qwen3.5-9B/exports/global_step_15"
```

## Scaling notes

`config.env` defaults `POLICY_MODEL` to Qwen3.5-9B because that is what was asked for,
but **no run in this project used a 9B policy** — 4B is the largest measured. Expect to
retune, in this order:

| Knob | Default | If you OOM |
|---|---|---|
| `MICRO_TRAIN` / `MICRO_FORWARD` | 1 | already minimal; go to the next row |
| `GPU_MEM_UTIL` | 0.8 | lower to 0.7 — engines and policy share GPUs under `colocate_all` |
| `TP` / `ENGINES` | 1 / 8 | `TP=2 ENGINES=4`; `ENGINES*TP` must equal `POLICY_GPUS` or Ray hangs in placement |
| `BATCH` / `N_SAMPLES` | 32 / 8 | reduce `BATCH`; keep `N_SAMPLES` ≥ 4 or GRPO groups get too small to give advantage |

`colocate_all=true` stays on even at 8 GPUs — SkyRL's own large recipe keeps it, and
sleeping the engine while training is worth more than the swap cost. It is also what
would let a 30B policy fit.

On the previous single-GPU machine the binding constraint for 4B was **host RAM, not
VRAM**. Watch both.

## Clean up

The runtimes stay deployed across epochs on purpose. When the run is done:

```bash
cd ../part2-agentcore-runtime && uv run scripts/cleanup_runtimes.py --delete
```
