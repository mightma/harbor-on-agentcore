#!/usr/bin/env bash
# GRPO with SkyRL on 8 GPUs, FSDP, and every rollout's sandbox an AgentCore
# Runtime session. Measured on 8x H100 80GB; the current policy needs the 143 GB
# of an H200 (see the sizing note below).
#
#   Train: SWE-smith arm64      (part 1 built it, part 2 gated it)
#   Eval:  SWE-bench Verified arm64 held-out repositories
#
# That split is the point of using SWE-smith at all: the training set and the
# evaluation benchmark share no instances and no repositories, which the
# SWE-bench-only setup could not offer (see part 1).
#
#   scripts/run_train.sh                                  # async, $POLICY_MODEL
#   ASYNC=0 scripts/run_train.sh                          # colocated and synchronous
#   MAX_STEPS=1 CONCURRENCY=16 BATCH=8 scripts/run_train.sh   # shakedown, do this first
#
# Async by default, and it is not only a scheduling change -- read this before
# comparing a run against a synchronous one.
#
#   ASYNC=1  Generation and training run *at the same time* on disjoint GPUs. The
#            engines never sleep, so the rollout wall clock stops being dead GPU
#            time -- which matters here because agentic rollouts dominate the step
#            (8 turns of model call + sandbox exec, then a pytest suite). Costs:
#            fewer GPUs for the policy, rollouts up to max_staleness_steps old, and
#            a different loss (see POLICY_LOSS_TYPE below).
#   ASYNC=0  One phase at a time on all 8 GPUs, engines sleeping while training.
#            Strictly on-policy, and the configuration this kit measured at 4B.
#
# The loss is the part that surprises people. Async cannot recompute old logprobs
# -- the policy that produced a rollout is already gone -- so the objective has to
# be anchored on the *rollout* logprobs. SkyRL asserts this outright: `regular`,
# `gspo`, `cispo` and friends are rejected for async, and its own async recipe uses
# `rollout_is`. So an async run and a sync run differ in algorithm, not just in
# schedule, and their reward curves are not directly comparable.
#
# Sizing, **[computed, not measured]**, 35.95B params at bf16 on 143 GB H200s.
# FSDP full-shard state is 16 bytes/param (params 2 + grads 2 + Adam m,v 8 + fp32
# master 4):
#
#   policy GPUs   8 -> 71.9 GB each    5 -> 115.0 GB    <- 28 GB for activations
#                 7 -> 82.2 GB         4 -> 143.8 GB    <- does not fit at all
#                 6 -> 95.9 GB  (async default, 47 GB spare)
#
# That is why the async split is 6+2 rather than upstream's 4+4: a quarter of this
# model's optimizer state does not fit on one H200. The two engine GPUs hold 72 GB
# of weights each at TP=1, leaving ~57 GB of KV cache per replica at 0.9.
#
# Under ASYNC=0 the engines share the policy's cards instead, so 72 GB of FSDP plus
# 72 GB of engine weights at TP=1 would be 144 GB on a 143 GB card -- hence
# ENGINES=2 TP=4 (18 GB/GPU) and GPU_MEM_UTIL 0.6 there. A mixture of experts is
# sized by its *total* parameters, not its active ones: 35B-A3B activates 3B per
# token and still carries 72 GB.
#
# Structured after SkyRL's own 8-GPU recipe
# (examples/train_integrations/harbor/run_codecontest.sh).
#
# Why the batch and concurrency are what they are:
#
#   policy_num_gpus_per_node 1 -> 8   FSDP only starts saving memory once the
#                                     model is actually sharded.
#   train_batch_size 2 -> 32          32 prompts x 8 samples = 256 rollouts per
#   n_samples_per_prompt 4 -> 8        step. On one box the ceiling was container
#                                     concurrency; with AgentCore each rollout
#                                     gets its own microVM.
#   max_concurrency 4 -> 128          The gate is no longer this host's 192 vCPU
#                                     at 2 CPUs per container. It is the
#                                     service's session-creation rate (25/s) and
#                                     how many trajectories a step needs.
#   environment docker -> agentcore   configs/harbor_trial_acr.yaml.
#   enforce_eager true -> false       Compile the engine; there is room now.
#
# What did NOT change, and is the part most likely to bite:
#
#   colocate_all                      now follows ASYNC: false when async (the
#                                     trainer asserts it), true when not.
#   remove_microbatch_padding=false   Qwen3.5 is Qwen3_5ForConditionalGeneration,
#                                     a VLM shell over hybrid linear/full
#                                     attention. FSDP's microbatch packing path
#                                     rejects it:
#                                       AssertionError: remove_microbatch_padding
#                                       is not supported with VLM vision inputs
#                                     This has nothing to do with GPU count — it
#                                     bites identically on an L40S and an H100,
#                                     and only after every rollout in step 1 has
#                                     already been paid for. See the part 4 README.
#   use_kl_loss stays false           There is room for a reference policy now,
#                                     but keeping it off means this run's
#                                     algorithm matches the L40S run exactly.
#                                     Set USE_KL_LOSS=true to turn it on.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PART="$(cd "$HERE/.." && pwd)"
set -a; . "$PART/../config.env"; set +a

MODEL="${1:-$POLICY_MODEL}"; shift || true
SERVED_NAME="$(basename "$MODEL")"
RUN_NAME="${RUN_NAME:-swesmith-$SERVED_NAME}"

SKYRL_DIR="${SKYRL_DIR:-$KIT_WORK_DIR/skyrl}"
RL_TRAIN_DATA="${RL_TRAIN_DATA:-$HARBOR_DATASETS/swesmith-arm64-train}"
RL_EVAL_DATA="${RL_EVAL_DATA:-$HARBOR_DATASETS/swebv-arm64-gated}"

# Both are load-bearing on this host and both fail late and confusingly.
#
#   NCCL_NVLS_ENABLE=0  FSDP's first collective dies with "Failed to bind NVLink
#     SHARP (NVLS) Multicast memory ... CUDA error 401". fabricmanager is active
#     and the NVSwitch devices are present, but `nvidia-smi -q` reports
#     "GPU Fabric GUID: N/A" -- inside a VM the GPUs are not registered in a
#     fabric that can hand out multicast handles. Ring/tree over NV18 is
#     unaffected; the cost is losing in-switch reduction.
#
#   FLA_TILELANG=0  Qwen3.5's gated-delta-rule layers come from
#     flash-linear-attention, which prefers a TileLang kernel and JITs it on
#     first use -- which is the first *backward*, i.e. after a full step of
#     rollouts is already paid for. It fails with "CUDA compiler and CUDA toolkit
#     headers are incompatible" because fla's own guard only checks that an nvcc
#     binary exists. This falls back to the Triton kernel.
#
#   VLLM_USE_FLASHINFER_SAMPLER=0  the third of the same kind, and the one that
#     bites first: FlashInfer JIT-compiles its top-k/top-p sampler during vLLM's
#     *warmup*, so the engine dies right after loading the weights with
#     "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist".
#     There is no CUDA toolkit on a DLAMI outside the venv. Falls back to the
#     PyTorch-native sampler: same distribution, different throughput. Measured on
#     part 3's eval before it reached part 4.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export FLA_TILELANG="${FLA_TILELANG:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# No EFA device on this instance; the DLAMI's aws-ofi-nccl plugin fails to
# initialise and then NCCL segfaults inside commAlloc() on the first collective.
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TURNS="${MAX_TURNS:-8}"
MAX_STEPS="${MAX_STEPS:-20}"
EPOCHS="${EPOCHS:-1}"
N_SAMPLES="${N_SAMPLES:-8}"
BATCH="${BATCH:-32}"
CONCURRENCY="${CONCURRENCY:-128}"
TRAJ_PER_SEC="${TRAJ_PER_SEC:-5}"
MICRO_TRAIN="${MICRO_TRAIN:-1}"
MICRO_FORWARD="${MICRO_FORWARD:-1}"
USE_KL_LOSS="${USE_KL_LOSS:-false}"
LOGGER="${LOGGER:-wandb}"
TOTAL_GPUS="${TOTAL_GPUS:-8}"

# ASYNC=1 overlaps rollout generation with training on disjoint GPUs; ASYNC=0 is the
# colocated, strictly-sequential loop. This is not a throughput knob with everything
# else held equal -- see the block below -- so it is an explicit choice.
ASYNC="${ASYNC:-1}"

if [ "$ASYNC" = "1" ]; then
  # Fully async. Generation and training run at the same time on separate GPUs, so
  # the engines never sleep and the rollout wall clock stops being dead GPU time.
  #
  # Why the split is 6+2 and not 4+4. FSDP full-shard state is 16 bytes/param
  # (params bf16 2 + grads bf16 2 + Adam m,v fp32 8 + fp32 master 4), so for 35.95B:
  #
  #   8 policy GPUs   71.9 GB each      4 policy GPUs  143.8 GB each -> does not fit
  #   7 policy GPUs   82.2 GB each      5 policy GPUs  115.0 GB each -> 28 GB spare
  #   6 policy GPUs   95.9 GB each  <-  47 GB spare for activations
  #
  # A 143 GB H200 cannot hold a quarter of this model's optimizer state, which is
  # what upstream's own async recipe assumes (NUM_POLICY_GPUS=4 for a smaller
  # policy). So the policy takes 6 and the engines take 2 -- 72 GB of weights on
  # each at TP=1, leaving ~57 GB of KV cache per replica at 0.9 utilisation.
  #
  # **[computed, not measured]** like the rest of this model's sizing.
  POLICY_GPUS="${POLICY_GPUS:-6}"
  ENGINES="${ENGINES:-2}"
  TP="${TP:-1}"
  # The engines have the card to themselves now, so there is nothing to leave room
  # for. This is upstream's async value.
  GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
  MAX_STALENESS="${MAX_STALENESS:-4}"
  # mini_batch <= workers <= mini_batch * (staleness + 1), asserted by the trainer.
  GEN_WORKERS="${GEN_WORKERS:-$((BATCH * 2))}"
  # Async cannot recompute old logprobs -- the policy that produced a rollout is
  # already gone -- so the loss has to optimize against the *rollout* logprobs.
  # `regular` (the sync default) is rejected outright; upstream's async recipe uses
  # rollout importance sampling. This changes the algorithm, not just the schedule.
  POLICY_LOSS_TYPE="${POLICY_LOSS_TYPE:-rollout_is}"
  ENTRYPOINT="examples.train_integrations.harbor.entrypoints.main_harbor_fully_async"
else
  # Colocated and synchronous: engines share the policy's GPUs and sleep during
  # training, so GPU_MEM_UTIL has to leave room for the FSDP shards.
  POLICY_GPUS="${POLICY_GPUS:-$TOTAL_GPUS}"
  ENGINES="${ENGINES:-2}"
  TP="${TP:-4}"
  GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"
  POLICY_LOSS_TYPE="${POLICY_LOSS_TYPE:-regular}"
  ENTRYPOINT="examples.train_integrations.harbor.entrypoints.main_harbor"
fi

# The two Qwen3.5 flags, both taken from SkyRL's own
# examples/train/models/run_qwen3.5_0.8b.sh rather than guessed:
#
#   remove_microbatch_padding=false  sample packing is not supported for Qwen3.5 on
#     the FSDP backend (GDN layers + flash attention:
#     huggingface/transformers#44910, QwenLM/Qwen3.5#104). Without it the run dies
#     on the first forward pass, *after* paying for a full step of rollouts:
#       AssertionError: remove_microbatch_padding is not supported with VLM vision inputs
#
#   language_model_only=true  Qwen3.5 ships as Qwen3_5ForConditionalGeneration, a
#     multimodal shell (image_token_id 248056) over a hybrid
#     linear/full-attention stack. Nothing here sends images, so the vision tower
#     is dead weight in the policy, the reference model and the engine alike.
#
# Both are set for any Qwen3.5-family policy. They are harmless on a plain
# text model, so they are not conditioned on the model name.
#
# The MoE members of the family are the same shell over the same hybrid stack:
# 35B-A3B is Qwen3_5MoeForConditionalGeneration, image_token_id 248056, 40 layers
# with full attention every 4th, 256 experts and 8 per token. So both flags carry
# over unchanged and FLA_TILELANG=0 stays load-bearing -- the gated-delta-rule
# layers are what needs it, and an MoE has more of them, not fewer. What does not
# carry over is the engine grid; see the sizing note in the header.
REMOVE_MICROBATCH_PADDING="${REMOVE_MICROBATCH_PADDING:-false}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-true}"

# With colocate_all the engines share the policy's GPUs, so the engine grid has
# to land exactly on them. Catch it here instead of in a Ray placement-group
# timeout that never resolves.
if [ "$ASYNC" = "1" ]; then
  # Disjoint sets: the policy and the engines each need their own cards, and
  # together they must not exceed the box.
  if [ "$((POLICY_GPUS + ENGINES * TP))" -ne "$TOTAL_GPUS" ]; then
    echo "async placement must use every GPU exactly once:" >&2
    echo "  POLICY_GPUS ($POLICY_GPUS) + ENGINES*TP ($ENGINES*$TP) != TOTAL_GPUS ($TOTAL_GPUS)" >&2
    exit 1
  fi
  # 16 bytes/param of FSDP state, against ~143 GB of H200. Refuse rather than OOM
  # after a step of rollouts has been paid for.
  fsdp_gb=$(python3 -c "print(round(16 * 35.95e9 / $POLICY_GPUS / 1e9))" 2>/dev/null || echo 0)
  if [ "$fsdp_gb" -gt 130 ]; then
    echo "POLICY_GPUS=$POLICY_GPUS puts ~${fsdp_gb} GB of FSDP state on each card," >&2
    echo "which leaves nothing for activations on a 143 GB GPU. Use 6 or more." >&2
    exit 1
  fi
  if [ "$BATCH" -gt "$GEN_WORKERS" ] \
     || [ "$GEN_WORKERS" -gt "$((BATCH * (MAX_STALENESS + 1)))" ]; then
    echo "the trainer asserts BATCH <= GEN_WORKERS <= BATCH*(MAX_STALENESS+1):" >&2
    echo "  BATCH=$BATCH GEN_WORKERS=$GEN_WORKERS MAX_STALENESS=$MAX_STALENESS" >&2
    exit 1
  fi
elif [ "$((ENGINES * TP))" -ne "$POLICY_GPUS" ]; then
  # Colocated: the engine grid has to land exactly on the policy's GPUs.
  echo "ENGINES*TP ($ENGINES*$TP) must equal POLICY_GPUS ($POLICY_GPUS)" >&2
  exit 1
fi

# The AgentCore environment needs an execution role, and part 4 is the one place an
# environment variable is not enough. Every other script in this kit exports
# HARBOR_AGENTCORE_ROLE_ARN and that works, because they run Harbor in their own
# process. Here the trials run inside Ray workers, and Ray only forwards the env vars
# in its runtime_env -- SkyRL puts NCCL/VLLM/HF/WANDB there, plus anything named
# SKYRL_*, and nothing else. So the role has to travel in the *trial config*, which is
# serialised into the worker; `harbor_trial_config.environment.kwargs.execution_role_arn`
# below is that route. The export is kept for the driver's own calls.
#
# Getting this wrong is silent in the worst way. Every trial raises before creating a
# sandbox, so the trial directory holds a 0-byte trial.log, no result.json and an empty
# agent/ -- while the training loop completes steps, writes checkpoints, and reports a
# reward of 0 for everything. **[measured]** 1152 trials, 0 completed, 5 checkpoints
# saved. `scripts/check_run.sh` is the 10-second way to see it.
if [ -z "${ACR_EXECUTION_ROLE_ARN:-}" ]; then
  echo "ACR_EXECUTION_ROLE_ARN is unset -- set it in config.env." >&2
  echo "Without it every rollout fails before creating a sandbox and every reward is 0," >&2
  echo "while the training loop happily reports steps." >&2
  exit 1
fi
export HARBOR_AGENTCORE_ROLE_ARN="$ACR_EXECUTION_ROLE_ARN"

if [ "$LOGGER" = "wandb" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  echo "LOGGER=wandb but WANDB_API_KEY is unset (set it in config.env)" >&2
  exit 1
fi

OUT="$KIT_WORK_DIR/runs/$RUN_NAME"
mkdir -p "$OUT"

# SkyRL always merges examples/.../harbor_trial_config/default.yaml, so replacing
# that file is the way to change the Harbor trial defaults.
trial_config="$SKYRL_DIR/examples/train_integrations/harbor/harbor_trial_config/default.yaml"
cp "$PART/configs/harbor_trial_acr.yaml" "$trial_config"

# SANDBOX=docker swaps the sandbox and nothing else. Unlike part 3 there is no `-e`
# to pass -- SkyRL constructs the trial itself -- so the type is rewritten in the
# config that was just copied. The agentcore-only kwargs below it are left in place:
# Harbor's docker environment takes **kwargs and ignores what it does not use, which
# is how part 3's SANDBOX=docker works against the same file.
#
# Two things to check before believing a docker run here, both of which cost more
# than the flag saves:
#
#   1. The task images are arm64. On x86 docker needs qemu -- measured at 15.5-19x
#      slower -- and without binfmt registered it cannot run them at all
#      (`exec format error`). This host is checked below.
#   2. Every container is 2 vCPU of *this* box, the one that is also training. At
#      CONCURRENCY=128 that is 256 vCPU of rollouts competing with the trainer for
#      CPU and RAM. AgentCore's sessions are elsewhere, which is the whole reason
#      part 4 uses it.
SANDBOX="${SANDBOX:-agentcore}"
if [ "$SANDBOX" != "agentcore" ]; then
  sed -i "s/^  type: agentcore$/  type: $SANDBOX/" "$trial_config"
  grep -q "^  type: $SANDBOX$" "$trial_config" || {
    echo "could not rewrite the environment type to '$SANDBOX' in $trial_config" >&2
    exit 1
  }
  if [ "$SANDBOX" = "docker" ] && ! docker run --rm --platform linux/arm64 \
       arm64v8/alpine true >/dev/null 2>&1; then
    echo "SANDBOX=docker, but this host cannot run an arm64 container." >&2
    echo "The task images are arm64. Register the emulator with" >&2
    echo "  docker run --privileged --rm tonistiigi/binfmt --install arm64" >&2
    echo "and expect 15-19x slower trials, or run on an arm64 host." >&2
    exit 1
  fi
fi
echo "sandbox=$SANDBOX"

# The provider pushes each task image to ECR on first use. `docker push` needs a
# login; do it once here so 32 concurrent trials do not each discover that.
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin \
      "$(aws sts get-caller-identity --query Account --output text).dkr.ecr.$AWS_REGION.amazonaws.com" \
      >/dev/null

if [ ! -d "$RL_TRAIN_DATA" ]; then
  echo "no training set at $RL_TRAIN_DATA -- run scripts/make_train_set.sh" >&2
  exit 1
fi

# SkyRL takes a *directory* of tasks with no per-task filter, so unlike part 3 the
# oracle gate cannot be applied here with a task list: whatever is in this directory is
# the validation set. A task that never passes the gate contributes an exception or a
# permanent reward of 0 to every epoch, which is indistinguishable from a model that
# cannot solve it. Hence a directory generated from the gated list.
if [ ! -d "$RL_EVAL_DATA" ]; then
  echo "no eval set at $RL_EVAL_DATA" >&2
  echo "it holds the gate-clean task dirs; after part 2's gate, generate them with:" >&2
  echo "  cd ../part1-arm64-images && swebench/tasks.sh \\" >&2
  echo "      swebench/data/swebv-arm64-gated.txt \"\$HARBOR_DATASETS/swebv-arm64-gated\"" >&2
  exit 1
fi

# -L: make_train_set.sh symlinks the task dirs.
n_train=$(find "$RL_TRAIN_DATA/" -mindepth 1 -maxdepth 1 \( -type d -o -type l \) | wc -l)
n_eval=$(find "$RL_EVAL_DATA/" -mindepth 1 -maxdepth 1 \( -type d -o -type l \) | wc -l)
echo "run=$RUN_NAME model=$MODEL"
echo "train=$RL_TRAIN_DATA ($n_train tasks)  eval=$RL_EVAL_DATA ($n_eval tasks)"
echo "steps=$MAX_STEPS"
echo "rollouts/step=$((BATCH * N_SAMPLES)) concurrency=$CONCURRENCY"
if [ "$ASYNC" = "1" ]; then
  echo "async: policy on $POLICY_GPUS GPU(s), ${ENGINES} engine(s) x TP${TP} on $((ENGINES * TP))," \
       "staleness<=$MAX_STALENESS, gen_workers=$GEN_WORKERS, loss=$POLICY_LOSS_TYPE"
  async_flags=(
    trainer.fully_async.enabled=true
    trainer.fully_async.max_staleness_steps="$MAX_STALENESS"
    trainer.fully_async.num_parallel_generation_workers="$GEN_WORKERS"
    # Weight sync does not invalidate the KV cache: the tokens already generated
    # stay valid, and re-prefilling every in-flight rollout on every step is what
    # async is trying to avoid.
    trainer.fully_async.clear_kv_cache_on_weight_sync=false
    trainer.placement.colocate_all=false
  )
else
  echo "sync: colocated on $POLICY_GPUS GPU(s), engines=${ENGINES}xTP${TP}, loss=$POLICY_LOSS_TYPE"
  async_flags=(trainer.placement.colocate_all=true)
fi

cd "$SKYRL_DIR"
exec uv run --extra fsdp --extra harbor -m "$ENTRYPOINT" \
  "${async_flags[@]}" \
  data.train_data="['$RL_TRAIN_DATA']" \
  data.val_data="['$RL_EVAL_DATA']" \
  harbor_trial_config.trials_dir="$OUT/trials" \
  harbor_trial_config.environment.kwargs.execution_role_arn="$ACR_EXECUTION_ROLE_ARN" \
  harbor_trial_config.agent.kwargs.max_turns="$MAX_TURNS" \
  harbor_trial_config.agent.kwargs.model_info.max_input_tokens="$MAX_MODEL_LEN" \
  trainer.policy.model.path="$MODEL" \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node="$POLICY_GPUS" \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.ref_num_gpus_per_node="$POLICY_GPUS" \
  trainer.placement.critic_num_gpus_per_node="$POLICY_GPUS" \
  trainer.remove_microbatch_padding="$REMOVE_MICROBATCH_PADDING" \
  trainer.policy.language_model_only="$LANGUAGE_MODEL_ONLY" \
  trainer.ref.language_model_only="$LANGUAGE_MODEL_ONLY" \
  generator.inference_engine.language_model_only="$LANGUAGE_MODEL_ONLY" \
  trainer.epochs="$EPOCHS" \
  trainer.max_training_steps="$MAX_STEPS" \
  trainer.train_batch_size="$BATCH" \
  trainer.policy_mini_batch_size="$BATCH" \
  trainer.micro_forward_batch_size_per_gpu="$MICRO_FORWARD" \
  trainer.micro_train_batch_size_per_gpu="$MICRO_TRAIN" \
  trainer.update_epochs_per_batch=1 \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.policy_loss_type="$POLICY_LOSS_TYPE" \
  trainer.algorithm.use_kl_loss="$USE_KL_LOSS" \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.algorithm.max_seq_len="$MAX_MODEL_LEN" \
  trainer.algorithm.off_policy_correction.tis_ratio_type=token \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=2.0 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=5 \
  trainer.hf_save_interval=5 \
  trainer.max_ckpts_to_keep=2 \
  trainer.ckpt_path="$OUT/ckpts" \
  trainer.export_path="$OUT/exports" \
  trainer.log_path="$OUT/logs" \
  trainer.logger="$LOGGER" \
  trainer.project_name="$WANDB_PROJECT" \
  trainer.run_name="$RUN_NAME" \
  trainer.resume_mode=latest \
  generator.inference_engine.served_model_name="$SERVED_NAME" \
  generator.inference_engine.num_engines="$ENGINES" \
  generator.inference_engine.tensor_parallel_size="$TP" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization="$GPU_MEM_UTIL" \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.engine_init_kwargs.max_model_len="$MAX_MODEL_LEN" \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  generator.sampling_params.max_generate_length=2048 \
  generator.n_samples_per_prompt="$N_SAMPLES" \
  generator.eval_n_samples_per_prompt=1 \
  generator.step_wise_trajectories=true \
  generator.merge_stepwise_output=true \
  generator.apply_overlong_filtering=true \
  generator.batched=false \
  generator.rate_limit.enabled=true \
  generator.rate_limit.trajectories_per_second="$TRAJ_PER_SEC" \
  generator.rate_limit.max_concurrency="$CONCURRENCY" \
  "$@"
