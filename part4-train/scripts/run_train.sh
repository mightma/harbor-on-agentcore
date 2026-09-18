#!/usr/bin/env bash
# GRPO with SkyRL on 8x H100, FSDP, and every rollout's sandbox an AgentCore
# Runtime session.
#
#   Train: SWE-smith arm64      (part 1 built it, part 2 gated it)
#   Eval:  SWE-bench Verified arm64 held-out repositories
#
# That split is the point of using SWE-smith at all: the training set and the
# evaluation benchmark share no instances and no repositories, which the
# SWE-bench-only setup could not offer (see part 1).
#
#   scripts/run_train.sh                                  # $POLICY_MODEL
#   scripts/run_train.sh Qwen/Qwen3.5-4B                  # the measured config
#   MAX_STEPS=1 CONCURRENCY=16 BATCH=8 scripts/run_train.sh   # shakedown, do this first
#   scripts/run_train.sh Qwen/Qwen3-Coder-30B-A3B-Instruct TP=4 ENGINES=2
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
#   colocate_all stays true           Even at 8 GPUs SkyRL's own large recipe
#                                     keeps it: sleeping the engine while
#                                     training is worth more than the swap cost,
#                                     and it is what will let a 30B policy fit.
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
RL_EVAL_DATA="${RL_EVAL_DATA:-$HARBOR_DATASETS/swebv-arm64}"

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
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export FLA_TILELANG="${FLA_TILELANG:-0}"
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
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
POLICY_GPUS="${POLICY_GPUS:-8}"
ENGINES="${ENGINES:-8}"
TP="${TP:-1}"
MICRO_TRAIN="${MICRO_TRAIN:-1}"
MICRO_FORWARD="${MICRO_FORWARD:-1}"
USE_KL_LOSS="${USE_KL_LOSS:-false}"
LOGGER="${LOGGER:-wandb}"

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
REMOVE_MICROBATCH_PADDING="${REMOVE_MICROBATCH_PADDING:-false}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-true}"

# With colocate_all the engines share the policy's GPUs, so the engine grid has
# to land exactly on them. Catch it here instead of in a Ray placement-group
# timeout that never resolves.
if [ "$((ENGINES * TP))" -ne "$POLICY_GPUS" ]; then
  echo "ENGINES*TP ($ENGINES*$TP) must equal POLICY_GPUS ($POLICY_GPUS)" >&2
  exit 1
fi

if [ "$LOGGER" = "wandb" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  echo "LOGGER=wandb but WANDB_API_KEY is unset (set it in config.env)" >&2
  exit 1
fi

OUT="$KIT_WORK_DIR/runs/$RUN_NAME"
mkdir -p "$OUT"

# SkyRL always merges examples/.../harbor_trial_config/default.yaml, so replacing
# that file is the way to change the Harbor trial defaults.
cp "$PART/configs/harbor_trial_acr.yaml" \
   "$SKYRL_DIR/examples/train_integrations/harbor/harbor_trial_config/default.yaml"

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

# -L: make_train_set.sh symlinks the task dirs.
n_train=$(find "$RL_TRAIN_DATA/" -mindepth 1 -maxdepth 1 \( -type d -o -type l \) | wc -l)
echo "run=$RUN_NAME model=$MODEL"
echo "train=$RL_TRAIN_DATA ($n_train tasks)  eval=$RL_EVAL_DATA"
echo "steps=$MAX_STEPS"
echo "rollouts/step=$((BATCH * N_SAMPLES)) concurrency=$CONCURRENCY gpus=$POLICY_GPUS engines=${ENGINES}xTP${TP}"

cd "$SKYRL_DIR"
exec uv run --extra fsdp --extra harbor -m examples.train_integrations.harbor.entrypoints.main_harbor \
  data.train_data="['$RL_TRAIN_DATA']" \
  data.val_data="['$RL_EVAL_DATA']" \
  harbor_trial_config.trials_dir="$OUT/trials" \
  harbor_trial_config.agent.kwargs.max_turns="$MAX_TURNS" \
  harbor_trial_config.agent.kwargs.model_info.max_input_tokens="$MAX_MODEL_LEN" \
  trainer.policy.model.path="$MODEL" \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=true \
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
