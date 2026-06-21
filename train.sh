#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export WANDB_MODE="${WANDB_MODE:-online}"
MASTER_ADDR="127.0.0.1"
MASTER_PORT="29542"
DATA_ROOT_DIR="/home/data/users/sjq/cavla/dataset"
RUN_ROOT_DIR="/home/data/users/sjq/ckpts/spatial-memory-diffusion"
VLA_PATH="${VLA_PATH:-/home/data/huggingface/openvla-7b-prismatic}"
WANDB_ENTITY="sjq111-shanghai-jiaotong-university"
WANDB_PROJECT="spatial-memory-diffusion-prismatic-lora-vlm-only-bs8"
RESUME_TRAIN="False"
RESUME_STEP=""
RESUME_CKPT_DIR="/home/data/users/sjq/ckpts/spatial-memory-vla/31f090d05236101ebfc381b61c674dd4746d4ce0+libero_spatial_cotdep+b1+lr-5e-05+lora-r16+dropout-0.0+dit-l+gated-3d-memory--image_aug--gated_3d_memory_dit--25000_chkpt"
RUN_ID_NOTE="memoryvla_form_no_memory_bank"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/train_${RUN_ID_NOTE}_$(date +%Y%m%d_%H%M%S).log"

export WANDB_API_KEY="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"

RESUME_ARGS=()
if [[ "${RESUME_TRAIN}" == "True" || "${RESUME_TRAIN}" == "true" ]]; then
  RESUME_ARGS+=(--resume True)
  RESUME_ARGS+=(--resume_checkpoint_dir "${RESUME_CKPT_DIR}")
  if [[ -n "${RESUME_STEP}" ]]; then
    RESUME_ARGS+=(--resume_step "${RESUME_STEP}")
  fi
fi

mkdir -p "${LOG_DIR}"

nohup torchrun --standalone --nnodes 1 --nproc-per-node 1 --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" vla-scripts/finetune.py \
  --vla_path "${VLA_PATH}" \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --dataset_name libero_spatial_cotdep \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --use_spatial_memory_diffusion False \
  --use_vlm_diffusion True \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_depth False \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --num_steps_before_decay 100000 \
  --max_steps 50000 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --shuffle_buffer_size 32 \
  --image_aug True \
  --use_lora True \
  --merge_lora_during_training False \
  --lora_rank 16 \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_log_freq 10 \
  --action_model_type DiT-L \
  --action_diffusion_steps 100 \
  --per_token_size 256 \
  --repeated_diffusion_steps 4 \
  --run_id_note "${RUN_ID_NOTE}" \
  "${RESUME_ARGS[@]}" \
  > "${LOG_FILE}" 2>&1 &

TRAIN_PID=$!
disown "${TRAIN_PID}" 2>/dev/null || true

echo "Started spatial-memory-diffusion training in background."
echo "PID: ${TRAIN_PID}"
echo "Log: ${LOG_FILE}"
