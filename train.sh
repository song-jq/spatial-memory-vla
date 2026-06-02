#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

# export PYTHON_BIN="/home/data/users/sjq/anaconda3/envs/cavla3d/bin/python"
export WANDB_MODE="${WANDB_MODE:-online}"
MASTER_ADDR="127.0.0.1"
MASTER_PORT="29541"
DATA_ROOT_DIR="/home/data/users/sjq/cavla/dataset"
RUN_ROOT_DIR="/home/data/users/sjq/ckpts/spatial-memory-vla"
VLA_PATH="/home/data/huggingface/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0"
DEPTH_ENCODER_CHECKPOINT="/home/data/users/sjq/ckpts/3dcavla/31f090d05236101ebfc381b61c674dd4746d4ce0+libero_spatial_cotdep+b8+lr-5e-05+lora-r32+dropout-0.0--image_aug--libero-spatial-cotdep-3dcavla--80000_chkpt"
WANDB_ENTITY="sjq111-shanghai-jiaotong-university"
WANDB_PROJECT="spatial-memory-train-batch_size_8"
RESUME_TRAIN="False"
RESUME_DIR=""
RESUME_STEP=""

export WANDB_API_KEY="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"


RESUME_ARGS=()
if [[ "${RESUME_TRAIN}" == "True" || "${RESUME_TRAIN}" == "true" ]]; then
  RESUME_ARGS+=(--resume_train True --resume_dir "${RESUME_DIR}")
  if [[ -n "${RESUME_STEP}" ]]; then
    RESUME_ARGS+=(--resume_step "${RESUME_STEP}")
  fi
fi

torchrun --standalone --nnodes 1 --nproc-per-node 1 --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" vla-scripts/finetune.py \
  --vla_path "${VLA_PATH}" \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --dataset_name libero_spatial_cotdep \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --use_spatial_memory True \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --max_steps 150000 \
  --save_freq 10000 \
  --shuffle_buffer_size 32 \
  --image_aug True \
  --use_lora True \
  --lora_rank 16 \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_log_freq 10 \
  --depth_encoder_checkpoint "${DEPTH_ENCODER_CHECKPOINT}" \
  --dataloader_type stream \
  --group_size 16 \
  --mem_length 16 \
  --retrieval_layers 2 \
  --per_token_size 256 \
  --action_l1_log_freq 10 \
  --run_id_note memory3d \
  "${RESUME_ARGS[@]}"
