#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export WANDB_MODE="${WANDB_MODE:-offline}"
MASTER_ADDR="127.0.0.1"
MASTER_PORT="29543"
DATA_ROOT_DIR="/home/data/users/sjq/cavla/dataset"
RUN_ROOT_DIR="/home/data/users/sjq/ckpts/spatial-memory-diffusion"
VLA_PATH="/home/data/huggingface/models--openvla--openvla-7b/snapshots/31f090d05236101ebfc381b61c674dd4746d4ce0"
DEPTH_ENCODER_CHECKPOINT="/home/data/users/sjq/ckpts/3dcavla/31f090d05236101ebfc381b61c674dd4746d4ce0+libero_spatial_cotdep+b8+lr-5e-05+lora-r32+dropout-0.0--image_aug--libero-spatial-cotdep-3dcavla--80000_chkpt/depth_projector1--80000_checkpoint.pt"
WANDB_ENTITY="sjq111-shanghai-jiaotong-university"
WANDB_PROJECT="spatial-memory-diffusion-smoke"

# if [[ "${CONDA_DEFAULT_ENV:-}" != "spatial-memory-vla" ]]; then
#   echo "Expected conda environment spatial-memory-vla; current CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}" >&2
#   exit 1
# fi

torchrun --standalone --nnodes 1 --nproc-per-node 1 --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" vla-scripts/finetune.py \
  --vla_path "${VLA_PATH}" \
  --data_root_dir "${DATA_ROOT_DIR}" \
  --dataset_name libero_spatial_cotdep \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --use_spatial_memory_diffusion True \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_depth True \
  --depth_encoder_checkpoint "${DEPTH_ENCODER_CHECKPOINT}" \
  --batch_size 1 \
  --learning_rate 5e-5 \
  --num_steps_before_decay 50 \
  --max_steps 20 \
  --save_freq 100 \
  --save_latest_checkpoint_only False \
  --shuffle_buffer_size 16 \
  --image_aug True \
  --use_lora True \
  --lora_rank 4 \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_log_freq 10 \
  --action_model_type DiT-S \
  --action_diffusion_steps 100 \
  --repeated_diffusion_steps 1 \
  --run_id_note diffusion_test_vlm_cog_3d_perattn_smoke
