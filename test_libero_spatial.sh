#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DEFAULT_CKPT_PATH="/home/data/users/sjq/ckpts/spatial-memory-diffusion/31f090d05236101ebfc381b61c674dd4746d4ce0+libero_spatial_cotdep+b1+lr-5e-05+lora-r16+dropout-0.0+dit-l+permem-prefix+3d-perattn--image_aug--permem_prefix_depth_crossattn--25000_chkpt"

CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT_PATH}}"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/data/users/sjq/anaconda3/envs/spatial-memory-vla/bin/python}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_spatial}"
UNNORM_KEY="${UNNORM_KEY:-libero_spatial_cotdep}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
NUM_OPEN_LOOP_STEPS="${NUM_OPEN_LOOP_STEPS:-8}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-2}"
USE_PROPRIO="${USE_PROPRIO:-True}"
LORA_RANK="${LORA_RANK:-16}"
RUN_ID_NOTE="${RUN_ID_NOTE:-spatial-memory-diffusion-libero-spatial}"

if [[ ! -d "${CKPT_PATH}" ]]; then
  echo "Checkpoint directory does not exist: ${CKPT_PATH}" >&2
  echo "Set CKPT_PATH=/path/to/checkpoint before running this script." >&2
  exit 1
fi

LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-$(dirname "$(dirname "${CKPT_PATH}")")/eval_libero_spatial/$(basename "${CKPT_PATH}")}"
mkdir -p "${LOCAL_LOG_DIR}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable does not exist or is not executable: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN=/path/to/python before running this script." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${GPU_ID}}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${SCRIPT_DIR}/libero_config}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
unset DISPLAY

echo "Evaluating spatial-memory-diffusion on ${TASK_SUITE_NAME}"
echo "  checkpoint: ${CKPT_PATH}"
echo "  unnorm_key: ${UNNORM_KEY}"
echo "  log dir:    ${LOCAL_LOG_DIR}"

"${PYTHON_BIN}" experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "${CKPT_PATH}" \
  --task_suite_name "${TASK_SUITE_NAME}" \
  --unnorm_key "${UNNORM_KEY}" \
  --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
  --num_open_loop_steps "${NUM_OPEN_LOOP_STEPS}" \
  --num_images_in_input "${NUM_IMAGES_IN_INPUT}" \
  --use_proprio "${USE_PROPRIO}" \
  --use_l1_regression False \
  --use_diffusion False \
  --center_crop True \
  --lora_rank "${LORA_RANK}" \
  --run_id_note "${RUN_ID_NOTE}" \
  --local_log_dir "${LOCAL_LOG_DIR}" \
  "$@"
