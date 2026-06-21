#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PRISMATIC_VLA_PATH="${PRISMATIC_VLA_PATH:-/home/data/huggingface/openvla-7b-prismatic}"
PRISMATIC_VLA_HF_PATH="${PRISMATIC_VLA_HF_PATH:-/home/data/huggingface/openvla-7b-prismatic-hf}"

python vla-scripts/extern/convert_openvla_weights_to_hf.py \
  --openvla_model_path_or_id "${PRISMATIC_VLA_PATH}" \
  --output_hf_model_local_path "${PRISMATIC_VLA_HF_PATH}"
