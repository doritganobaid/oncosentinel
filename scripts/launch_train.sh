#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT="${1:-main}"
SEED="${2:-0}"
OUTDIR="${OUTDIR:-runs}"
mkdir -p "${OUTDIR}"

fm-mapb calibrate \
  --experiment "${EXPERIMENT}" \
  --seed "${SEED}" \
  --output "${OUTDIR}/${EXPERIMENT}_seed${SEED}_head.pt"

echo "Reported full-scale setting: 8x A100 80GB, bf16 backbones, fp32 kernel, 187 GPU-hours."
echo "torchrun --standalone --nproc_per_node=8 -m oncosentinel calibrate --experiment ${EXPERIMENT}"
