#!/usr/bin/env bash

set -u

LRS=(
  "1e-2"
  "5e-3"
  "2e-3"
  "1e-3"
  "5e-4"
)

DEVICES="${DEVICES:-1}"
EXTRA_ARGS=("$@")

for lr in "${LRS[@]}"; do
  tag="${lr//-/m}"
  tag="${tag//./p}"

  echo "============================================================"
  echo "MoNuSeg sweep | lr=${lr}"
  echo "============================================================"

  python main.py fit \
    -c configs/monuseg/vit_query_mul_scale_fusion.yaml \
    --trainer.devices "${DEVICES}" \
    --model.lr "${lr}" \
    --trainer.logger.init_args.name "monuseg_vit_query_mul_scale_fusion_lr_${tag}" \
    --trainer.default_root_dir "runs/monuseg/lr_${tag}" \
    "${EXTRA_ARGS[@]}"

  status=$?
  if [ $status -ne 0 ]; then
    echo "[WARN] MoNuSeg run failed for lr=${lr} with exit code ${status}. Continuing."
  fi
done
