#!/usr/bin/env bash

set -u

DEVICES="${DEVICES:-1}"
EXTRA_ARGS=("$@")
FIXED_LR="1e-3"

for fold in 0 1 2 3 4; do
  run_name="monuseg_5fold_fold${fold}_vit_query_mul_scale_fusion_lr_1em3"
  fold_root="datasets/MoNuSeg_5fold/fold${fold}"

  echo "============================================================"
  echo "MoNuSeg 5-fold | fold=${fold} | TRAIN"
  echo "============================================================"

  python main.py fit \
    -c configs/monuseg/vit_query_mul_scale_fusion.yaml \
    --trainer.devices "${DEVICES}" \
    --model.lr "${FIXED_LR}" \
    --trainer.logger.init_args.name "${run_name}" \
    --trainer.default_root_dir "runs/${run_name}" \
    --data.train_dir "${fold_root}/Train_Folder" \
    --data.val_dir "${fold_root}/Test_Folder" \
    --data.test_dir "${fold_root}/Test_Folder" \
    "${EXTRA_ARGS[@]}"

  status=$?
  if [ $status -ne 0 ]; then
    echo "[WARN] Training failed for MoNuSeg fold${fold} with exit code ${status}. Continuing."
    continue
  fi
done

python scripts/summarize_5fold_best.py \
  runs \
  "monuseg_5fold_fold" \
  "runs/monuseg_5fold_vit_query_mul_scale_fusion_lr_1em3_summary.txt"
