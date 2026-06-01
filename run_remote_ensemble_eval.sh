#!/usr/bin/env bash
set -u

cd /data1/gushengda/deepfake_detection_dfgc
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export DFGC_TTA="${DFGC_TTA:-strong}"
export DFGC_BATCH_SIZE="${DFGC_BATCH_SIZE:-4}"

/data1/gushengda/anaconda3/envs/hci/bin/python eval_dfgc21_any_submission.py \
  --module submission_det_ensemble \
  --out dfgc21_eval_ensemble_results.json \
  --tta "${DFGC_TTA}" \
  --batch-size "${DFGC_BATCH_SIZE}"

echo "EXIT_CODE:$?"
