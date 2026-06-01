# Deepfake Detection Course Solution

This repository contains code for a UCAS DeepfakesAdvTrack-style deepfake
detection solution. It includes inference wrappers, DFGC/UCAS evaluation
utilities, DINO feature probing scripts, and official Excel submission helpers.

Datasets, generated predictions, report artifacts, and pretrained model weights
are intentionally not committed. See `DEEPFAKE_DETECTION_README.md` for the
full experiment log and result summary.

## Repository Contents

- `submission_det_strong/`: EfficientNet-B3 submission code.
- `submission_det_ensemble/`: EfficientNet-B3 plus DINO probe ensemble code.
- `make_official_submission.py`: Generates the official detection `.xlsx`
  submission file from a UCAS test folder.
- `evaluate_strong.py`, `eval_dfgc21_*.py`: Local and DFGC-style evaluation
  helpers.
- `dinov3_*.py`, `dino_*.py`: DINOv2/DINOv3 probing and fusion experiments.
- `scripts/`: Dataset download/sync helper scripts.

## What Is Not Included

The following files are excluded from Git and must be provided separately:

- UCAS/DFGC/CelebDF datasets and extracted images.
- Downloaded archives such as `Celeb-DF-v2.zip`.
- Pretrained and trained weights such as `.pth`, `.pt`, `.ckpt`,
  `.safetensors`, `.joblib`, and `.dat`.
- Generated submissions, feature caches, figures, and logs.

Expected checkpoint locations for the submission modules are:

```text
submission_det_strong/efn-b3_3c_60_acc0.9975.pth
submission_det_ensemble/efn-b3_3c_60_acc0.9975.pth
submission_det_ensemble/dino_dfgc21_probe_ts.pt
weights/vit_small_patch14_dinov2_lvd142m.safetensors
weights/dinov3_timm/*.safetensors
```

## Environment

Install the Python dependencies with:

```powershell
python -m pip install -r requirements.txt
```

GPU inference is recommended. The scripts use CPU fallback where practical, but
full test-set inference can be slow without CUDA.

## UCAS Official Submission

Prepare the official detection data folder with:

```text
UCAS_AISA-testX/
  imgs/
  img_list.txt
  face_info.txt
```

Generate an Excel submission:

```powershell
python make_official_submission.py `
  --team-name YOUR_TEAM_NAME `
  --data-folder path\to\UCAS_AISA-testX `
  --result-path official_submit `
  --module submission_det_ensemble `
  --tta strong `
  --batch-size 4
```

The output is written to:

```text
official_submit/YOUR_TEAM_NAME.xlsx
```

## Notes

If competition rules only allow CelebDF-v2 training data, use the EfficientNet
baseline. If declared extra training data such as DFGC-21 is allowed, the
ensemble/DINO variants can be used. Do not train on UCAS validation labels for a
final submission unless the rules explicitly permit it.
