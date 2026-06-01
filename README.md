# WeAreKurf-rst Deepfake Detection Mix

This repository contains only the final mixed deepfake detector code used for
the course submission:

- `submission_det_ensemble/`: final mixed inference module.
- `make_official_submission.py`: runs inference on an official UCAS test folder
  and writes the required `.xlsx` submission.
- `dinov3_dfgc_probe_ucas.py`: trains the DINOv3 linear probe used by the mix.
- `dinov3_dfgc_alpha_search.py`: searches the EfficientNet/DINOv3 fusion
  coefficient.
- `apply_dinov3_dfgc_probe_test1.py`: applies the trained DINOv3 probe and
  fuses it with cached EfficientNet scores.
- `ucas_val_tune.py`: shared official-folder parsing, label loading, AUC, and
  fusion helpers.

Datasets, generated predictions, cached features, and pretrained/trained model
weights are intentionally not included.

## Excluded Files

Provide these locally when running the code:

```text
submission_det_ensemble/efn-b3_3c_60_acc0.9975.pth
submission_det_ensemble/dino_dfgc21_probe_ts.pt
weights/dinov3_timm/*.safetensors
datasets/
```

The repository ignores archives, datasets, caches, `.xlsx` submissions, and
model files such as `.pth`, `.pt`, `.ckpt`, `.safetensors`, `.joblib`, and
`.npz`.

## Install

```powershell
python -m pip install -r requirements.txt
```

GPU inference is recommended.

## Official Mix Inference

The official data folder must contain:

```text
UCAS_AISA-testX/
  imgs/
  img_list.txt
  face_info.txt
```

Generate a submission file:

```powershell
python make_official_submission.py `
  --team-name TeamKurfuerst `
  --data-folder path\to\UCAS_AISA-testX `
  --result-path official_submit `
  --module submission_det_ensemble `
  --tta strong `
  --batch-size 4
```

The output is:

```text
official_submit/TeamKurfuerst.xlsx
```

## Train DINOv3 Probe

Example:

```powershell
python dinov3_dfgc_probe_ucas.py `
  --dfgc-root datasets\DFGC-21-extracted `
  --dfgc-json datasets\DFGC-21\bbox&landmarks.json `
  --ucas-val datasets\UCAS_AISA\extracted\val `
  --eff-val-cache ucas_artifact_adapter_output\feature_cache.npz `
  --model-name vit_small_plus_patch16_dinov3_qkvb.lvd1689m `
  --weights weights\dinov3_timm\vit_small_plus_patch16_dinov3_qkvb.lvd1689m.safetensors `
  --out-dir dinov3_splus_dfgc_probe_output
```

Then search the fusion alpha:

```powershell
python dinov3_dfgc_alpha_search.py `
  --out-dir dinov3_splus_dfgc_probe_output
```

Keep generated `.joblib`, `.npz`, and result JSON files local; they are not part
of the public repository.
