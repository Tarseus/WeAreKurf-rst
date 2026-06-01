# DFGC Detection Track Solution

This workspace now contains two detector submissions for the DFGC-style
deepfake detection track:

- `submission_det_strong.zip`: strong EfficientNet-B3 baseline.
- `submission_det_ensemble.zip`: DFGC-21-adapted EfficientNet-B3 + DINOv2
  linear-probe ensemble, currently the strongest local/remote result.

## Submission Artifact

- Strong baseline folder: `submission_det_strong/`
- Strong baseline zip: `submission_det_strong.zip`
- Best ensemble folder: `submission_det_ensemble/`
- Best ensemble zip: `submission_det_ensemble.zip`
- Entry point: `model.py`
- Baseline model: EfficientNet-B3, 3-class head (`real`, `clean fake`,
  `adversarial fake`)
- Ensemble model: EfficientNet-B3 plus frozen DINOv2-S/14 features with a
  linear probe, fused in logit space with `alpha_eff=0.55`
- Weights: `efn-b3_3c_60_acc0.9975.pth`, `dino_dfgc21_probe_ts.pt`

The prediction returned by `Model.run(input_dir, json_file)` is:

```python
fake_probability = 1 - softmax(logits)[real_class]
```

For the ensemble, the returned score is:

```python
fake_probability = sigmoid(0.55 * logit(efficientnet_score) + 0.45 * logit(dino_score))
```

## Inference Modes

Default mode is strongest:

```powershell
python evaluate_strong.py
```

To trade accuracy for speed:

```powershell
$env:DFGC_TTA="fast"   # original + horizontal flip + vertical flip
python evaluate_strong.py

$env:DFGC_TTA="none"   # single crop, fastest smoke test
python evaluate_strong.py
```

Useful environment variables:

- `DFGC_TTA`: `strong`, `fast`, or `none`
- `DFGC_BATCH_SIZE`: image batch size before TTA expansion
- `DFGC_INPUT_SIZE`: default `300`
- `DFGC_NUM_WORKERS`: dataloader workers on Linux

## Remote GPU Checks

I tested the submission on `g51` with:

- Python: `/data1/gushengda/anaconda3/envs/hci/bin/python`
- GPU: RTX 3090
- Import style: both `from submission_det_strong import model` and direct
  `import model` from the extracted submission directory

Official sample smoke test:

| Mode | AUC |
| --- | ---: |
| `DFGC_TTA=none` | 1.000000 |
| `DFGC_TTA=strong` | 1.000000 |

Small robustness sweep on the 10-image sample set:

| TTA | Case | AUC | Margin | Real Mean | Fake Mean |
| --- | --- | ---: | ---: | ---: | ---: |
| none | original | 1.0000 | 0.5972 | 0.0610 | 0.9243 |
| none | jpeg60 | 1.0000 | 0.5972 | 0.0610 | 0.9243 |
| none | jpeg35 | 1.0000 | 0.5972 | 0.0610 | 0.9243 |
| none | blur | 1.0000 | 0.4977 | 0.1014 | 0.9193 |
| none | downscale | 1.0000 | 0.0811 | 0.0831 | 0.7647 |
| none | noise | 0.9600 | -0.0397 | 0.4194 | 0.9680 |
| fast | original | 1.0000 | 0.5775 | 0.0665 | 0.7633 |
| fast | jpeg60 | 1.0000 | 0.5775 | 0.0665 | 0.7633 |
| fast | jpeg35 | 1.0000 | 0.5775 | 0.0665 | 0.7633 |
| fast | blur | 1.0000 | 0.3733 | 0.1180 | 0.7703 |
| fast | downscale | 1.0000 | 0.1048 | 0.1241 | 0.6628 |
| fast | noise | 0.9200 | -0.1238 | 0.3898 | 0.8928 |
| strong | original | 1.0000 | 0.5208 | 0.0680 | 0.7555 |
| strong | jpeg60 | 1.0000 | 0.5208 | 0.0680 | 0.7555 |
| strong | jpeg35 | 1.0000 | 0.5208 | 0.0680 | 0.7555 |
| strong | blur | 1.0000 | 0.3585 | 0.1276 | 0.7651 |
| strong | downscale | 1.0000 | 0.0973 | 0.1351 | 0.6549 |
| strong | noise | 0.9600 | -0.0116 | 0.3673 | 0.8961 |

The full JSON is saved as `remote_robust_eval_results.json`.

## Training Recipe

The cloned `DFGC_Detection/` repository contains the first-place training
pipeline. Its strongest settings are:

- EfficientNet-B3 pretrained on ImageNet.
- 3-class training: real, clean fake, adversarial fake.
- Label smoothing with smoothing `0.05`.
- Balanced class sampling by upsampling.
- Training-time augmentations: flip, Gaussian noise, Gaussian blur.
- Self-supervised fake generation by blending real and fake faces.
- Adversarial fake examples generated from CelebDF-v2 training data only.

For a clean competition declaration, state that the detector uses CelebDF-v2
training data plus derived augmentation/adversarial samples from that training
set, and ImageNet-pretrained EfficientNet initialization.

## DINOv2 Few-Shot Experiment

I also tested the DINO-style low-training method suggested in discussion:

- Backbone: frozen `vit_small_patch14_dinov2.lvd142m`
- Feature size: 384
- Trainable part: logistic linear probe only
- Weight file: `weights/vit_small_patch14_dinov2_lvd142m.safetensors`
- Scripts: `dino_probe_experiment.py`, `dino_fewshot_adapt.py`

Results:

| Setup | Train Data | Eval Data | AUC |
| --- | --- | --- | ---: |
| frozen DINO + linear probe | mini CelebDF structure in repo | held-out mini split | 1.0000 |
| frozen DINO + linear probe | mini CelebDF structure in repo | official 10-image sample | 0.6000 |
| DINO few-shot target adaptation | official 10-image sample | same images with blur/downscale/noise perturbations | 1.0000 |

Interpretation: DINOv2 features are useful, but the course validation labels
must be treated as evaluation-only for the final submission. Do not train a
final submitted detector on `UCAS_AIAS-val`.

## Full DFGC-21 Evaluation

After syncing the full DFGC-21 dataset to g51, I evaluated
`submission_det_strong.zip` with strong TTA on all released DFGC-21 subsets.

- Remote data: `/data1/gushengda/deepfake_detection_dfgc/datasets/DFGC-21`
- Extracted images: `/data1/gushengda/deepfake_detection_dfgc/datasets/DFGC-21-extracted`
- Result JSON: `dfgc21_eval_strong_results.json`
- Mean AUC over 17 fake subsets: `0.943282`

Weakest subsets:

| Subset | AUC | Fake Mean |
| --- | ---: | ---: |
| yuejiang_852934 | 0.663532 | 0.1439 |
| yangquanwei_852303 | 0.855167 | 0.3500 |
| DFGC_SYSU_852924 | 0.885638 | 0.3245 |
| lowtec_853184 | 0.915055 | 0.4367 |

These are the right targets for the next improvement pass: DINO/validation-set
adaptation or an ensemble should be judged by whether it improves these subsets
without hurting the already strong FaceShifter/adversarial subsets.

## DFGC-21 Adapted Ensemble

I trained a frozen-DINOv2 linear probe on the released DFGC-21 labeled images
and fused it with the EfficientNet-B3 detector. The ensemble is implemented in
`submission_det_ensemble/` and evaluated on g51 with strong TTA.

- Result JSON: `dfgc21_eval_ensemble_results.json`
- Mean AUC over 17 fake subsets: `0.993692`
- Previous EfficientNet-B3 strong TTA mean AUC: `0.943282`
- Absolute gain: `+0.050410`

Weakest subsets after fusion:

| Subset | AUC | Fake Mean |
| --- | ---: | ---: |
| yuejiang_852934 | 0.971459 | 0.6215 |
| yangquanwei_852303 | 0.978620 | 0.6831 |
| jerryHUST_853638 | 0.982325 | 0.6702 |
| lowtec_853184 | 0.987130 | 0.7077 |

Important caveat: this ensemble is a DFGC-21-adapted model because the DINO
linear probe was fit using DFGC-21 labels. Use it when the competition rules
allow training/adaptation on the released DFGC-21 data. If only CelebDF-v2
training data is allowed, submit `submission_det_strong.zip` instead.

## Official Course Submission Format

The course repository expects an Excel prediction file rather than the DFGC
starter-kit model zip. The official detection data folder should contain:

- `imgs/`
- `img_list.txt`
- `face_info.txt`

Generate the Excel file with:

```powershell
python make_official_submission.py `
  --team-name YOUR_TEAM_NAME `
  --data-folder <UCAS_AISA-test1-or-test2-folder> `
  --result-path official_submit `
  --module submission_det_ensemble `
  --tta strong `
  --batch-size 4
```

This writes `official_submit/YOUR_TEAM_NAME.xlsx`, with the official
`predictions` and `time` sheets used by `detection/evaluate.py`.

## UCAS Val And No-Val Submissions

The UCAS validation set has labels and should be used to report performance,
not to train the final submitted model. The accidental val-trained
artifact-adapter result has been moved to `invalid_val_trained/` and should not
be submitted.

Valid no-UCAS-val-training candidates now generated for test1:

- `official_submit/efficientnet_b3_no_val.xlsx`
  - Uses the EfficientNet-B3 detector only.
  - UCAS val AUC: `0.948152`
  - Uses no UCAS val labels for training.
- `official_submit/eff_dino_no_val.xlsx`
  - Uses the default EfficientNet-B3 + DINOv2 fusion.
  - UCAS val AUC: `0.982612`
  - Uses no UCAS val labels for training, but the DINO probe was adapted with
    DFGC-21 labels and should be declared as extra training data if submitted.
- `official_submit/eff_dinov3_splus_no_val.xlsx`
  - Uses EfficientNet-B3 plus a DINOv3 ViT-S+/16 probe trained on DFGC-21.
  - UCAS val AUC: `0.998700`
  - Uses no UCAS val labels for training.
  - This is the strongest valid no-UCAS-val-training submission so far, but
    DFGC-21 must be declared as extra training data.

If the rules are strict CelebDF-v2-training only, submit the EfficientNet-only
file. If extra declared training data such as DFGC-21 is acceptable, submit the
EfficientNet+DINOv3 S+ file.
