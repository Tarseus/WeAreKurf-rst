import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from dinov3_dfgc_probe_ucas import extract_features, ucas_items
from ucas_val_tune import logit_np, sigmoid_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", default="datasets/UCAS_AISA/extracted/test1")
    parser.add_argument("--eff-cache", default="ucas_artifact_adapter_output/test1_feature_cache.npz")
    parser.add_argument("--probe", default="dinov3_splus_dfgc_probe_output/dinov3_dfgc21_probe.joblib")
    parser.add_argument("--model-name", default="vit_small_plus_patch16_dinov3_qkvb.lvd1689m")
    parser.add_argument("--weights", default="vit_small_plus_patch16_dinov3_qkvb.lvd1689m.safetensors")
    parser.add_argument("--out", default="ucas_test1_eff_dinov3_splus_no_val.json")
    parser.add_argument("--feature-cache", default="dinov3_splus_dfgc_probe_output/ucas_test1_features.npz")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--alpha-eff", type=float, default=0.55)
    args = parser.parse_args()

    import timm

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.model_name, pretrained=False, num_classes=0, checkpoint_path=args.weights)
    model.to(device).eval()

    items, img_list, face_info = ucas_items(args.data_folder)
    features = extract_features(
        items,
        face_info,
        model,
        device,
        args.batch_size,
        args.num_workers,
        args.image_size,
        cache_path=args.feature_cache,
    )
    probe = joblib.load(args.probe)
    dinov3_scores = probe.predict_proba(features)[:, 1].astype(np.float64)

    eff_cache = np.load(args.eff_cache, allow_pickle=True)
    eff_names = eff_cache["img_names"].astype(str).tolist()
    eff_scores = eff_cache["eff_views"].mean(axis=1).astype(np.float64)
    eff_by_name = {name: score for name, score in zip(eff_names, eff_scores)}
    eff_ordered = np.asarray([eff_by_name[name] for name in img_list], dtype=np.float64)

    fusion = sigmoid_np(args.alpha_eff * logit_np(eff_ordered) + (1.0 - args.alpha_eff) * logit_np(dinov3_scores))
    out = {
        "img_names": img_list,
        "predictions": [float(np.clip(x, 0.0, 1.0)) for x in fusion],
        "time": 0.0,
        "data_volume": len(img_list),
        "method": f"eff_dinov3_splus_dfgc_selected_fusion_alpha_{args.alpha_eff:.2f}",
        "uses_ucas_val_training": False,
        "train_data": "DFGC-21 labels for DINOv3 probe plus EfficientNet-B3 existing weights; no UCAS val labels",
        "model_name": args.model_name,
    }
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Images: {len(img_list)}")


if __name__ == "__main__":
    main()
