import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_dinov3_probe import extract_features, ucas_items
from ucas_val_tune import logit_np, sigmoid_np


def load_submission_module(submission_dir):
    submission_dir = str(Path(submission_dir).resolve())
    if submission_dir not in sys.path:
        sys.path.insert(0, submission_dir)
    if "model" in sys.modules:
        del sys.modules["model"]
    return importlib.import_module("model")


def efficientnet_scores(data_folder, face_info, img_list, args):
    if args.eff_cache and Path(args.eff_cache).is_file():
        eff_cache = np.load(args.eff_cache, allow_pickle=True)
        eff_names = eff_cache["img_names"].astype(str).tolist()
        eff_scores = eff_cache["eff_views"].mean(axis=1).astype(np.float64)
        eff_by_name = {name: score for name, score in zip(eff_names, eff_scores)}
        return np.asarray([eff_by_name[name] for name in img_list], dtype=np.float64)

    module = load_submission_module(args.submission_dir)
    dataset = module.FolderDataset(str(Path(data_folder) / "imgs"), face_info, tta_mode=args.tta)
    loader = DataLoader(
        dataset,
        batch_size=args.eff_batch_size,
        shuffle=False,
        num_workers=args.num_workers if os.name != "nt" else 0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = module.TransferModel("efficientnet-b3", num_out_classes=3)
    weight_path = Path(args.submission_dir) / "efn-b3_3c_60_acc0.9975.pth"
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.to(device).eval()

    softmax = nn.Softmax(dim=1)
    eff_views = []
    with torch.no_grad():
        for eff_batch, _ in loader:
            bs, views, channels, height, width = eff_batch.shape
            images = eff_batch.reshape(bs * views, channels, height, width).to(device)
            probs = 1.0 - softmax(model(images))[:, 0]
            eff_views.append(probs.reshape(bs, views).cpu().numpy())
    eff_views = np.concatenate(eff_views, axis=0).astype(np.float32)
    if args.eff_cache:
        Path(args.eff_cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.eff_cache, img_names=np.asarray(dataset.img_names), eff_views=eff_views)

    eff_scores = eff_views.mean(axis=1).astype(np.float64)
    eff_by_name = {name: score for name, score in zip(dataset.img_names, eff_scores)}
    return np.asarray([eff_by_name[name] for name in img_list], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--submission-dir", default="submission_det_ensemble")
    parser.add_argument("--eff-cache", default=None)
    parser.add_argument("--probe", default="dinov3_splus_dfgc_probe_output/dinov3_dfgc21_probe.joblib")
    parser.add_argument("--model-name", default="vit_small_plus_patch16_dinov3_qkvb.lvd1689m")
    parser.add_argument("--weights", default="vit_small_plus_patch16_dinov3_qkvb.lvd1689m.safetensors")
    parser.add_argument("--out", default="dinov3_mix_predictions.json")
    parser.add_argument("--team-name", default=None)
    parser.add_argument("--result-path", default=None)
    parser.add_argument("--feature-cache", default="dinov3_splus_dfgc_probe_output/ucas_features.npz")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eff-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--alpha-eff", type=float, default=0.55)
    parser.add_argument("--tta", default="strong", choices=["none", "fast", "strong"])
    args = parser.parse_args()

    import timm

    start = time.time()
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

    eff_ordered = efficientnet_scores(args.data_folder, face_info, img_list, args)

    fusion = sigmoid_np(args.alpha_eff * logit_np(eff_ordered) + (1.0 - args.alpha_eff) * logit_np(dinov3_scores))
    predictions = [float(np.clip(x, 0.0, 1.0)) for x in fusion]
    elapsed = time.time() - start
    out = {
        "img_names": img_list,
        "predictions": predictions,
        "time": elapsed,
        "data_volume": len(img_list),
        "method": f"eff_dinov3_splus_dfgc_selected_fusion_alpha_{args.alpha_eff:.2f}",
        "uses_ucas_val_training": False,
        "train_data": "DFGC-21 labels for DINOv3 probe plus EfficientNet-B3 existing weights; no UCAS val labels",
        "model_name": args.model_name,
    }
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"Wrote {args.out}")
    if args.team_name and args.result_path:
        result_path = Path(args.result_path)
        result_path.mkdir(parents=True, exist_ok=True)
        xlsx_path = result_path / f"{args.team_name}.xlsx"
        with pd.ExcelWriter(xlsx_path) as writer:
            pd.DataFrame({"img_names": img_list, "predictions": predictions}).to_excel(
                writer, sheet_name="predictions", index=False
            )
            pd.DataFrame({"Data Volume": [len(predictions)], "Time": [elapsed]}).to_excel(
                writer, sheet_name="time", index=False
            )
        print(f"Wrote {xlsx_path}")
    print(f"Images: {len(img_list)}")
    print(f"Seconds: {elapsed:.3f}")


if __name__ == "__main__":
    main()
