import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from ucas_val_tune import logit_np, official_face_info, read_labels, sigmoid_np


ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def best_box(raw_box):
    if raw_box is None or len(raw_box) == 0:
        return None
    if isinstance(raw_box[0], (list, tuple)):
        raw_box = max(raw_box, key=lambda b: (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])))
    if len(raw_box) < 4:
        return None
    return [float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3])]


def crop_face(image, image_name, face_info, scale=1.3):
    width, height = image.size
    stem = Path(image_name).stem
    entry = face_info.get(stem) or face_info.get(image_name) or {}
    box = best_box(entry.get("box"))
    if box is None:
        return image
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * scale
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    left = max(int(round(cx - side * 0.5)), 0)
    top = max(int(round(cy - side * 0.5)), 0)
    right = min(int(round(cx + side * 0.5)), width)
    bottom = min(int(round(cy + side * 0.5)), height)
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def to_tensor(image, size=256):
    image = image.convert("RGB").resize((size, size), RESAMPLE_BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).float()


class ImageFeatureDataset(Dataset):
    def __init__(self, items, face_info, image_size=256):
        self.items = items
        self.face_info = face_info
        self.image_size = image_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, name = self.items[idx]
        with Image.open(path) as image:
            face = crop_face(image.convert("RGB"), name, self.face_info, 1.3)
            tensor = to_tensor(face, self.image_size)
        return tensor


def dfgc_items_and_labels(dfgc_extract_root, dfgc_json):
    face_info = json.loads(Path(dfgc_json).read_text(encoding="utf-8"))
    root = Path(dfgc_extract_root)
    real_dir = root / "real_fulls"
    if not real_dir.is_dir():
        raise FileNotFoundError(real_dir)
    items = []
    labels = []
    for p in sorted(real_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            items.append((p, p.name))
            labels.append(0)
    for subset in sorted(root.iterdir()):
        if not subset.is_dir() or subset.name == "real_fulls":
            continue
        for p in sorted(subset.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((p, p.name))
                labels.append(1)
    return items, np.asarray(labels, dtype=np.int64), face_info


def ucas_items(data_folder):
    data_folder = Path(data_folder)
    img_list, face_info = official_face_info(data_folder)
    items = [(data_folder / "imgs" / name, name) for name in img_list]
    return items, img_list, face_info


def extract_features(items, face_info, model, device, batch_size, num_workers, image_size, cache_path=None):
    if cache_path and Path(cache_path).is_file():
        data = np.load(cache_path)
        return data["features"]
    dataset = ImageFeatureDataset(items, face_info, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if os.name != "nt" else 0,
        pin_memory=device.type == "cuda",
    )
    feats = []
    start = time.time()
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            out = model(batch)
            feats.append(out.float().cpu().numpy())
            if (i + 1) % 50 == 0:
                done = min((i + 1) * batch_size, len(dataset))
                print(f"features {done}/{len(dataset)} elapsed={time.time() - start:.1f}s", flush=True)
    features = np.concatenate(feats, axis=0).astype(np.float32)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, features=features)
    return features


def auc(labels, scores):
    return float(roc_auc_score(labels, scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfgc-root", default="datasets/DFGC-21-extracted")
    parser.add_argument("--dfgc-json", default="datasets/DFGC-21/bbox&landmarks.json")
    parser.add_argument("--ucas-val", default="datasets/UCAS_AISA/extracted/val")
    parser.add_argument("--eff-val-cache", default=None)
    parser.add_argument("--model-name", default="vit_small_patch16_dinov3_qkvb.lvd1689m")
    parser.add_argument("--weights", default="vit_small_patch16_dinov3_qkvb.lvd1689m.safetensors")
    parser.add_argument("--out-dir", default="dinov3_dfgc_probe_output")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    import timm

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.model_name, pretrained=False, num_classes=0, checkpoint_path=args.weights)
    model.to(device).eval()
    print(f"Loaded {args.model_name} on {device}, num_features={model.num_features}", flush=True)

    dfgc_train_items, dfgc_labels, dfgc_face_info = dfgc_items_and_labels(args.dfgc_root, args.dfgc_json)
    val_items, val_img_list, val_face_info = ucas_items(args.ucas_val)
    val_labels, label_file = read_labels(args.ucas_val, len(val_img_list))

    dfgc_features = extract_features(
        dfgc_train_items,
        dfgc_face_info,
        model,
        device,
        args.batch_size,
        args.num_workers,
        args.image_size,
        cache_path=out_dir / "dfgc21_features.npz",
    )
    val_features = extract_features(
        val_items,
        val_face_info,
        model,
        device,
        args.batch_size,
        args.num_workers,
        args.image_size,
        cache_path=out_dir / "ucas_val_features.npz",
    )

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            C=1.0,
            random_state=42,
        ),
    )
    clf.fit(dfgc_features, dfgc_labels)
    dino3_scores = clf.predict_proba(val_features)[:, 1]
    dino3_auc = auc(val_labels, dino3_scores)

    eff_auc = None
    fusion_auc = None
    fusion = None
    if args.eff_val_cache and Path(args.eff_val_cache).is_file():
        eff_cache = np.load(args.eff_val_cache, allow_pickle=True)
        eff_names = eff_cache["img_names"].astype(str).tolist()
        eff_scores = eff_cache["eff_views"].mean(axis=1).astype(np.float64)
        eff_by_name = {name: score for name, score in zip(eff_names, eff_scores)}
        eff_ordered = np.asarray([eff_by_name[name] for name in val_img_list], dtype=np.float64)
        fusion = sigmoid_np(0.55 * logit_np(eff_ordered) + 0.45 * logit_np(dino3_scores))
        eff_auc = auc(val_labels, eff_ordered)
        fusion_auc = auc(val_labels, fusion)

    joblib.dump(clf, out_dir / "dinov3_dfgc21_probe.joblib")
    score_payload = {
        "img_names": np.asarray(val_img_list),
        "labels": val_labels,
        "dinov3_scores": dino3_scores,
    }
    if fusion is not None:
        score_payload["eff_scores"] = eff_ordered
        score_payload["fusion_scores"] = fusion
    np.savez_compressed(out_dir / "ucas_val_scores.npz", **score_payload)
    summary = {
        "model_name": args.model_name,
        "weights": args.weights,
        "train_data": "DFGC-21 labels only, no UCAS val training",
        "label_file": label_file,
        "num_dfgc_train": int(len(dfgc_labels)),
        "num_ucas_val": int(len(val_labels)),
        "auc_dinov3_probe": dino3_auc,
    }
    if eff_auc is not None:
        summary["auc_efficientnet"] = eff_auc
        summary["auc_eff_dinov3_default_fusion_alpha_0.55"] = fusion_auc
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
