import argparse
import json
import os
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import timm
import torch
from PIL import Image, ImageFile, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REAL_FOLDERS = {"Celeb-real", "YouTube-real"}
FAKE_PREFIXES = ("Celeb-synthesis",)
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def best_box(raw_box):
    if raw_box is None or len(raw_box) == 0:
        return None
    if isinstance(raw_box[0], (list, tuple)):
        raw_box = max(
            raw_box,
            key=lambda box: max(float(box[2]) - float(box[0]), 1.0)
            * max(float(box[3]) - float(box[1]), 1.0)
            * (float(box[4]) if len(box) > 4 else 1.0),
        )
    if len(raw_box) < 4:
        return None
    return [float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3])]


def crop_face(image, name, face_info=None, scale=1.3):
    if not face_info:
        return image
    stem = Path(name).stem
    entry = face_info.get(stem) or face_info.get(name) or {}
    box = best_box(entry.get("box"))
    if box is None:
        return image

    width, height = image.size
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


def image_to_tensor(image, size):
    image = image.convert("RGB").resize((size, size), RESAMPLE_BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).float()


def collect_train_images(root):
    rows = []
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(root)
        top = rel.parts[0]
        if top in REAL_FOLDERS:
            label = 0
        elif top.startswith(FAKE_PREFIXES):
            label = 1
        else:
            continue
        rows.append((str(path), label))
    return rows


def collect_folder_images(folder):
    return [
        str(path)
        for path in sorted(Path(folder).iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def load_json(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dino(model_name, weight_path, device):
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        checkpoint_path=weight_path,
    )
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def extract_features(model, paths, device, image_size, batch_size, face_info=None):
    feats = []
    names = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start : start + batch_size]:
                name = Path(path).name
                with Image.open(path) as image:
                    image = crop_face(image, name, face_info=face_info)
                    tensors.append(image_to_tensor(image, image_size))
                names.append(name)
            batch = torch.stack(tensors, dim=0).to(device)
            feat = model(batch).detach().cpu().numpy()
            feats.append(feat)
    if not feats:
        return names, np.zeros((0, 384), dtype=np.float32)
    return names, np.concatenate(feats, axis=0)


def prepare_robust_cases(src_dir, out_root):
    src_dir = Path(src_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    names = [p.name for p in sorted(src_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    cases = {}
    for case in ["original", "jpeg35", "blur", "downscale", "noise"]:
        case_dir = out_root / case
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True)
        cases[case] = case_dir

    rng = np.random.default_rng(2026)
    for name in names:
        image = Image.open(src_dir / name).convert("RGB")
        image.save(cases["original"] / name)
        image.save(cases["jpeg35"] / name, quality=35)
        image.filter(ImageFilter.GaussianBlur(radius=1.4)).save(cases["blur"] / name)
        small = image.resize((max(8, image.width // 3), max(8, image.height // 3)), RESAMPLE_BILINEAR)
        small.resize(image.size, RESAMPLE_BILINEAR).save(cases["downscale"] / name)
        array = np.asarray(image).astype(np.float32)
        array = np.clip(array + rng.normal(0.0, 6.0, size=array.shape), 0, 255).astype(np.uint8)
        Image.fromarray(array).save(cases["noise"] / name)
    return cases


def train_probe(features, labels):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    clf = LogisticRegression(
        C=0.25,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=2026,
    )
    clf.fit(scaled, labels)
    return scaler, clf


def predict_probe(scaler, clf, features):
    return clf.predict_proba(scaler.transform(features))[:, 1]


def evaluate(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "real_mean": float(scores[labels == 0].mean()),
        "fake_mean": float(scores[labels == 1].mean()),
        "margin": float(scores[labels == 1].min() - scores[labels == 0].max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", default="DFGC_Detection/data_preparation/data_structure/Celeb-DF-v2-face")
    parser.add_argument("--sample-dir", default="DFGC_Detection/sample_imgs")
    parser.add_argument("--sample-json", default="DFGC_Detection/sample_meta.json")
    parser.add_argument("--dino-weights", default="vit_small_patch14_dinov2_lvd142m.safetensors")
    parser.add_argument("--model-name", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", default="dino_probe_output")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rows = collect_train_images(args.train_root)
    labels = np.asarray([label for _, label in rows], dtype=np.int64)
    paths = [path for path, _ in rows]
    print(f"training images: {len(paths)} real={(labels == 0).sum()} fake={(labels == 1).sum()}")

    model = build_dino(args.model_name, args.dino_weights, device)
    t0 = time.time()
    _, features = extract_features(model, paths, device, args.image_size, args.batch_size)
    print("feature shape:", features.shape, "extract seconds:", round(time.time() - t0, 2))

    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        stratify=labels,
        random_state=2026,
    )
    scaler, clf = train_probe(features[train_idx], labels[train_idx])
    val_scores = predict_probe(scaler, clf, features[val_idx])
    val_metrics = evaluate(labels[val_idx], val_scores)
    print("mini-val:", val_metrics)

    full_scaler, full_clf = train_probe(features, labels)
    joblib.dump(
        {
            "scaler": full_scaler,
            "classifier": full_clf,
            "model_name": args.model_name,
            "image_size": args.image_size,
        },
        out_dir / "dino_probe.joblib",
    )

    face_info = load_json(args.sample_json)
    sample_paths = collect_folder_images(args.sample_dir)
    sample_names, sample_features = extract_features(
        model,
        sample_paths,
        device,
        args.image_size,
        args.batch_size,
        face_info=face_info,
    )
    sample_scores = predict_probe(full_scaler, full_clf, sample_features)
    sample_labels = np.asarray([0] * 5 + [1] * 5, dtype=np.int64)
    sample_metrics = evaluate(sample_labels, sample_scores)
    print("sample:", sample_metrics)

    robust_cases = prepare_robust_cases(args.sample_dir, out_dir / "robust_cases")
    robust = []
    for case, folder in robust_cases.items():
        case_paths = collect_folder_images(folder)
        _, case_features = extract_features(
            model,
            case_paths,
            device,
            args.image_size,
            args.batch_size,
            face_info=face_info,
        )
        case_scores = predict_probe(full_scaler, full_clf, case_features)
        metrics = evaluate(sample_labels, case_scores)
        metrics["case"] = case
        robust.append(metrics)
        print("robust", case, metrics)

    result = {
        "train_count": int(len(paths)),
        "real_count": int((labels == 0).sum()),
        "fake_count": int((labels == 1).sum()),
        "mini_val": val_metrics,
        "sample": {
            **sample_metrics,
            "names": sample_names,
            "scores": [float(x) for x in sample_scores],
        },
        "robust": robust,
    }
    with open(out_dir / "dino_probe_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
