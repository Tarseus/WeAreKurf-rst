import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import timm
import torch
from PIL import Image, ImageFile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from dino_probe_experiment import (
    IMAGE_EXTENSIONS,
    crop_face,
    extract_features,
    image_to_tensor,
    load_json,
    prepare_robust_cases,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True


def collect_folder_images(folder):
    return [
        str(path)
        for path in sorted(Path(folder).iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def load_labels(label_file, names):
    if label_file:
        labels_by_name = {}
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "," in line:
                    name, label = line.split(",", 1)
                else:
                    name, label = line.split()[:2]
                labels_by_name[name] = int(label)
        return np.asarray([labels_by_name[Path(name).name] for name in names], dtype=np.int64)
    if len(names) == 10:
        return np.asarray([0] * 5 + [1] * 5, dtype=np.int64)
    raise ValueError("Provide --label-file for non-sample folders.")


def build_model(model_name, weight_path, device):
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


def train_probe(features, labels):
    scaler = StandardScaler()
    x = scaler.fit_transform(features)
    clf = LogisticRegression(
        C=0.05,
        class_weight="balanced",
        max_iter=2000,
        random_state=2026,
        solver="liblinear",
    )
    clf.fit(x, labels)
    return scaler, clf


def predict(scaler, clf, features):
    return clf.predict_proba(scaler.transform(features))[:, 1]


def metrics(labels, scores):
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
    parser.add_argument("--train-dir", default="DFGC_Detection/sample_imgs")
    parser.add_argument("--train-json", default="DFGC_Detection/sample_meta.json")
    parser.add_argument("--label-file", default=None)
    parser.add_argument("--dino-weights", default="vit_small_patch14_dinov2_lvd142m.safetensors")
    parser.add_argument("--model-name", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", default="dino_fewshot_output")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_name, args.dino_weights, device)

    train_paths = collect_folder_images(args.train_dir)
    train_names = [Path(p).name for p in train_paths]
    labels = load_labels(args.label_file, train_names)
    face_info = load_json(args.train_json)

    _, features = extract_features(
        model,
        train_paths,
        device,
        args.image_size,
        args.batch_size,
        face_info=face_info,
    )
    scaler, clf = train_probe(features, labels)
    train_scores = predict(scaler, clf, features)
    result = {"train": {**metrics(labels, train_scores), "scores": [float(x) for x in train_scores]}}
    print("train", result["train"])

    robust_cases = prepare_robust_cases(args.train_dir, out_dir / "robust_cases")
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
        scores = predict(scaler, clf, case_features)
        item = {**metrics(labels, scores), "case": case, "scores": [float(x) for x in scores]}
        robust.append(item)
        print("robust", case, item)
    result["robust"] = robust

    joblib.dump(
        {
            "scaler": scaler,
            "classifier": clf,
            "model_name": args.model_name,
            "image_size": args.image_size,
        },
        out_dir / "dino_fewshot_probe.joblib",
    )
    with open(out_dir / "dino_fewshot_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
