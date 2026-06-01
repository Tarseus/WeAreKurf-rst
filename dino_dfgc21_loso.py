import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import timm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from dino_probe_experiment import extract_features, load_json


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


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


def image_paths(folder):
    return [
        str(path)
        for path in sorted(Path(folder).iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def load_or_extract_subset(model, subset_dir, cache_dir, face_info, device, image_size, batch_size):
    subset = Path(subset_dir).name
    feat_path = cache_dir / f"{subset}_features.npy"
    names_path = cache_dir / f"{subset}_names.json"
    if feat_path.exists() and names_path.exists():
        features = np.load(feat_path)
        names = json.loads(names_path.read_text(encoding="utf-8"))
        return names, features

    paths = image_paths(subset_dir)
    names, features = extract_features(
        model,
        paths,
        device,
        image_size,
        batch_size,
        face_info=face_info,
    )
    np.save(feat_path, features.astype(np.float32))
    names_path.write_text(json.dumps(names, indent=2), encoding="utf-8")
    return names, features


def train_probe(x, y, c=0.05):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    clf = LogisticRegression(
        C=c,
        class_weight="balanced",
        max_iter=2000,
        random_state=2026,
        solver="lbfgs",
    )
    clf.fit(x_scaled, y)
    return scaler, clf


def predict(scaler, clf, x):
    return clf.predict_proba(scaler.transform(x))[:, 1]


def auc(real_scores, fake_scores):
    labels = np.concatenate([
        np.zeros(len(real_scores), dtype=np.int64),
        np.ones(len(fake_scores), dtype=np.int64),
    ])
    scores = np.concatenate([real_scores, fake_scores])
    return float(roc_auc_score(labels, scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-root", default="datasets/DFGC-21-extracted")
    parser.add_argument("--meta-json", default="datasets/DFGC-21/bbox&landmarks.json")
    parser.add_argument("--dino-weights", default="vit_small_patch14_dinov2_lvd142m.safetensors")
    parser.add_argument("--model-name", default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache-dir", default="dino_dfgc21_cache")
    parser.add_argument("--out", default="dino_dfgc21_loso_results.json")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extract_root = Path(args.extract_root)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    face_info = load_json(args.meta_json)
    model = build_model(args.model_name, args.dino_weights, device)

    subset_dirs = sorted([p for p in extract_root.iterdir() if p.is_dir()])
    fake_subsets = [p.name for p in subset_dirs if p.name != "real_fulls"]
    features = {}
    names = {}
    for subset_dir in subset_dirs:
        print("features", subset_dir.name)
        names[subset_dir.name], features[subset_dir.name] = load_or_extract_subset(
            model,
            subset_dir,
            cache_dir,
            face_info,
            device,
            args.image_size,
            args.batch_size,
        )
        print(subset_dir.name, features[subset_dir.name].shape)

    real_x = features["real_fulls"]
    real_train = real_x[::2]
    real_eval = real_x[1::2]

    loso_results = []
    loso_scores = {"real_names": names["real_fulls"][1::2], "real_scores": {}}
    for holdout in fake_subsets:
        train_parts = [real_train]
        train_labels = [np.zeros(len(real_train), dtype=np.int64)]
        for subset in fake_subsets:
            if subset == holdout:
                continue
            train_parts.append(features[subset])
            train_labels.append(np.ones(len(features[subset]), dtype=np.int64))
        x_train = np.concatenate(train_parts, axis=0)
        y_train = np.concatenate(train_labels, axis=0)
        scaler, clf = train_probe(x_train, y_train)

        real_scores = predict(scaler, clf, real_eval)
        fake_scores = predict(scaler, clf, features[holdout])
        item = {
            "subset": holdout,
            "auc": auc(real_scores, fake_scores),
            "real_mean": float(real_scores.mean()),
            "fake_mean": float(fake_scores.mean()),
            "margin": float(fake_scores.min() - real_scores.max()),
        }
        print("LOSO", holdout, item)
        loso_results.append(item)

    # Train all-subset probe for candidate ensembling/calibration.
    x_all = np.concatenate([real_x] + [features[s] for s in fake_subsets], axis=0)
    y_all = np.concatenate(
        [np.zeros(len(real_x), dtype=np.int64)]
        + [np.ones(len(features[s]), dtype=np.int64) for s in fake_subsets],
        axis=0,
    )
    scaler, clf = train_probe(x_all, y_all)
    joblib.dump({"scaler": scaler, "classifier": clf, "model_name": args.model_name}, cache_dir / "dino_dfgc21_all_probe.joblib")

    all_results = []
    real_all_scores = predict(scaler, clf, real_x)
    for subset in fake_subsets:
        fake_scores = predict(scaler, clf, features[subset])
        item = {
            "subset": subset,
            "auc": auc(real_all_scores, fake_scores),
            "real_mean": float(real_all_scores.mean()),
            "fake_mean": float(fake_scores.mean()),
            "margin": float(fake_scores.min() - real_all_scores.max()),
        }
        print("ALL", subset, item)
        all_results.append(item)

    summary = {
        "model_name": args.model_name,
        "image_size": args.image_size,
        "loso_mean_auc": float(np.mean([r["auc"] for r in loso_results])),
        "all_train_mean_auc": float(np.mean([r["auc"] for r in all_results])),
        "loso_results": loso_results,
        "all_train_results": all_results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("LOSO_MEAN_AUC", summary["loso_mean_auc"])
    print("ALL_TRAIN_MEAN_AUC", summary["all_train_mean_auc"])


if __name__ == "__main__":
    main()
