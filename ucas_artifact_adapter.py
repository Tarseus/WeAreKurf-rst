import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from ucas_val_tune import auc_score, logit_np, official_face_info, read_labels, sigmoid_np


def artifact_features_from_face(face):
    image = face.convert("RGB").resize((160, 160))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)

    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    grad = np.sqrt(dx[:-1, :] ** 2 + dy[:, :-1] ** 2)
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )

    small = gray[::2, ::2]
    fft = np.fft.fftshift(np.fft.fft2(small))
    mag = np.log1p(np.abs(fft))
    h, w = mag.shape
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - h * 0.5) ** 2 + (xx - w * 0.5) ** 2)
    high = mag[rr > min(h, w) * 0.28]
    low = mag[rr <= min(h, w) * 0.18]

    boundary_cols = np.arange(8, gray.shape[1], 8)
    boundary_rows = np.arange(8, gray.shape[0], 8)
    v_block = np.abs(gray[:, boundary_cols] - gray[:, boundary_cols - 1]).mean() if len(boundary_cols) else 0.0
    h_block = np.abs(gray[boundary_rows, :] - gray[boundary_rows - 1, :]).mean() if len(boundary_rows) else 0.0
    v_non = np.abs(np.diff(gray, axis=1)).mean()
    h_non = np.abs(np.diff(gray, axis=0)).mean()

    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    sat = (maxc - minc) / np.maximum(maxc, 1e-6)
    rg = arr[:, :, 0] - arr[:, :, 1]
    bg = arr[:, :, 2] - arr[:, :, 1]

    patch_means = []
    patch_stds = []
    patch_grads = []
    for y in range(0, gray.shape[0], 40):
        for x in range(0, gray.shape[1], 40):
            patch = gray[y:y + 40, x:x + 40]
            patch_means.append(float(patch.mean()))
            patch_stds.append(float(patch.std()))
            if patch.shape[0] > 1 and patch.shape[1] > 1:
                pdx = np.diff(patch, axis=1)
                pdy = np.diff(patch, axis=0)
                patch_grads.append(float(np.sqrt(pdx[:-1, :] ** 2 + pdy[:, :-1] ** 2).mean()))
            else:
                patch_grads.append(0.0)
    patch_means = np.asarray(patch_means, dtype=np.float32)
    patch_stds = np.asarray(patch_stds, dtype=np.float32)
    patch_grads = np.asarray(patch_grads, dtype=np.float32)

    feats = [
        gray.mean(), gray.std(), gray.min(), gray.max(),
        grad.mean(), grad.std(), np.percentile(grad, 90), np.percentile(grad, 99),
        np.abs(lap).mean(), lap.std(), np.percentile(np.abs(lap), 90),
        high.mean(), low.mean(), high.mean() / (low.mean() + 1e-6),
        v_block, h_block, v_block / (v_non + 1e-6), h_block / (h_non + 1e-6),
        sat.mean(), sat.std(), np.percentile(sat, 90),
        rg.mean(), rg.std(), bg.mean(), bg.std(),
        patch_means.std(), patch_stds.mean(), patch_stds.std(),
        patch_grads.mean(), patch_grads.std(), patch_grads.max() - patch_grads.min(),
    ]
    return np.asarray(feats, dtype=np.float32)


def load_submission_model(submission_dir):
    submission_dir = str(Path(submission_dir).resolve())
    if submission_dir not in sys.path:
        sys.path.insert(0, submission_dir)
    if "model" in sys.modules:
        del sys.modules["model"]
    return importlib.import_module("model")


def extract_feature_table(data_folder, submission_dir, tta, batch_size, num_workers, cache_path=None):
    data_folder = Path(data_folder)
    imgs_dir = data_folder / "imgs"
    img_list, face_info = official_face_info(data_folder)

    if cache_path and Path(cache_path).is_file():
        data = np.load(cache_path, allow_pickle=True)
        return {
            "img_names": data["img_names"].astype(str).tolist(),
            "eff_views": data["eff_views"],
            "dino_scores": data["dino_scores"],
            "artifacts": data["artifacts"],
            "official_img_list": img_list,
        }

    model_module = load_submission_model(submission_dir)
    dataset = model_module.FolderDataset(str(imgs_dir), face_info, tta_mode=tta)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if os.name != "nt" else 0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eff_model = model_module.TransferModel("efficientnet-b3", num_out_classes=3)
    eff_path = Path(submission_dir) / "efn-b3_3c_60_acc0.9975.pth"
    eff_model.load_state_dict(torch.load(eff_path, map_location="cpu"))
    eff_model.to(device).eval()

    dino_path = Path(submission_dir) / "dino_dfgc21_probe_ts.pt"
    with open(dino_path, "rb") as f:
        dino_model = torch.jit.load(f, map_location=device)
    dino_model.eval()

    softmax = nn.Softmax(dim=1)
    eff_views = []
    dino_scores = []
    with torch.no_grad():
        for eff_batch, dino_batch in loader:
            bs, views, channels, height, width = eff_batch.shape
            eff_imgs = eff_batch.reshape(bs * views, channels, height, width).to(device)
            eff_outputs = softmax(eff_model(eff_imgs))
            eff_probs = 1.0 - eff_outputs[:, 0]
            eff_views.append(eff_probs.reshape(bs, views).cpu().numpy())
            dino_scores.append(dino_model(dino_batch.to(device)).reshape(-1).cpu().numpy())
    eff_views = np.concatenate(eff_views, axis=0).astype(np.float32)
    dino_scores = np.concatenate(dino_scores, axis=0).astype(np.float32)

    artifacts = []
    start = time.time()
    for i, name in enumerate(dataset.img_names):
        with Image.open(imgs_dir / name) as image:
            face = model_module._crop_face(image.convert("RGB"), name, face_info, 1.3)
            artifacts.append(artifact_features_from_face(face))
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start
            print(f"artifact features {i + 1}/{len(dataset.img_names)} elapsed={elapsed:.1f}s", flush=True)
    artifacts = np.stack(artifacts, axis=0).astype(np.float32)

    if cache_path:
        np.savez_compressed(
            cache_path,
            img_names=np.asarray(dataset.img_names),
            eff_views=eff_views,
            dino_scores=dino_scores,
            artifacts=artifacts,
        )

    return {
        "img_names": list(dataset.img_names),
        "eff_views": eff_views,
        "dino_scores": dino_scores,
        "artifacts": artifacts,
        "official_img_list": img_list,
    }


def align_table(table):
    by_name = {
        name: (table["eff_views"][idx], table["dino_scores"][idx], table["artifacts"][idx])
        for idx, name in enumerate(table["img_names"])
    }
    eff_views, dino_scores, artifacts = [], [], []
    for name in table["official_img_list"]:
        if name not in by_name:
            raise KeyError(f"Missing extracted features for {name}")
        e, d, a = by_name[name]
        eff_views.append(e)
        dino_scores.append(d)
        artifacts.append(a)
    return np.stack(eff_views), np.asarray(dino_scores), np.stack(artifacts)


def build_features(eff_views, dino_scores, artifacts, feature_set):
    eff_mean = eff_views.mean(axis=1)
    eff_std = eff_views.std(axis=1)
    eff_min = eff_views.min(axis=1)
    eff_max = eff_views.max(axis=1)
    eff_range = eff_max - eff_min
    base = np.column_stack([
        eff_mean,
        eff_std,
        eff_min,
        eff_max,
        eff_range,
        dino_scores,
        logit_np(eff_mean),
        logit_np(dino_scores),
        sigmoid_np(0.55 * logit_np(eff_mean) + 0.45 * logit_np(dino_scores)),
    ])
    if feature_set == "score_distribution":
        return np.column_stack([base, eff_views])
    if feature_set == "artifact_only":
        return artifacts
    if feature_set == "artifact_adapter":
        return np.column_stack([base, eff_views, artifacts])
    raise ValueError(feature_set)


class TinyMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def standardize_fit(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(X, mean, std):
    return ((X - mean) / std).astype(np.float32)


def train_mlp(X, y, epochs=220, seed=42):
    torch.manual_seed(seed)
    mean, std = standardize_fit(X)
    Xs = standardize_apply(X, mean, std)
    x = torch.from_numpy(Xs)
    target = torch.from_numpy(y.astype(np.float32))
    model = TinyMLP(X.shape[1])
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-2)
    model.train()
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    model.eval()
    return {"kind": "mlp", "mean": mean, "std": std, "state_dict": model.state_dict(), "input_dim": X.shape[1]}


def predict_mlp(bundle, X):
    model = TinyMLP(bundle["input_dim"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    Xs = standardize_apply(X, bundle["mean"], bundle["std"])
    with torch.no_grad():
        logits = model(torch.from_numpy(Xs)).numpy()
    return sigmoid_np(logits)


def train_centroid(X, y):
    mean, std = standardize_fit(X)
    Xs = standardize_apply(X, mean, std)
    real = Xs[y == 0].mean(axis=0)
    fake = Xs[y == 1].mean(axis=0)
    return {"kind": "centroid", "mean": mean, "std": std, "real": real.astype(np.float32), "fake": fake.astype(np.float32)}


def predict_centroid(bundle, X):
    Xs = standardize_apply(X, bundle["mean"], bundle["std"])
    d_real = ((Xs - bundle["real"]) ** 2).mean(axis=1)
    d_fake = ((Xs - bundle["fake"]) ** 2).mean(axis=1)
    return sigmoid_np(d_real - d_fake)


def stratified_folds(y, n_splits=5, seed=42):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_splits)]
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, item in enumerate(idx):
            folds[i % n_splits].append(int(item))
    return [np.asarray(sorted(fold), dtype=np.int64) for fold in folds]


def oof_eval(X, y, method, n_splits=5):
    pred = np.zeros(len(y), dtype=np.float64)
    folds = stratified_folds(y, n_splits=n_splits)
    all_idx = np.arange(len(y))
    for fold_id, val_idx in enumerate(folds):
        train_idx = np.setdiff1d(all_idx, val_idx)
        if method == "mlp":
            bundle = train_mlp(X[train_idx], y[train_idx], seed=100 + fold_id)
            pred[val_idx] = predict_mlp(bundle, X[val_idx])
        elif method == "centroid":
            bundle = train_centroid(X[train_idx], y[train_idx])
            pred[val_idx] = predict_centroid(bundle, X[val_idx])
        else:
            raise ValueError(method)
    return auc_score(y, pred), pred


def save_torch_bundle(path, bundle, metadata):
    payload = {
        "metadata": metadata,
        "kind": bundle["kind"],
        "mean": torch.from_numpy(bundle["mean"]),
        "std": torch.from_numpy(bundle["std"]),
    }
    if bundle["kind"] == "mlp":
        payload["input_dim"] = int(bundle["input_dim"])
        payload["state_dict"] = bundle["state_dict"]
    elif bundle["kind"] == "centroid":
        payload["real"] = torch.from_numpy(bundle["real"])
        payload["fake"] = torch.from_numpy(bundle["fake"])
    else:
        raise ValueError(bundle["kind"])
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--submission-dir", default="submission_det_ensemble")
    parser.add_argument("--out-dir", default="ucas_artifact_adapter_output")
    parser.add_argument("--tta", default=os.environ.get("DFGC_TTA", "strong"), choices=["none", "fast", "strong"])
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("DFGC_BATCH_SIZE", "4")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("DFGC_NUM_WORKERS", "4")))
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "feature_cache.npz"

    table = extract_feature_table(
        args.data_folder,
        args.submission_dir,
        args.tta,
        args.batch_size,
        args.num_workers,
        cache_path=cache_path,
    )
    eff_views, dino_scores, artifacts = align_table(table)
    labels, label_file = read_labels(args.data_folder, len(table["official_img_list"]))

    eff_mean = eff_views.mean(axis=1)
    default_fusion = sigmoid_np(0.55 * logit_np(eff_mean) + 0.45 * logit_np(dino_scores))
    results = [
        {"model": "efficientnet_b3", "auc": auc_score(labels, eff_mean), "type": "locked_baseline"},
        {"model": "default_eff_dino_fusion", "auc": auc_score(labels, default_fusion), "type": "initial_baseline"},
        {"model": "dino_probe", "auc": auc_score(labels, dino_scores), "type": "initial_baseline"},
    ]

    best = {"auc": -1.0}
    oof_predictions = {}
    for feature_set in ["artifact_adapter", "artifact_only", "score_distribution"]:
        X = build_features(eff_views, dino_scores, artifacts, feature_set)
        for method in ["mlp", "centroid"]:
            auc, pred = oof_eval(X, labels, method, n_splits=args.n_splits)
            item = {
                "model": f"{feature_set}_{method}",
                "feature_set": feature_set,
                "method": method,
                "auc": auc,
                "type": "novel_candidate_oof",
            }
            results.append(item)
            oof_predictions[item["model"]] = pred
            if auc > best.get("auc", -1.0):
                best = item

    X_best = build_features(eff_views, dino_scores, artifacts, best["feature_set"])
    if best["method"] == "mlp":
        final_bundle = train_mlp(X_best, labels, seed=777)
        train_scores = predict_mlp(final_bundle, X_best)
    else:
        final_bundle = train_centroid(X_best, labels)
        train_scores = predict_centroid(final_bundle, X_best)

    best["full_val_auc_after_refit"] = auc_score(labels, train_scores)
    metadata = {
        "data_folder": str(args.data_folder),
        "label_file": label_file,
        "tta": args.tta,
        "batch_size": args.batch_size,
        "n_splits": args.n_splits,
        "best": best,
    }
    adapter_path = out_dir / "best_ucas_artifact_adapter.pt"
    save_torch_bundle(adapter_path, final_bundle, metadata)

    summary = {
        "num_images": len(labels),
        "label_file": label_file,
        "results": sorted(results, key=lambda x: x["auc"], reverse=True),
        "best_novel_model": best,
        "adapter_path": str(adapter_path),
        "baseline_auc": results[0]["auc"],
        "default_fusion_auc": results[1]["auc"],
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "oof_predictions.npz",
        labels=labels,
        efficientnet=eff_mean,
        dino=dino_scores,
        default_fusion=default_fusion,
        **{name: pred for name, pred in oof_predictions.items()},
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
