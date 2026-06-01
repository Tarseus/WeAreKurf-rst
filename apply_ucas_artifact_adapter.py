import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ucas_artifact_adapter import (
    TinyMLP,
    build_features,
    extract_feature_table,
    predict_centroid,
    sigmoid_np,
)


def predict_mlp_payload(payload, X):
    mean = payload["mean"].cpu().numpy()
    std = payload["std"].cpu().numpy()
    Xs = ((X - mean) / std).astype(np.float32)
    model = TinyMLP(int(payload["input_dim"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xs)).numpy()
    return sigmoid_np(logits)


def predict_centroid_payload(payload, X):
    bundle = {
        "mean": payload["mean"].cpu().numpy(),
        "std": payload["std"].cpu().numpy(),
        "real": payload["real"].cpu().numpy(),
        "fake": payload["fake"].cpu().numpy(),
    }
    return predict_centroid(bundle, X)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--submission-dir", default="submission_det_ensemble")
    parser.add_argument("--adapter", default="ucas_artifact_adapter_output/best_ucas_artifact_adapter.pt")
    parser.add_argument("--out", default="ucas_test1_artifact_adapter_predictions.json")
    parser.add_argument("--cache", default="ucas_artifact_adapter_output/test1_feature_cache.npz")
    parser.add_argument("--tta", default="strong", choices=["none", "fast", "strong"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    start = time.time()
    payload = torch.load(args.adapter, map_location="cpu")
    metadata = payload.get("metadata", {})
    best = metadata.get("best", {})
    feature_set = best.get("feature_set", "artifact_adapter")

    table = extract_feature_table(
        args.data_folder,
        args.submission_dir,
        args.tta,
        args.batch_size,
        args.num_workers,
        cache_path=args.cache,
    )
    eff_views, dino_scores, artifacts = [], [], []
    by_name = {
        name: (table["eff_views"][idx], table["dino_scores"][idx], table["artifacts"][idx])
        for idx, name in enumerate(table["img_names"])
    }
    for name in table["official_img_list"]:
        e, d, a = by_name[name]
        eff_views.append(e)
        dino_scores.append(d)
        artifacts.append(a)
    eff_views = np.stack(eff_views)
    dino_scores = np.asarray(dino_scores)
    artifacts = np.stack(artifacts)
    X = build_features(eff_views, dino_scores, artifacts, feature_set)

    if payload["kind"] == "mlp":
        predictions = predict_mlp_payload(payload, X)
    elif payload["kind"] == "centroid":
        predictions = predict_centroid_payload(payload, X)
    else:
        raise ValueError(payload["kind"])

    elapsed = time.time() - start
    out = {
        "img_names": table["official_img_list"],
        "predictions": [float(np.clip(x, 0.0, 1.0)) for x in predictions],
        "time": elapsed,
        "data_volume": len(predictions),
        "adapter": args.adapter,
        "feature_set": feature_set,
        "kind": payload["kind"],
    }
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"Images: {len(predictions)}")
    print(f"Seconds: {elapsed:.3f}")


if __name__ == "__main__":
    main()
