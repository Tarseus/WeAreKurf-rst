import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score


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
    parser.add_argument("--cache-dir", default="dino_dfgc21_cache")
    parser.add_argument("--probe", default="dino_dfgc21_cache/dino_dfgc21_all_probe.joblib")
    parser.add_argument("--out", default="dfgc21_dino_all_scores.json")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    probe = joblib.load(args.probe)
    scaler = probe["scaler"]
    clf = probe["classifier"]

    real_x = np.load(cache_dir / "real_fulls_features.npy")
    real_names = json.loads((cache_dir / "real_fulls_names.json").read_text(encoding="utf-8"))
    real_scores = predict(scaler, clf, real_x)

    subsets = {}
    metrics = []
    for feat_path in sorted(cache_dir.glob("*_features.npy")):
        subset = feat_path.name[:-len("_features.npy")]
        if subset == "real_fulls":
            continue
        x = np.load(feat_path)
        names = json.loads((cache_dir / f"{subset}_names.json").read_text(encoding="utf-8"))
        scores = predict(scaler, clf, x)
        subsets[subset] = {
            "names": names,
            "scores": [float(v) for v in scores],
        }
        item = {
            "subset": subset,
            "auc": auc(real_scores, scores),
            "fake_mean": float(scores.mean()),
        }
        print(item)
        metrics.append(item)

    result = {
        "real": {
            "names": real_names,
            "scores": [float(v) for v in real_scores],
        },
        "subsets": subsets,
        "metrics": metrics,
        "mean_auc": float(np.mean([m["auc"] for m in metrics])),
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("MEAN_AUC", result["mean_auc"])


if __name__ == "__main__":
    main()
