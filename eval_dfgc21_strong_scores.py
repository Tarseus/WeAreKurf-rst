import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def image_files(folder):
    return sorted(
        p.name
        for p in Path(folder).iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


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
    parser.add_argument("--out", default="dfgc21_strong_scores.json")
    parser.add_argument("--tta", default=os.environ.get("DFGC_TTA", "strong"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("DFGC_BATCH_SIZE", "16")))
    args = parser.parse_args()

    os.environ["DFGC_TTA"] = args.tta
    os.environ["DFGC_BATCH_SIZE"] = str(args.batch_size)
    from submission_det_strong import model as strong_det

    extract_root = Path(args.extract_root)
    detector = strong_det.Model()

    real_dir = extract_root / "real_fulls"
    real_names, real_scores = detector.run(str(real_dir), args.meta_json)
    real_scores = np.asarray(real_scores, dtype=np.float32)

    subsets = {}
    metrics = []
    for subset_dir in sorted(p for p in extract_root.iterdir() if p.is_dir() and p.name != "real_fulls"):
        fake_names, fake_scores = detector.run(str(subset_dir), args.meta_json)
        fake_scores = np.asarray(fake_scores, dtype=np.float32)
        subsets[subset_dir.name] = {
            "names": fake_names,
            "scores": [float(x) for x in fake_scores],
        }
        item = {
            "subset": subset_dir.name,
            "auc": auc(real_scores, fake_scores),
            "fake_mean": float(fake_scores.mean()),
        }
        print(item)
        metrics.append(item)

    result = {
        "tta": args.tta,
        "batch_size": args.batch_size,
        "real": {
            "names": real_names,
            "scores": [float(x) for x in real_scores],
        },
        "subsets": subsets,
        "mean_auc": float(np.mean([m["auc"] for m in metrics])),
        "metrics": metrics,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("MEAN_AUC", result["mean_auc"])


if __name__ == "__main__":
    main()
