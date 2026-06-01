import argparse
import importlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def ensure_extracted(zip_path, out_dir, expected=1000):
    out_dir = Path(out_dir)
    image_count = len([p for p in out_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if image_count == expected:
        return
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)
    image_count = len([p for p in tmp_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if image_count != expected:
        raise RuntimeError(f"{zip_path} extracted {image_count} images, expected {expected}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)


def auc(real_scores, fake_scores):
    labels = np.concatenate([
        np.zeros(len(real_scores), dtype=np.int64),
        np.ones(len(fake_scores), dtype=np.int64),
    ])
    scores = np.concatenate([real_scores, fake_scores])
    return float(roc_auc_score(labels, scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default="submission_det_ensemble")
    parser.add_argument("--dfgc-root", default="datasets/DFGC-21")
    parser.add_argument("--extract-root", default="datasets/DFGC-21-extracted")
    parser.add_argument("--out", default="dfgc21_eval_submission_results.json")
    parser.add_argument("--tta", default=os.environ.get("DFGC_TTA", "strong"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("DFGC_BATCH_SIZE", "4")))
    args = parser.parse_args()

    os.environ["DFGC_TTA"] = args.tta
    os.environ["DFGC_BATCH_SIZE"] = str(args.batch_size)
    module = importlib.import_module(f"{args.module}.model")

    dfgc_root = Path(args.dfgc_root)
    extract_root = Path(args.extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    json_file = dfgc_root / "bbox&landmarks.json"
    real_zip = dfgc_root / "real_fulls.zip"
    fake_zips = sorted(p for p in dfgc_root.glob("*.zip") if p.name != "real_fulls.zip")

    for zip_path in [real_zip] + fake_zips:
        ensure_extracted(zip_path, extract_root / zip_path.stem)

    detector = module.Model()
    real_names, real_scores = detector.run(str(extract_root / "real_fulls"), str(json_file))
    real_scores = np.asarray(real_scores, dtype=np.float32)

    results = []
    for fake_zip in fake_zips:
        subset = fake_zip.stem
        start = time.time()
        _, fake_scores = detector.run(str(extract_root / subset), str(json_file))
        fake_scores = np.asarray(fake_scores, dtype=np.float32)
        item = {
            "subset": subset,
            "auc": auc(real_scores, fake_scores),
            "real_mean": float(real_scores.mean()),
            "fake_mean": float(fake_scores.mean()),
            "margin": float(fake_scores.min() - real_scores.max()),
            "seconds": float(time.time() - start),
        }
        print(
            f"{subset}: AUC={item['auc']:.6f} fake_mean={item['fake_mean']:.4f} "
            f"margin={item['margin']:.4f} seconds={item['seconds']:.1f}"
        )
        results.append(item)

    summary = {
        "module": args.module,
        "tta": args.tta,
        "batch_size": args.batch_size,
        "mean_auc": float(np.mean([r["auc"] for r in results])),
        "results": results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("MEAN_AUC", summary["mean_auc"])


if __name__ == "__main__":
    main()
