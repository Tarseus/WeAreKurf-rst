import argparse
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


def as_list(x):
    if isinstance(x, list):
        return x
    return [float(x)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfgc-root", default="datasets/DFGC-21")
    parser.add_argument("--extract-root", default="datasets/DFGC-21-extracted")
    parser.add_argument("--out", default="dfgc21_eval_strong_results.json")
    parser.add_argument("--tta", default=os.environ.get("DFGC_TTA", "strong"))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("DFGC_BATCH_SIZE", "16")))
    args = parser.parse_args()

    os.environ["DFGC_TTA"] = args.tta
    os.environ["DFGC_BATCH_SIZE"] = str(args.batch_size)

    from submission_det_strong import model as strong_det

    dfgc_root = Path(args.dfgc_root)
    extract_root = Path(args.extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    json_file = dfgc_root / "bbox&landmarks.json"

    zip_paths = sorted(dfgc_root.glob("*.zip"))
    real_zip = dfgc_root / "real_fulls.zip"
    fake_zips = [p for p in zip_paths if p.name != "real_fulls.zip"]

    print(f"extracting/checking {1 + len(fake_zips)} subsets...")
    for zip_path in [real_zip] + fake_zips:
        ensure_extracted(zip_path, extract_root / zip_path.stem)

    detector = strong_det.Model()

    print("running real_fulls inference...")
    t0 = time.time()
    real_names, real_scores = detector.run(str(extract_root / "real_fulls"), str(json_file))
    real_scores = np.asarray(as_list(real_scores), dtype=np.float32)
    print(f"real_fulls done: n={len(real_scores)} seconds={time.time() - t0:.1f}")

    results = []
    for fake_zip in fake_zips:
        subset = fake_zip.stem
        print(f"running {subset} inference...")
        start = time.time()
        fake_names, fake_scores = detector.run(str(extract_root / subset), str(json_file))
        fake_scores = np.asarray(as_list(fake_scores), dtype=np.float32)

        labels = np.concatenate([np.zeros_like(real_scores, dtype=np.int64), np.ones_like(fake_scores, dtype=np.int64)])
        scores = np.concatenate([real_scores, fake_scores])
        auc = float(roc_auc_score(labels, scores))
        item = {
            "subset": subset,
            "auc": auc,
            "real_mean": float(real_scores.mean()),
            "fake_mean": float(fake_scores.mean()),
            "real_max": float(real_scores.max()),
            "fake_min": float(fake_scores.min()),
            "margin": float(fake_scores.min() - real_scores.max()),
            "seconds": float(time.time() - start),
            "fake_count": int(len(fake_scores)),
        }
        print(
            f"{subset}: AUC={auc:.6f} real_mean={item['real_mean']:.4f} "
            f"fake_mean={item['fake_mean']:.4f} margin={item['margin']:.4f} "
            f"seconds={item['seconds']:.1f}"
        )
        results.append(item)

    mean_auc = float(np.mean([item["auc"] for item in results]))
    summary = {
        "tta": args.tta,
        "batch_size": args.batch_size,
        "real_count": int(len(real_scores)),
        "mean_auc": mean_auc,
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"MEAN_AUC={mean_auc:.6f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
