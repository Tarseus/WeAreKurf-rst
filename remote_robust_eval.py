import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def binary_auc(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) * 0.5
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    n_pos = sum(1 for label in labels if label == 1)
    n_neg = len(labels) - n_pos
    pos_ranks = sum(rank for rank, (_, label) in zip(ranks, pairs) if label == 1)
    return (pos_ranks - n_pos * (n_pos + 1) * 0.5) / (n_pos * n_neg)


def prepare_cases(src_dir, out_root):
    out_root.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in src_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    cases = {}

    for case in ["original", "jpeg60", "jpeg35", "blur", "downscale", "noise"]:
        case_dir = out_root / case
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True)
        cases[case] = case_dir

    rng = np.random.default_rng(2026)
    for name in names:
        image = Image.open(src_dir / name).convert("RGB")
        image.save(cases["original"] / name)
        image.save(cases["jpeg60"] / name, quality=60)
        image.save(cases["jpeg35"] / name, quality=35)
        image.filter(ImageFilter.GaussianBlur(radius=1.4)).save(cases["blur"] / name)

        small = image.resize((max(8, image.width // 3), max(8, image.height // 3)), Image.Resampling.BILINEAR)
        small.resize(image.size, Image.Resampling.BILINEAR).save(cases["downscale"] / name)

        arr = np.asarray(image).astype(np.float32)
        arr = np.clip(arr + rng.normal(0.0, 6.0, size=arr.shape), 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(cases["noise"] / name)

    return cases


def run_detector(case_name, case_dir, json_file, tta_mode, batch_size):
    os.environ["DFGC_TTA"] = tta_mode
    os.environ["DFGC_BATCH_SIZE"] = str(batch_size)
    from submission_det_strong import model as strong_det

    start = time.time()
    detector = strong_det.Model()
    names, preds = detector.run(str(case_dir), str(json_file))
    elapsed = time.time() - start
    labels = [0] * 5 + [1] * 5
    real_scores = preds[:5]
    fake_scores = preds[5:]
    return {
        "case": case_name,
        "tta": tta_mode,
        "auc": binary_auc(labels, preds),
        "real_mean": float(np.mean(real_scores)),
        "fake_mean": float(np.mean(fake_scores)),
        "margin": float(min(fake_scores) - max(real_scores)),
        "seconds": elapsed,
        "names": names,
        "predictions": [float(p) for p in preds],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default="DFGC_Detection/sample_imgs")
    parser.add_argument("--json-file", default="DFGC_Detection/sample_meta.json")
    parser.add_argument("--out", default="remote_robust_eval_results.json")
    parser.add_argument("--case-root", default="robust_cases")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    src_dir = Path(args.img_dir)
    json_file = Path(args.json_file)
    cases = prepare_cases(src_dir, Path(args.case_root))
    results = []
    for tta_mode in ["none", "fast", "strong"]:
        for case_name, case_dir in cases.items():
            result = run_detector(case_name, case_dir, json_file, tta_mode, args.batch_size)
            results.append(result)
            print(
                f"{tta_mode:6s} {case_name:9s} "
                f"AUC={result['auc']:.4f} margin={result['margin']:.4f} "
                f"real={result['real_mean']:.4f} fake={result['fake_mean']:.4f} "
                f"time={result['seconds']:.2f}s"
            )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
