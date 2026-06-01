import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def load_scores(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def align_scores(names_a, scores_a, names_b, scores_b):
    map_b = {name: score for name, score in zip(names_b, scores_b)}
    return np.asarray(scores_a, dtype=np.float64), np.asarray([map_b[name] for name in names_a], dtype=np.float64)


def auc(real_scores, fake_scores):
    labels = np.concatenate([
        np.zeros(len(real_scores), dtype=np.int64),
        np.ones(len(fake_scores), dtype=np.int64),
    ])
    scores = np.concatenate([real_scores, fake_scores])
    return float(roc_auc_score(labels, scores))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fuse(a, b, alpha, mode):
    if mode == "prob":
        return alpha * a + (1.0 - alpha) * b
    if mode == "logit":
        return sigmoid(alpha * logit(a) + (1.0 - alpha) * logit(b))
    raise ValueError(mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eff", default="dfgc21_strong_scores.json")
    parser.add_argument("--dino", default="dfgc21_dino_all_scores.json")
    parser.add_argument("--out", default="dfgc21_fusion_search.json")
    args = parser.parse_args()

    eff = load_scores(args.eff)
    dino = load_scores(args.dino)

    eff_real, dino_real = align_scores(
        eff["real"]["names"],
        eff["real"]["scores"],
        dino["real"]["names"],
        dino["real"]["scores"],
    )

    best = None
    all_rows = []
    for mode in ["prob", "logit"]:
        for alpha in np.linspace(0.0, 1.0, 101):
            rows = []
            for subset, eff_item in eff["subsets"].items():
                dino_item = dino["subsets"][subset]
                eff_fake, dino_fake = align_scores(
                    eff_item["names"],
                    eff_item["scores"],
                    dino_item["names"],
                    dino_item["scores"],
                )
                real_scores = fuse(eff_real, dino_real, alpha, mode)
                fake_scores = fuse(eff_fake, dino_fake, alpha, mode)
                rows.append({
                    "subset": subset,
                    "auc": auc(real_scores, fake_scores),
                    "fake_mean": float(fake_scores.mean()),
                })
            mean_auc = float(np.mean([r["auc"] for r in rows]))
            item = {"mode": mode, "alpha_eff": float(alpha), "mean_auc": mean_auc, "results": rows}
            all_rows.append({"mode": mode, "alpha_eff": float(alpha), "mean_auc": mean_auc})
            if best is None or mean_auc > best["mean_auc"]:
                best = item

    Path(args.out).write_text(json.dumps({"best": best, "grid": all_rows}, indent=2), encoding="utf-8")
    print("BEST", best["mode"], best["alpha_eff"], best["mean_auc"])
    for row in sorted(best["results"], key=lambda x: x["auc"]):
        print(f"{row['subset']:<22} {row['auc']:.6f} fake_mean={row['fake_mean']:.4f}")


if __name__ == "__main__":
    main()
