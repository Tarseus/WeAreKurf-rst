import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from dinov3_dfgc_probe_ucas import dfgc_items_and_labels
from ucas_val_tune import logit_np, official_face_info, read_labels, sigmoid_np


def auc(real_scores, fake_scores):
    labels = np.concatenate([
        np.zeros(len(real_scores), dtype=np.int64),
        np.ones(len(fake_scores), dtype=np.int64),
    ])
    scores = np.concatenate([real_scores, fake_scores])
    return float(roc_auc_score(labels, scores))


def mean_dfgc_auc(strong_json, dino_scores, dfgc_items, alpha):
    strong = json.loads(Path(strong_json).read_text(encoding="utf-8"))
    idx = 0
    real_names = [name for _, name in dfgc_items[:len(strong["real"]["names"])]]
    if real_names != strong["real"]["names"]:
        raise RuntimeError("Real image order mismatch")
    real_eff = np.asarray(strong["real"]["scores"], dtype=np.float64)
    real_dino = dino_scores[:len(real_eff)]
    idx += len(real_eff)

    results = []
    for subset in sorted(strong["subsets"]):
        entry = strong["subsets"][subset]
        n = len(entry["names"])
        names = [name for _, name in dfgc_items[idx:idx + n]]
        if names != entry["names"]:
            raise RuntimeError(f"Image order mismatch for {subset}")
        fake_eff = np.asarray(entry["scores"], dtype=np.float64)
        fake_dino = dino_scores[idx:idx + n]
        idx += n
        real_fused = sigmoid_np(alpha * logit_np(real_eff) + (1.0 - alpha) * logit_np(real_dino))
        fake_fused = sigmoid_np(alpha * logit_np(fake_eff) + (1.0 - alpha) * logit_np(fake_dino))
        results.append({"subset": subset, "auc": auc(real_fused, fake_fused)})
    return float(np.mean([r["auc"] for r in results])), results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dfgc-root", default="datasets/DFGC-21-extracted")
    parser.add_argument("--dfgc-json", default="datasets/DFGC-21/bbox&landmarks.json")
    parser.add_argument("--strong-json", default="dfgc21_strong_scores.json")
    parser.add_argument("--ucas-val", default="datasets/UCAS_AISA/extracted/val")
    parser.add_argument("--eff-val-cache", default="ucas_artifact_adapter_output/feature_cache.npz")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    features = np.load(out_dir / "dfgc21_features.npz")["features"]
    probe = joblib.load(out_dir / "dinov3_dfgc21_probe.joblib")
    dfgc_scores = probe.predict_proba(features)[:, 1].astype(np.float64)
    dfgc_items, labels, _ = dfgc_items_and_labels(args.dfgc_root, args.dfgc_json)

    search = []
    for alpha in np.linspace(0.0, 1.0, 101):
        mean_auc, results = mean_dfgc_auc(args.strong_json, dfgc_scores, dfgc_items, float(alpha))
        search.append({"alpha_eff": float(alpha), "mean_auc": mean_auc, "results": results})
    best = max(search, key=lambda x: x["mean_auc"])

    val_features = np.load(out_dir / "ucas_val_features.npz")["features"]
    val_dino = probe.predict_proba(val_features)[:, 1].astype(np.float64)
    val_img_list, _ = official_face_info(args.ucas_val)
    val_labels, label_file = read_labels(args.ucas_val, len(val_img_list))
    eff_cache = np.load(args.eff_val_cache, allow_pickle=True)
    eff_names = eff_cache["img_names"].astype(str).tolist()
    eff_scores = eff_cache["eff_views"].mean(axis=1).astype(np.float64)
    eff_by_name = {name: score for name, score in zip(eff_names, eff_scores)}
    eff_ordered = np.asarray([eff_by_name[name] for name in val_img_list], dtype=np.float64)
    fixed = sigmoid_np(0.55 * logit_np(eff_ordered) + 0.45 * logit_np(val_dino))
    chosen = sigmoid_np(best["alpha_eff"] * logit_np(eff_ordered) + (1.0 - best["alpha_eff"]) * logit_np(val_dino))

    summary = {
        "selection_data": "DFGC-21 only; UCAS val labels used only for final evaluation",
        "out_dir": str(out_dir),
        "best_dfgc_alpha_eff": best["alpha_eff"],
        "best_dfgc_mean_auc": best["mean_auc"],
        "ucas_label_file": label_file,
        "ucas_auc_dino": float(roc_auc_score(val_labels, val_dino)),
        "ucas_auc_eff_fixed_alpha_0.55": float(roc_auc_score(val_labels, fixed)),
        "ucas_auc_eff_dfgc_selected_alpha": float(roc_auc_score(val_labels, chosen)),
        "search": search,
    }
    out_path = Path(args.out) if args.out else out_dir / "dfgc_alpha_search_results.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "search"}, indent=2))


if __name__ == "__main__":
    main()
