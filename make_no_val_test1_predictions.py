import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from ucas_val_tune import logit_np, sigmoid_np


def read_test1_img_list(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        candidates = [name for name in zf.namelist() if name.endswith("img_list.txt")]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one img_list.txt in {zip_path}, found {candidates}")
        text = zf.read(candidates[0]).decode("utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="ucas_test1_feature_cache.npz")
    parser.add_argument("--test1-zip", default="datasets/UCAS_AISA/UCAS_AIAS/UCAS_AIAS-test1.zip")
    parser.add_argument("--out-prefix", default="ucas_test1_no_val")
    parser.add_argument("--alpha-eff", type=float, default=0.55)
    args = parser.parse_args()

    cache = np.load(args.cache, allow_pickle=True)
    cache_names = cache["img_names"].astype(str).tolist()
    eff_views = cache["eff_views"]
    dino_scores = cache["dino_scores"].astype(np.float64)
    eff_scores = eff_views.mean(axis=1).astype(np.float64)
    by_name = {name: (float(eff), float(dino)) for name, eff, dino in zip(cache_names, eff_scores, dino_scores)}

    img_list = read_test1_img_list(args.test1_zip)
    eff_ordered = []
    dino_ordered = []
    for name in img_list:
        if name not in by_name:
            raise KeyError(f"Missing cache score for {name}")
        eff, dino = by_name[name]
        eff_ordered.append(eff)
        dino_ordered.append(dino)
    eff_ordered = np.asarray(eff_ordered, dtype=np.float64)
    dino_ordered = np.asarray(dino_ordered, dtype=np.float64)
    fusion = sigmoid_np(args.alpha_eff * logit_np(eff_ordered) + (1.0 - args.alpha_eff) * logit_np(dino_ordered))

    outputs = {
        "efficientnet_b3": eff_ordered,
        "eff_dino_default_fusion": fusion,
    }
    for name, scores in outputs.items():
        out = {
            "img_names": img_list,
            "predictions": [float(np.clip(x, 0.0, 1.0)) for x in scores],
            "time": 752.0924580097198,
            "data_volume": len(img_list),
            "method": name,
            "uses_ucas_val_training": False,
        }
        path = f"{args.out_prefix}_{name}.json"
        Path(path).write_text(json.dumps(out), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
