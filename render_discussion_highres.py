import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw


ARTIFACT_NAMES = [
    "gray_mean", "gray_std", "gray_min", "gray_max",
    "grad_mean", "grad_std", "grad_p90", "grad_p99",
    "lap_abs_mean", "lap_std", "lap_abs_p90",
    "fft_high_mean", "fft_low_mean", "fft_high_low_ratio",
    "jpeg_v_block", "jpeg_h_block", "jpeg_v_block_ratio", "jpeg_h_block_ratio",
    "sat_mean", "sat_std", "sat_p90",
    "rg_mean", "rg_std", "bg_mean", "bg_std",
    "patch_mean_std", "patch_std_mean", "patch_std_std",
    "patch_grad_mean", "patch_grad_std", "patch_grad_range",
]


class SAE(nn.Module):
    def __init__(self, dim, latents, topk=0):
        super().__init__()
        self.topk = int(topk)
        self.encoder = nn.Linear(dim, latents)
        self.decoder = nn.Linear(latents, dim)

    def forward(self, x):
        h = torch.relu(self.encoder(x))
        if self.topk > 0 and self.topk < h.shape[1]:
            vals, idx = torch.topk(h, self.topk, dim=1)
            sparse = torch.zeros_like(h)
            sparse.scatter_(1, idx, vals)
            h = sparse
        recon = self.decoder(h)
        return recon, h


def logit_np(x, eps=1e-6):
    x = np.clip(np.asarray(x, dtype=np.float64), eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def rankdata(x):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return corr(rankdata(x), rankdata(y))


def read_face_boxes(data_folder):
    data_folder = Path(data_folder)
    names = (data_folder / "img_list.txt").read_text(encoding="utf-8").splitlines()
    lines = (data_folder / "face_info.txt").read_text(encoding="utf-8").splitlines()
    boxes = {}
    for name, line in zip(names, lines):
        nums = [float(v) for v in line.replace(",", " ").split()[:4]]
        boxes[name] = nums if len(nums) == 4 else None
    return boxes


def face_crop(image, box, scale=1.75):
    if box is None:
        return image
    w, h = image.size
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * scale
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    left = max(int(round(cx - side / 2.0)), 0)
    top = max(int(round(cy - side / 2.0)), 0)
    right = min(int(round(cx + side / 2.0)), w)
    bottom = min(int(round(cy + side / 2.0)), h)
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def cell_image(path, title, subtitle, box=None, cell=360):
    image = Image.open(path).convert("RGB")
    image = face_crop(image, box)
    image.thumbnail((cell, cell), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (cell, cell + 82), (255, 255, 255))
    canvas.paste(image, ((cell - image.width) // 2, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, cell - 1, cell + 81), outline=(210, 210, 210), width=2)
    draw.text((8, cell + 8), title[:58], fill=(0, 0, 0))
    draw.text((8, cell + 40), subtitle[:70], fill=(40, 40, 40))
    return canvas


def save_grid(data_folder, boxes, names, labels, eff, dino, out_path, cell=360, max_items=16):
    data_folder = Path(data_folder)
    imgs_dir = data_folder / "imgs"
    cells = []
    for name in names[:max_items]:
        i = name_to_idx[str(name)]
        title = f"{name} | y={int(labels[i])}"
        subtitle = f"Eff={eff[i]:.4f}, DINO={dino[i]:.4f}, delta={dino[i]-eff[i]:+.4f}"
        cells.append(cell_image(imgs_dir / str(name), title, subtitle, boxes.get(str(name)), cell=cell))
    if not cells:
        return
    cols = 4
    rows = math.ceil(len(cells) / cols)
    w, h = cells[0].size
    grid = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for j, cell_img in enumerate(cells):
        grid.paste(cell_img, ((j % cols) * w, (j // cols) * h))
    grid.save(out_path, quality=95)


def save_latent_grid(data_folder, boxes, latent_id, h, img_names, labels, eff, dino, out_path, cell=360, max_items=16):
    order = np.argsort(h[:, latent_id])[::-1]
    names = img_names[order[:max_items]]
    data_folder = Path(data_folder)
    imgs_dir = data_folder / "imgs"
    cells = []
    for name in names:
        i = name_to_idx[str(name)]
        title = f"{name} | y={int(labels[i])} | L{latent_id}={h[i, latent_id]:.3f}"
        subtitle = f"Eff={eff[i]:.4f}, DINO={dino[i]:.4f}"
        cells.append(cell_image(imgs_dir / str(name), title, subtitle, boxes.get(str(name)), cell=cell))
    cols = 4
    rows = math.ceil(len(cells) / cols)
    w, hh = cells[0].size
    grid = Image.new("RGB", (cols * w, rows * hh), (255, 255, 255))
    for j, cell_img in enumerate(cells):
        grid.paste(cell_img, ((j % cols) * w, (j // cols) * hh))
    grid.save(out_path, quality=95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucas-val", default="datasets/UCAS_AISA/extracted/val")
    parser.add_argument("--source-dir", default="discussion_vfm_probe_topk128")
    parser.add_argument("--out-dir", default="discussion_vfm_probe_topk128_highres")
    parser.add_argument("--scores", default="dinov3_splus_dfgc_probe_output/ucas_val_scores.npz")
    parser.add_argument("--features", default="dinov3_splus_dfgc_probe_output/ucas_val_features.npz")
    parser.add_argument("--probe", default="dinov3_splus_dfgc_probe_output/dinov3_dfgc21_probe.joblib")
    parser.add_argument("--artifact-cache", default="ucas_artifact_adapter_output/feature_cache.npz")
    parser.add_argument("--sae-latents", type=int, default=2048)
    parser.add_argument("--sae-topk", type=int, default=128)
    parser.add_argument("--cell", type=int, default=420)
    parser.add_argument("--dpi", type=int, default=420)
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.source_dir)
    results = json.loads((source / "discussion_results.json").read_text(encoding="utf-8"))

    scores_npz = np.load(args.scores, allow_pickle=True)
    img_names = scores_npz["img_names"].astype(str)
    labels = scores_npz["labels"].astype(np.int64)
    eff_scores = scores_npz["eff_scores"].astype(np.float64)
    dino_scores = scores_npz["dinov3_scores"].astype(np.float64)
    fusion_scores = sigmoid_np(0.61 * logit_np(eff_scores) + 0.39 * logit_np(dino_scores))
    global name_to_idx
    name_to_idx = {name: i for i, name in enumerate(img_names)}

    features = np.load(args.features)["features"].astype(np.float32)
    probe = joblib.load(args.probe)
    scaler = probe.steps[0][1]
    val_z = scaler.transform(features).astype(np.float32)
    sae = SAE(val_z.shape[1], args.sae_latents, args.sae_topk)
    state = torch.load(source / "sae_dinov3_splus_linear_probe.pt", map_location="cpu")
    sae.load_state_dict(state)
    sae.eval()
    with torch.no_grad():
        _, h = sae(torch.from_numpy(val_z))
    h = h.numpy()

    boxes = read_face_boxes(args.ucas_val)

    metrics = {
        "EfficientNet-B3": auc(labels, eff_scores),
        "DINOv3-S+ Linear": auc(labels, dino_scores),
        "Fusion alpha=0.61": auc(labels, fusion_scores),
    }
    plt.figure(figsize=(9.5, 5.2))
    bars = plt.bar(metrics.keys(), metrics.values(), color=["#4E6475", "#298C5B", "#C65A3A"])
    plt.ylim(0.93, 1.0)
    plt.ylabel("UCAS val AUC")
    plt.xticks(rotation=10, ha="right")
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001, f"{b.get_height():.6f}", ha="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "auc_bars_highres.png", dpi=args.dpi)
    plt.close()

    eff_logit = logit_np(eff_scores)
    dino_logit = logit_np(dino_scores)
    plt.figure(figsize=(7.8, 7.0))
    plt.scatter(eff_logit[labels == 0], dino_logit[labels == 0], s=7, alpha=0.24, label="real", c="#1f77b4")
    plt.scatter(eff_logit[labels == 1], dino_logit[labels == 1], s=7, alpha=0.24, label="fake", c="#d62728")
    plt.xlabel("EfficientNet logit")
    plt.ylabel("DINOv3-S+ linear logit")
    plt.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(out_dir / "score_scatter_highres.png", dpi=args.dpi)
    plt.close()

    art = results["top_artifact_correlations"][:16]
    plt.figure(figsize=(10.5, 6.0))
    vals = [r["spearman_with_dino_minus_eff_margin"] for r in art]
    names = [r["feature"] for r in art]
    colors = ["#C65A3A" if v > 0 else "#4E6475" for v in vals]
    plt.barh(names[::-1], vals[::-1], color=colors[::-1])
    plt.xlabel("Spearman corr. with DINO margin advantage")
    plt.tight_layout()
    plt.savefig(out_dir / "artifact_delta_corr_highres.png", dpi=args.dpi)
    plt.close()

    sae_rows = results["top_sae_latents"][:24]
    plt.figure(figsize=(10.5, 7.0))
    vals = [r["mean_logit_contribution_fake_minus_real"] for r in sae_rows]
    names = [f"L{r['latent']}" for r in sae_rows]
    colors = ["#C65A3A" if v > 0 else "#4E6475" for v in vals]
    plt.barh(names[::-1], vals[::-1], color=colors[::-1])
    plt.xlabel("SAE latent contribution: fake mean - real mean")
    plt.tight_layout()
    plt.savefig(out_dir / "sae_top_latents_highres.png", dpi=args.dpi)
    plt.close()

    ab = results["sae_ablation"]
    plt.figure(figsize=(8.8, 5.5))
    plt.plot([r["k"] for r in ab], [r["remove_top_k_auc"] for r in ab], marker="o", label="remove top-k latents")
    plt.plot([r["k"] for r in ab], [r["only_top_k_auc"] for r in ab], marker="o", label="only top-k latents")
    plt.axhline(auc(labels, dino_scores), color="#333333", linestyle="--", linewidth=1, label="original DINO linear")
    plt.xlabel("k SAE latents ranked by contribution")
    plt.ylabel("UCAS val AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "sae_ablation_auc_highres.png", dpi=args.dpi)
    plt.close()

    bins = results["efficientnet_confidence_bins"]
    x = np.arange(len(bins))
    plt.figure(figsize=(9.2, 5.4))
    plt.plot(x, [r["eff_auc"] for r in bins], marker="o", label="EfficientNet")
    plt.plot(x, [r["dino_auc"] for r in bins], marker="o", label="DINOv3-S+ linear")
    plt.plot(x, [r["fusion_auc"] for r in bins], marker="o", label="Fusion")
    plt.xticks(x, [r["bin"] for r in bins], rotation=15, ha="right")
    plt.ylabel("AUC inside EfficientNet-confidence bin")
    plt.xlabel("|EfficientNet logit| quantile bin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "eff_confidence_bins_highres.png", dpi=args.dpi)
    plt.close()

    sample_sets = results["sample_sets"]
    for key, names in sample_sets.items():
        save_grid(args.ucas_val, boxes, names, labels, eff_scores, dino_scores, out_dir / f"samples_{key}_highres.jpg", cell=args.cell)

    for row in results["top_sae_latents"][:8]:
        latent = int(row["latent"])
        save_latent_grid(
            args.ucas_val,
            boxes,
            latent,
            h,
            img_names,
            labels,
            eff_scores,
            dino_scores,
            out_dir / f"latent_{latent}_top_activations_highres.jpg",
            cell=args.cell,
        )

    manifest = {
        "source_dir": args.source_dir,
        "cell": args.cell,
        "dpi": args.dpi,
        "metrics": metrics,
        "files": sorted([p.name for p in out_dir.iterdir() if p.is_file()]),
    }
    (out_dir / "highres_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
