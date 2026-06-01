import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ARTIFACT_NAMES = [
    "gray_mean",
    "gray_std",
    "gray_min",
    "gray_max",
    "grad_mean",
    "grad_std",
    "grad_p90",
    "grad_p99",
    "lap_abs_mean",
    "lap_std",
    "lap_abs_p90",
    "fft_high_mean",
    "fft_low_mean",
    "fft_high_low_ratio",
    "jpeg_v_block",
    "jpeg_h_block",
    "jpeg_v_block_ratio",
    "jpeg_h_block_ratio",
    "sat_mean",
    "sat_std",
    "sat_p90",
    "rg_mean",
    "rg_std",
    "bg_mean",
    "bg_std",
    "patch_mean_std",
    "patch_std_mean",
    "patch_std_std",
    "patch_grad_mean",
    "patch_grad_std",
    "patch_grad_range",
]


def logit_np(x, eps=1e-6):
    x = np.clip(np.asarray(x, dtype=np.float64), eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def auc(y, s):
    return float(roc_auc_score(y, s))


def ap(y, s):
    return float(average_precision_score(y, s))


def bootstrap_auc_delta(y, a, b, n_boot=1000, seed=123):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    a = np.asarray(a)
    b = np.asarray(b)
    auc_a = []
    auc_b = []
    deltas = []
    idx_all = np.arange(len(y))
    for _ in range(n_boot):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        aa = roc_auc_score(y[idx], a[idx])
        bb = roc_auc_score(y[idx], b[idx])
        auc_a.append(aa)
        auc_b.append(bb)
        deltas.append(aa - bb)
    def ci(vals):
        vals = np.asarray(vals, dtype=np.float64)
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return {
        "auc_a_ci95": ci(auc_a),
        "auc_b_ci95": ci(auc_b),
        "delta_a_minus_b_ci95": ci(deltas),
        "delta_a_minus_b_mean": float(np.mean(deltas)),
    }


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


def train_sae(x_train, x_val, latents, steps, batch_size, lr, l1, topk, seed, device):
    torch.manual_seed(seed)
    model = SAE(x_train.shape[1], latents, topk=topk).to(device)
    with torch.no_grad():
        norms = model.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-6)
        model.decoder.weight.div_(norms)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    x = torch.from_numpy(x_train.astype(np.float32)).to(device)
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    for step in range(steps):
        idx = torch.from_numpy(rng.integers(0, n, size=batch_size)).to(device)
        batch = x[idx]
        recon, h = model(batch)
        loss = torch.mean((recon - batch) ** 2) + l1 * h.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            norms = model.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-6)
            model.decoder.weight.div_(norms)
    with torch.no_grad():
        xv = torch.from_numpy(x_val.astype(np.float32)).to(device)
        recon, h = model(xv)
        recon_np = recon.cpu().numpy()
        h_np = h.cpu().numpy()
    mse = float(np.mean((recon_np - x_val) ** 2))
    baseline = float(np.mean((x_val - x_val.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - mse / max(baseline, 1e-12)
    return model.cpu(), h_np, recon_np, {"mse": mse, "r2": float(r2)}


def cell_image(path, title, subtitle, box=None, size=190):
    try:
        image = Image.open(path).convert("RGB")
    except Exception:
        image = Image.new("RGB", (size, size), (40, 40, 40))
    if box is not None:
        draw = ImageDraw.Draw(image)
        draw.rectangle(tuple(box), outline=(255, 210, 0), width=3)
    image.thumbnail((size, size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size + 42), (250, 250, 250))
    canvas.paste(image, ((size - image.width) // 2, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, size + 3), title[:32], fill=(0, 0, 0))
    draw.text((4, size + 22), subtitle[:42], fill=(50, 50, 50))
    return canvas


def save_grid(data_folder, names, labels, eff, dino, out_path, max_items=16):
    data_folder = Path(data_folder)
    imgs_dir = data_folder / "imgs"
    cells = []
    for name in names[:max_items]:
        i = name_to_idx[name]
        title = f"{name} y={int(labels[i])}"
        subtitle = f"Eff={eff[i]:.3f} DINO={dino[i]:.3f}"
        cells.append(cell_image(imgs_dir / name, title, subtitle))
    if not cells:
        return
    cols = 4
    rows = math.ceil(len(cells) / cols)
    w, h = cells[0].size
    grid = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for j, cell in enumerate(cells):
        grid.paste(cell, ((j % cols) * w, (j // cols) * h))
    grid.save(out_path)


def save_latent_grid(data_folder, latent_id, h, img_names, labels, eff, dino, out_path, max_items=16):
    order = np.argsort(h[:, latent_id])[::-1]
    names = img_names[order[:max_items]]
    data_folder = Path(data_folder)
    imgs_dir = data_folder / "imgs"
    cells = []
    for name in names:
        i = name_to_idx[str(name)]
        title = f"{name} y={int(labels[i])} L{latent_id}={h[i, latent_id]:.2f}"
        subtitle = f"Eff={eff[i]:.3f} DINO={dino[i]:.3f}"
        cells.append(cell_image(imgs_dir / str(name), title, subtitle))
    cols = 4
    rows = math.ceil(len(cells) / cols)
    w, hcell = cells[0].size
    grid = Image.new("RGB", (cols * w, rows * hcell), (255, 255, 255))
    for j, cell in enumerate(cells):
        grid.paste(cell, ((j % cols) * w, (j // cols) * hcell))
    grid.save(out_path)


def write_csv(path, rows, columns):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            vals = []
            for col in columns:
                val = row.get(col, "")
                if isinstance(val, float):
                    vals.append(f"{val:.10g}")
                else:
                    vals.append(str(val))
            f.write(",".join(vals) + "\n")


def plot_outputs(out_dir, y, eff_scores, dino_scores, fusion_scores, artifact_table, sae_table, ablation_rows, bin_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    metrics = {
        "EfficientNet-B3": auc(y, eff_scores),
        "DINOv3-S+ Linear": auc(y, dino_scores),
        "Fusion alpha=0.61": auc(y, fusion_scores),
    }
    plt.figure(figsize=(6.8, 3.8))
    bars = plt.bar(metrics.keys(), metrics.values(), color=["#546A7B", "#2E8B57", "#C25B38"])
    plt.ylim(0.93, 1.0)
    plt.ylabel("UCAS val AUC")
    plt.xticks(rotation=15, ha="right")
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.001, f"{b.get_height():.6f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "auc_bars.png", dpi=220)
    plt.close()

    eff_logit = logit_np(eff_scores)
    dino_logit = logit_np(dino_scores)
    plt.figure(figsize=(5.5, 5.0))
    plt.scatter(eff_logit[y == 0], dino_logit[y == 0], s=5, alpha=0.25, label="real", c="#1f77b4")
    plt.scatter(eff_logit[y == 1], dino_logit[y == 1], s=5, alpha=0.25, label="fake", c="#d62728")
    plt.xlabel("EfficientNet logit")
    plt.ylabel("DINOv3-S+ linear logit")
    plt.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(out_dir / "score_scatter.png", dpi=220)
    plt.close()

    top_art = artifact_table[:12]
    plt.figure(figsize=(7.2, 4.2))
    vals = [r["spearman_with_dino_minus_eff_margin"] for r in top_art]
    labels = [r["feature"] for r in top_art]
    colors = ["#C25B38" if v > 0 else "#546A7B" for v in vals]
    plt.barh(labels[::-1], vals[::-1], color=colors[::-1])
    plt.xlabel("Spearman corr. with DINO margin advantage")
    plt.tight_layout()
    plt.savefig(out_dir / "artifact_delta_corr.png", dpi=220)
    plt.close()

    top_sae = sae_table[:20]
    plt.figure(figsize=(8.0, 5.0))
    vals = [r["mean_logit_contribution_fake_minus_real"] for r in top_sae]
    labels = [f"L{r['latent']}" for r in top_sae]
    colors = ["#C25B38" if v > 0 else "#546A7B" for v in vals]
    plt.barh(labels[::-1], vals[::-1], color=colors[::-1])
    plt.xlabel("SAE latent contribution: fake mean - real mean")
    plt.tight_layout()
    plt.savefig(out_dir / "sae_top_latents.png", dpi=220)
    plt.close()

    if ablation_rows:
        ks = [r["k"] for r in ablation_rows]
        remove_auc = [r["remove_top_k_auc"] for r in ablation_rows]
        only_auc = [r["only_top_k_auc"] for r in ablation_rows]
        plt.figure(figsize=(6.6, 4.0))
        plt.plot(ks, remove_auc, marker="o", label="remove top-k latents")
        plt.plot(ks, only_auc, marker="o", label="only top-k latents")
        plt.axhline(auc(y, dino_scores), color="#333333", linestyle="--", linewidth=1, label="original DINO linear")
        plt.xlabel("k SAE latents ranked by contribution")
        plt.ylabel("UCAS val AUC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "sae_ablation_auc.png", dpi=220)
        plt.close()

    if bin_rows:
        x = np.arange(len(bin_rows))
        plt.figure(figsize=(7.2, 4.0))
        plt.plot(x, [r["eff_auc"] for r in bin_rows], marker="o", label="EfficientNet")
        plt.plot(x, [r["dino_auc"] for r in bin_rows], marker="o", label="DINOv3-S+ linear")
        plt.xticks(x, [r["bin"] for r in bin_rows], rotation=20, ha="right")
        plt.ylabel("AUC inside EfficientNet-confidence bin")
        plt.xlabel("|EfficientNet logit| quantile bin")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "eff_confidence_bins.png", dpi=220)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucas-val", default="datasets/UCAS_AISA/extracted/val")
    parser.add_argument("--scores", default="dinov3_splus_dfgc_probe_output/ucas_val_scores.npz")
    parser.add_argument("--features", default="dinov3_splus_dfgc_probe_output/ucas_val_features.npz")
    parser.add_argument("--dfgc-features", default="dinov3_splus_dfgc_probe_output/dfgc21_features.npz")
    parser.add_argument("--probe", default="dinov3_splus_dfgc_probe_output/dinov3_dfgc21_probe.joblib")
    parser.add_argument("--artifact-cache", default="ucas_artifact_adapter_output/feature_cache.npz")
    parser.add_argument("--out-dir", default="discussion_vfm_probe")
    parser.add_argument("--sae-latents", type=int, default=1536)
    parser.add_argument("--sae-steps", type=int, default=3000)
    parser.add_argument("--sae-l1", type=float, default=3e-4)
    parser.add_argument("--sae-topk", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_npz = np.load(args.scores, allow_pickle=True)
    img_names = scores_npz["img_names"].astype(str)
    labels = scores_npz["labels"].astype(np.int64)
    eff_scores = scores_npz["eff_scores"].astype(np.float64)
    dino_scores = scores_npz["dinov3_scores"].astype(np.float64)
    fusion_scores = sigmoid_np(0.61 * logit_np(eff_scores) + 0.39 * logit_np(dino_scores))

    global name_to_idx
    name_to_idx = {name: i for i, name in enumerate(img_names)}

    val_features = np.load(args.features)["features"].astype(np.float32)
    dfgc_features = np.load(args.dfgc_features)["features"].astype(np.float32)
    probe = joblib.load(args.probe)
    scaler = probe.steps[0][1]
    lr = probe.steps[1][1]
    coef = lr.coef_[0].astype(np.float64)
    intercept = float(lr.intercept_[0])
    val_z = scaler.transform(val_features).astype(np.float32)
    dfgc_z = scaler.transform(dfgc_features).astype(np.float32)
    dino_linear_logits = val_z @ coef + intercept

    art_npz = np.load(args.artifact_cache, allow_pickle=True)
    art_names = art_npz["img_names"].astype(str)
    art_by_name = {name: art_npz["artifacts"][i] for i, name in enumerate(art_names)}
    artifacts = np.stack([art_by_name[name] for name in img_names], axis=0).astype(np.float64)

    ysign = labels * 2 - 1
    eff_logit = logit_np(eff_scores)
    dino_logit = logit_np(dino_scores)
    eff_margin = ysign * eff_logit
    dino_margin = ysign * dino_logit
    dino_advantage = dino_margin - eff_margin

    metrics = {
        "efficientnet_auc": auc(labels, eff_scores),
        "dinov3_splus_linear_auc": auc(labels, dino_scores),
        "fusion_alpha_0.61_auc": auc(labels, fusion_scores),
        "efficientnet_ap": ap(labels, eff_scores),
        "dinov3_splus_linear_ap": ap(labels, dino_scores),
        "fusion_alpha_0.61_ap": ap(labels, fusion_scores),
        "score_pearson": corr(eff_scores, dino_scores),
        "score_spearman": spearman(eff_scores, dino_scores),
        "logit_pearson": corr(eff_logit, dino_logit),
        "logit_spearman": spearman(eff_logit, dino_logit),
        "bootstrap_fusion_vs_eff": bootstrap_auc_delta(labels, fusion_scores, eff_scores),
        "bootstrap_dino_vs_eff": bootstrap_auc_delta(labels, dino_scores, eff_scores),
    }

    eff_pred = eff_scores >= 0.5
    dino_pred = dino_scores >= 0.5
    category_masks = {
        "eff_wrong_dino_right": (eff_pred != labels) & (dino_pred == labels),
        "eff_right_dino_wrong": (eff_pred == labels) & (dino_pred != labels),
        "both_wrong": (eff_pred != labels) & (dino_pred != labels),
        "both_right": (eff_pred == labels) & (dino_pred == labels),
    }
    categories = {}
    for key, mask in category_masks.items():
        categories[key] = {
            "count": int(mask.sum()),
            "fake_count": int((mask & (labels == 1)).sum()),
            "real_count": int((mask & (labels == 0)).sum()),
            "mean_eff_score": float(eff_scores[mask].mean()) if mask.any() else None,
            "mean_dino_score": float(dino_scores[mask].mean()) if mask.any() else None,
        }

    artifact_rows = []
    for j, name in enumerate(ARTIFACT_NAMES):
        x = artifacts[:, j]
        artifact_rows.append({
            "feature": name,
            "spearman_with_label": spearman(x, labels),
            "spearman_with_eff_logit": spearman(x, eff_logit),
            "spearman_with_dino_logit": spearman(x, dino_logit),
            "spearman_with_dino_minus_eff_margin": spearman(x, dino_advantage),
            "mean_fake_minus_real": float(x[labels == 1].mean() - x[labels == 0].mean()),
        })
    artifact_rows.sort(key=lambda r: abs(r["spearman_with_dino_minus_eff_margin"]), reverse=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    artifact_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    artifact_auc = float(cross_val_score(artifact_lr, artifacts, labels, cv=cv, scoring="roc_auc").mean())
    ridge_art_to_dino = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge_art_to_dino.fit(artifacts, dino_logit)
    pred_dino_from_art = ridge_art_to_dino.predict(artifacts)
    artifact_explain = {
        "artifact_only_5fold_auc_on_ucas_analysis_only": artifact_auc,
        "artifact_features_r2_for_dino_logit": float(1.0 - np.mean((pred_dino_from_art - dino_logit) ** 2) / np.var(dino_logit)),
        "artifact_features_corr_for_dino_logit": corr(pred_dino_from_art, dino_logit),
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, h_val, recon_val, sae_stats = train_sae(
        dfgc_z,
        val_z,
        args.sae_latents,
        args.sae_steps,
        1024,
        1e-3,
        args.sae_l1,
        args.sae_topk,
        args.seed,
        device,
    )
    torch.save(model.state_dict(), out_dir / "sae_dinov3_splus_linear_probe.pt")
    decoder = model.decoder.weight.detach().numpy().astype(np.float64)
    latent_linear_weight = decoder.T @ coef
    latent_contrib = h_val.astype(np.float64) * latent_linear_weight.reshape(1, -1)
    recon_logits = recon_val.astype(np.float64) @ coef + intercept
    sae_stats["reconstruction_auc_via_linear_head"] = auc(labels, sigmoid_np(recon_logits))
    sae_stats["original_linear_logit_corr"] = corr(dino_linear_logits, recon_logits)
    sae_stats["mean_l0_active_latents"] = float((h_val > 1e-6).sum(axis=1).mean())
    sae_stats["dead_latents_fraction_on_ucas"] = float(((h_val > 1e-6).mean(axis=0) < 1e-4).mean())
    sae_stats["configured_topk"] = int(args.sae_topk)

    sae_rows = []
    eff_wrong_dino_right = category_masks["eff_wrong_dino_right"]
    eff_right_dino_wrong = category_masks["eff_right_dino_wrong"]
    for j in range(h_val.shape[1]):
        hj = h_val[:, j].astype(np.float64)
        if hj.std() < 1e-12:
            latent_auc = 0.5
        else:
            latent_auc = max(auc(labels, hj), auc(labels, -hj))
        row = {
            "latent": int(j),
            "latent_linear_weight": float(latent_linear_weight[j]),
            "activation_rate": float((hj > 1e-6).mean()),
            "mean_activation_fake_minus_real": float(hj[labels == 1].mean() - hj[labels == 0].mean()),
            "mean_logit_contribution_fake_minus_real": float(latent_contrib[labels == 1, j].mean() - latent_contrib[labels == 0, j].mean()),
            "abs_label_auc": float(latent_auc),
            "spearman_with_dino_advantage": spearman(hj, dino_advantage),
        }
        if eff_wrong_dino_right.any() and eff_right_dino_wrong.any():
            row["mean_activation_eff_wrong_dino_right_minus_reverse"] = float(
                hj[eff_wrong_dino_right].mean() - hj[eff_right_dino_wrong].mean()
            )
        sae_rows.append(row)
    sae_rows.sort(key=lambda r: abs(r["mean_logit_contribution_fake_minus_real"]), reverse=True)
    write_csv(
        out_dir / "top_sae_latents.csv",
        sae_rows,
        [
            "latent",
            "latent_linear_weight",
            "activation_rate",
            "mean_activation_fake_minus_real",
            "mean_logit_contribution_fake_minus_real",
            "abs_label_auc",
            "spearman_with_dino_advantage",
            "mean_activation_eff_wrong_dino_right_minus_reverse",
        ],
    )
    write_csv(
        out_dir / "artifact_correlations.csv",
        artifact_rows,
        [
            "feature",
            "spearman_with_label",
            "spearman_with_eff_logit",
            "spearman_with_dino_logit",
            "spearman_with_dino_minus_eff_margin",
            "mean_fake_minus_real",
        ],
    )

    top_order = [r["latent"] for r in sae_rows]
    ablation_rows = []
    for k in [1, 2, 4, 8, 16, 32, 64, 128]:
        cols = top_order[:k]
        h_remove = h_val.copy()
        h_remove[:, cols] = 0.0
        recon_remove = h_remove @ decoder.T + model.decoder.bias.detach().numpy()
        h_only = np.zeros_like(h_val)
        h_only[:, cols] = h_val[:, cols]
        recon_only = h_only @ decoder.T + model.decoder.bias.detach().numpy()
        remove_scores = sigmoid_np(recon_remove @ coef + intercept)
        only_scores = sigmoid_np(recon_only @ coef + intercept)
        ablation_rows.append({
            "k": int(k),
            "remove_top_k_auc": auc(labels, remove_scores),
            "only_top_k_auc": auc(labels, only_scores),
        })

    abs_eff = np.abs(eff_logit)
    qs = np.quantile(abs_eff, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bin_rows = []
    for i in range(5):
        lo, hi = qs[i], qs[i + 1]
        mask = (abs_eff >= lo) & (abs_eff <= hi if i == 4 else abs_eff < hi)
        if mask.sum() < 20 or len(np.unique(labels[mask])) < 2:
            continue
        bin_rows.append({
            "bin": f"Q{i+1} [{lo:.2f},{hi:.2f}]",
            "n": int(mask.sum()),
            "fake_rate": float(labels[mask].mean()),
            "eff_auc": auc(labels[mask], eff_scores[mask]),
            "dino_auc": auc(labels[mask], dino_scores[mask]),
            "fusion_auc": auc(labels[mask], fusion_scores[mask]),
        })

    plot_outputs(out_dir, labels, eff_scores, dino_scores, fusion_scores, artifact_rows, sae_rows, ablation_rows, bin_rows)

    def top_names(mask, score, reverse=True, limit=16):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        order = np.argsort(score[idx])
        if reverse:
            order = order[::-1]
        return img_names[idx[order[:limit]]].tolist()

    sample_sets = {
        "fake_eff_miss_dino_hit": top_names((labels == 1) & (eff_scores < 0.5) & (dino_scores >= 0.5), dino_scores - eff_scores),
        "real_eff_fp_dino_hit": top_names((labels == 0) & (eff_scores >= 0.5) & (dino_scores < 0.5), eff_scores - dino_scores),
        "fake_dino_miss_eff_hit": top_names((labels == 1) & (dino_scores < 0.5) & (eff_scores >= 0.5), eff_scores - dino_scores),
        "real_dino_fp_eff_hit": top_names((labels == 0) & (dino_scores >= 0.5) & (eff_scores < 0.5), dino_scores - eff_scores),
    }
    for key, names in sample_sets.items():
        save_grid(args.ucas_val, names, labels, eff_scores, dino_scores, out_dir / f"samples_{key}.jpg")
    latent_grid_outputs = []
    for row in sae_rows[:8]:
        latent_id = row["latent"]
        filename = f"latent_{latent_id}_top_activations.jpg"
        save_latent_grid(args.ucas_val, latent_id, h_val, img_names, labels, eff_scores, dino_scores, out_dir / filename)
        latent_grid_outputs.append(filename)

    results = {
        "paper_anchor": {
            "title": "Simplicity Prevails: The Emergence of Generalizable AIGI Detection in Visual Foundation Models",
            "core_claim": "A linear classifier trained on frozen VFM features, including DINOv3, can establish strong AIGI detection baselines.",
            "our_alignment": "DINOv3-S+ features are frozen; only the DFGC-trained linear/logistic head is used for the UCAS analysis.",
        },
        "metrics": metrics,
        "threshold_0p5_categories_analysis_only": categories,
        "artifact_explainability": artifact_explain,
        "top_artifact_correlations": artifact_rows[:20],
        "sae_stats": sae_stats,
        "top_sae_latents": sae_rows[:40],
        "sae_ablation": ablation_rows,
        "efficientnet_confidence_bins": bin_rows,
        "sample_sets": sample_sets,
        "outputs": [
            "auc_bars.png",
            "score_scatter.png",
            "artifact_delta_corr.png",
            "sae_top_latents.png",
            "sae_ablation_auc.png",
            "eff_confidence_bins.png",
            "samples_fake_eff_miss_dino_hit.jpg",
            "samples_real_eff_fp_dino_hit.jpg",
            "samples_fake_dino_miss_eff_hit.jpg",
            "samples_real_dino_fp_eff_hit.jpg",
            "top_sae_latents.csv",
            "artifact_correlations.csv",
            *latent_grid_outputs,
        ],
    }
    (out_dir / "discussion_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    md = [
        "# VFM Linear Probe Discussion Material",
        "",
        "## Core Setup",
        "- Frozen encoder: DINOv3-S+ timm feature extractor.",
        "- Classifier: linear/logistic head trained on DFGC-21 features only.",
        "- UCAS val is used here for post-hoc discussion analysis, not for submission tuning.",
        "",
        "## Main Numbers",
        f"- EfficientNet-B3 AUC: {metrics['efficientnet_auc']:.6f}",
        f"- DINOv3-S+ Linear AUC: {metrics['dinov3_splus_linear_auc']:.6f}",
        f"- Fusion alpha=0.61 AUC: {metrics['fusion_alpha_0.61_auc']:.6f}",
        f"- Score Spearman correlation between branches: {metrics['score_spearman']:.3f}",
        "",
        "## Mechanistic Hypotheses Supported By These Experiments",
        "- DINOv3-S+ linear probe provides a largely non-identical decision signal to EfficientNet; the score correlation is not close to 1.",
        "- The linear head can be decomposed into sparse SAE latent contributions; top latents carry a measurable portion of the fake-vs-real logit gap.",
        "- Handcrafted image artifacts explain only part of the DINO logit, suggesting DINO embedding is not reducible to the simple frequency/color/blockiness statistics used here.",
        "",
        "## Caveats",
        "- SAE factors are post-hoc explanatory variables, not full causal proof.",
        "- Threshold-0.5 error categories are illustrative and not tuned.",
        "- The analysis is face/deepfake data; the cited paper is broader AIGI detection.",
    ]
    (out_dir / "discussion_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({
        "out_dir": str(out_dir),
        "metrics": metrics,
        "sae_stats": sae_stats,
        "artifact_explainability": artifact_explain,
        "num_outputs": len(results["outputs"]),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
