import argparse
import importlib
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def read_lines(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def official_face_info(data_folder):
    img_names = read_lines(Path(data_folder) / "img_list.txt")
    face_boxes = read_lines(Path(data_folder) / "face_info.txt")
    if len(img_names) != len(face_boxes):
        raise ValueError(f"img_list has {len(img_names)} rows, face_info has {len(face_boxes)} rows")
    face_info = {}
    for name, box_line in zip(img_names, face_boxes):
        box = [float(x) for x in box_line.split()[:4]]
        face_info[Path(name).stem] = {"box": box}
        face_info[name] = {"box": box}
    return img_names, face_info


def _xlsx_rows(path):
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with zipfile.ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("main:si", ns):
                parts = [t.text or "" for t in si.findall(".//main:t", ns)]
                shared.append("".join(parts))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        sheet = workbook.find("main:sheets/main:sheet", ns)
        if sheet is None:
            return []
        rid = sheet.attrib.get(f"{{{ns['rel']}}}id")

        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in rels:
            if rel.attrib.get("Id") == rid:
                target = rel.attrib["Target"]
                break
        if target is None:
            return []
        sheet_path = "xl/" + target.lstrip("/")
        root = ET.fromstring(zf.read(sheet_path))
        rows = []
        for row in root.findall(".//main:sheetData/main:row", ns):
            values = []
            for c in row.findall("main:c", ns):
                value = c.find("main:v", ns)
                text = "" if value is None or value.text is None else value.text
                if c.attrib.get("t") == "s" and text != "":
                    text = shared[int(text)]
                values.append(text)
            rows.append(values)
        return rows


def read_labels(data_folder, expected_len):
    data_folder = Path(data_folder)
    candidates = []
    for pattern in ("gts.xlsx", "*gts*.xlsx", "*label*.xlsx", "labels.txt", "label.txt", "labels.csv", "label.csv"):
        candidates.extend(sorted(data_folder.glob(pattern)))
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        candidates.extend(sorted(data_folder.rglob("gts.xlsx")))
    if not candidates:
        raise FileNotFoundError(f"No label file found under {data_folder}")

    for path in candidates:
        suffix = path.suffix.lower()
        if suffix == ".xlsx":
            rows = _xlsx_rows(path)
            if not rows:
                continue
            header = [str(x).strip().lower() for x in rows[0]]
            label_col = header.index("labels") if "labels" in header else (header.index("label") if "label" in header else None)
            if label_col is None:
                if len(rows[0]) == 1:
                    label_col = 0
                    data_rows = rows
                else:
                    continue
            else:
                data_rows = rows[1:]
            labels = [int(float(row[label_col])) for row in data_rows if len(row) > label_col and str(row[label_col]).strip() != ""]
        else:
            labels = []
            for line in read_lines(path):
                parts = line.replace(",", " ").split()
                labels.append(int(float(parts[-1])))
        if len(labels) == expected_len:
            print(f"Using labels: {path}")
            return np.asarray(labels, dtype=np.int64), str(path)

    raise RuntimeError(f"Found label candidates but none matched {expected_len} rows: {candidates}")


def auc_score(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both positive and negative labels")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) * 0.5
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) * 0.5) / (n_pos * n_neg))


def logit_np(x, eps=1e-6):
    x = np.clip(np.asarray(x, dtype=np.float64), eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def rank01(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    if len(x) <= 1:
        return ranks
    return ranks / (len(x) - 1)


def search_fusion(labels, eff_scores, dino_scores, alpha_step):
    candidates = []
    candidates.append({"method": "efficientnet", "alpha_eff": 1.0, "auc": auc_score(labels, eff_scores)})
    candidates.append({"method": "dino", "alpha_eff": 0.0, "auc": auc_score(labels, dino_scores)})

    eff_logit = logit_np(eff_scores)
    dino_logit = logit_np(dino_scores)
    eff_rank = rank01(eff_scores)
    dino_rank = rank01(dino_scores)

    n_steps = int(round(1.0 / alpha_step))
    alphas = np.linspace(0.0, 1.0, n_steps + 1)
    for alpha in alphas:
        alpha = float(alpha)
        logit_scores = sigmoid_np(alpha * eff_logit + (1.0 - alpha) * dino_logit)
        prob_scores = alpha * eff_scores + (1.0 - alpha) * dino_scores
        rank_scores = alpha * eff_rank + (1.0 - alpha) * dino_rank
        candidates.append({"method": "logit", "alpha_eff": alpha, "auc": auc_score(labels, logit_scores)})
        candidates.append({"method": "prob", "alpha_eff": alpha, "auc": auc_score(labels, prob_scores)})
        candidates.append({"method": "rank", "alpha_eff": alpha, "auc": auc_score(labels, rank_scores)})

    return sorted(candidates, key=lambda x: x["auc"], reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--submission-dir", default="submission_det_ensemble")
    parser.add_argument("--out", default="ucas_val_tune_results.json")
    parser.add_argument("--scores-out", default="ucas_val_branch_scores.npz")
    parser.add_argument("--tta", default=os.environ.get("DFGC_TTA", "strong"), choices=["none", "fast", "strong"])
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("DFGC_BATCH_SIZE", "4")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("DFGC_NUM_WORKERS", "4")))
    parser.add_argument("--alpha-step", type=float, default=0.01)
    args = parser.parse_args()

    data_folder = Path(args.data_folder)
    imgs_dir = data_folder / "imgs"
    if not imgs_dir.is_dir():
        raise FileNotFoundError(imgs_dir)
    img_list, face_info = official_face_info(data_folder)
    labels, label_file = read_labels(data_folder, len(img_list))

    sys.path.insert(0, str(Path(args.submission_dir).resolve()))
    model_module = importlib.import_module("model")

    dataset = model_module.FolderDataset(str(imgs_dir), face_info, tta_mode=args.tta)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if os.name != "nt" else 0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eff_model = model_module.TransferModel("efficientnet-b3", num_out_classes=3)
    eff_path = Path(args.submission_dir) / "efn-b3_3c_60_acc0.9975.pth"
    eff_model.load_state_dict(torch.load(eff_path, map_location="cpu"))
    eff_model.to(device).eval()

    dino_path = Path(args.submission_dir) / "dino_dfgc21_probe_ts.pt"
    with open(dino_path, "rb") as f:
        dino_model = torch.jit.load(f, map_location=device)
    dino_model.eval()

    softmax = nn.Softmax(dim=1)
    eff_scores = []
    dino_scores = []
    start = time.time()
    with torch.no_grad():
        for eff_batch, dino_batch in loader:
            batch_size, views, channels, height, width = eff_batch.shape
            eff_imgs = eff_batch.reshape(batch_size * views, channels, height, width).to(device)
            eff_outputs = softmax(eff_model(eff_imgs))
            eff_probs = 1.0 - eff_outputs[:, 0]
            eff_probs = eff_probs.reshape(batch_size, views).mean(dim=1)
            dino_probs = dino_model(dino_batch.to(device)).reshape(-1)
            eff_scores.append(eff_probs.cpu().numpy())
            dino_scores.append(dino_probs.cpu().numpy())

    elapsed = time.time() - start
    eff_scores = np.concatenate(eff_scores).astype(np.float64)
    dino_scores = np.concatenate(dino_scores).astype(np.float64)
    score_by_name = {
        name: (float(eff), float(dino))
        for name, eff, dino in zip(dataset.img_names, eff_scores, dino_scores)
    }
    eff_ordered = np.asarray([score_by_name[name][0] for name in img_list], dtype=np.float64)
    dino_ordered = np.asarray([score_by_name[name][1] for name in img_list], dtype=np.float64)

    search = search_fusion(labels, eff_ordered, dino_ordered, args.alpha_step)
    current = sigmoid_np(0.55 * logit_np(eff_ordered) + 0.45 * logit_np(dino_ordered))
    summary = {
        "data_folder": str(data_folder),
        "label_file": label_file,
        "tta": args.tta,
        "batch_size": args.batch_size,
        "num_images": len(img_list),
        "seconds": elapsed,
        "auc_efficientnet": auc_score(labels, eff_ordered),
        "auc_dino": auc_score(labels, dino_ordered),
        "auc_current_logit_alpha_0.55": auc_score(labels, current),
        "best": search[0],
        "top10": search[:10],
    }
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.scores_out,
        img_names=np.asarray(img_list),
        labels=labels,
        eff_scores=eff_ordered,
        dino_scores=dino_ordered,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
