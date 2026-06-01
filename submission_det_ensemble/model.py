import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

try:
    from .efficientnet import TransferModel
except Exception:
    from efficientnet import TransferModel


ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    RESAMPLE_BILINEAR = Image.BILINEAR

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def _best_box(raw_box):
    if raw_box is None or len(raw_box) == 0:
        return None
    if isinstance(raw_box[0], (list, tuple)):
        def score(box):
            area = max(float(box[2]) - float(box[0]), 1.0) * max(float(box[3]) - float(box[1]), 1.0)
            conf = float(box[4]) if len(box) > 4 else 1.0
            return area * conf

        raw_box = max(raw_box, key=score)
    if len(raw_box) < 4:
        return None
    return [float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3])]


def _crop_face(image, image_name, face_info, scale):
    width, height = image.size
    stem = os.path.splitext(image_name)[0]
    entry = face_info.get(stem) or face_info.get(image_name) or {}
    box = _best_box(entry.get("box"))
    if box is None:
        return image

    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * scale
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    left = max(int(round(cx - side * 0.5)), 0)
    top = max(int(round(cy - side * 0.5)), 0)
    right = min(int(round(cx + side * 0.5)), width)
    bottom = min(int(round(cy + side * 0.5)), height)
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def _to_tensor(image, size):
    image = image.convert("RGB").resize((size, size), RESAMPLE_BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).float()


def _logit(p, eps=1e-6):
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p / (1.0 - p))


class FolderDataset(Dataset):
    def __init__(self, img_folder, face_info, eff_size=300, dino_size=518, tta_mode="strong"):
        self.img_folder = img_folder
        self.img_names = sorted(
            name for name in os.listdir(img_folder)
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
        )
        self.face_info = face_info
        self.eff_size = eff_size
        self.dino_size = dino_size
        if tta_mode == "none":
            self.scales = (1.3,)
            self.flips = ("none",)
        elif tta_mode == "fast":
            self.scales = (1.3,)
            self.flips = ("none", "hflip", "vflip")
        else:
            self.scales = (1.15, 1.3, 1.45)
            self.flips = ("none", "hflip", "vflip")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_folder, img_name)
        with Image.open(img_path) as image:
            image = image.convert("RGB")
            eff_views = []
            for scale in self.scales:
                face = _crop_face(image, img_name, self.face_info, scale)
                for flip in self.flips:
                    if flip == "hflip":
                        view = face.transpose(Image.FLIP_LEFT_RIGHT)
                    elif flip == "vflip":
                        view = face.transpose(Image.FLIP_TOP_BOTTOM)
                    else:
                        view = face
                    eff_views.append(_to_tensor(view, self.eff_size))
            dino_face = _crop_face(image, img_name, self.face_info, 1.3)
            dino_view = _to_tensor(dino_face, self.dino_size)
        return torch.stack(eff_views, dim=0), dino_view


class Model:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tta_mode = os.environ.get("DFGC_TTA", "strong").lower()
        if self.tta_mode not in {"none", "fast", "strong"}:
            self.tta_mode = "strong"

        self.eff_model = TransferModel("efficientnet-b3", num_out_classes=3)
        eff_path = os.path.join(THIS_DIR, "efn-b3_3c_60_acc0.9975.pth")
        self.eff_model.load_state_dict(torch.load(eff_path, map_location="cpu"))
        self.eff_model.to(self.device).eval()

        dino_path = os.path.join(THIS_DIR, "dino_dfgc21_probe_ts.pt")
        self.dino_model = None
        if os.path.isfile(dino_path):
            with open(dino_path, "rb") as f:
                self.dino_model = torch.jit.load(f, map_location=self.device)
            self.dino_model.eval()

        self.alpha_eff = float(os.environ.get("DFGC_ALPHA_EFF", "0.55"))
        default_batch = "4" if self.dino_model is not None else "8"
        if self.device.type == "cpu":
            default_batch = "1"
        self.batchsize = int(os.environ.get("DFGC_BATCH_SIZE", default_batch))

    def run(self, input_dir, json_file):
        with open(json_file, "r", encoding="utf-8") as load_f:
            json_info = json.load(load_f)

        dataset_eval = FolderDataset(input_dir, json_info, tta_mode=self.tta_mode)
        num_workers = 0 if os.name == "nt" else int(os.environ.get("DFGC_NUM_WORKERS", "4"))
        loader = DataLoader(
            dataset_eval,
            batch_size=self.batchsize,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        prediction = []
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for eff_batch, dino_batch in loader:
                batch_size, views, channels, height, width = eff_batch.shape
                eff_imgs = eff_batch.reshape(batch_size * views, channels, height, width).to(self.device)
                eff_outputs = softmax(self.eff_model(eff_imgs))
                eff_probs = 1.0 - eff_outputs[:, 0]
                eff_probs = eff_probs.reshape(batch_size, views).mean(dim=1)

                if self.dino_model is None:
                    fused = eff_probs
                else:
                    dino_probs = self.dino_model(dino_batch.to(self.device)).reshape(-1)
                    fused_logit = self.alpha_eff * _logit(eff_probs) + (1.0 - self.alpha_eff) * _logit(dino_probs)
                    fused = torch.sigmoid(fused_logit)
                prediction.append(fused.cpu())

        if prediction:
            prediction = torch.cat(prediction, dim=0).numpy().reshape(-1).tolist()
        else:
            prediction = []
        prediction = [float(min(max(p, 0.0), 1.0)) for p in prediction]
        img_names = list(dataset_eval.img_names)
        assert len(prediction) == len(img_names)
        return img_names, prediction
