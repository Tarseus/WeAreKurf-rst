import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


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
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def _best_box(raw_box):
    if raw_box is None:
        return None
    if len(raw_box) == 0:
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

    left = int(round(cx - side * 0.5))
    top = int(round(cy - side * 0.5))
    right = int(round(cx + side * 0.5))
    bottom = int(round(cy + side * 0.5))

    left = max(left, 0)
    top = max(top, 0)
    right = min(right, width)
    bottom = min(bottom, height)
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def _to_tensor(image, input_size):
    image = image.resize((input_size, input_size), RESAMPLE_BILINEAR)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).float()


class FolderDataset(Dataset):
    def __init__(self, img_folder, face_info, input_size=300, tta_mode="strong"):
        self.img_folder = img_folder
        self.img_names = sorted(
            name for name in os.listdir(img_folder)
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
        )
        self.face_info = face_info
        self.input_size = input_size
        self.tta_mode = tta_mode

        if tta_mode == "none":
            self.scales = (1.3,)
            self.flips = ("none",)
        elif tta_mode == "fast":
            self.scales = (1.3,)
            self.flips = ("none", "hflip", "vflip")
        else:
            self.scales = (1.15, 1.3, 1.45)
            self.flips = ("none", "hflip", "vflip")

    @property
    def num_views(self):
        return len(self.scales) * len(self.flips)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_folder, img_name)
        with Image.open(img_path) as image:
            image = image.convert("RGB")
            views = []
            for scale in self.scales:
                face = _crop_face(image, img_name, self.face_info, scale)
                for flip in self.flips:
                    if flip == "hflip":
                        view = face.transpose(Image.FLIP_LEFT_RIGHT)
                    elif flip == "vflip":
                        view = face.transpose(Image.FLIP_TOP_BOTTOM)
                    else:
                        view = face
                    views.append(_to_tensor(view, self.input_size))
        return torch.stack(views, dim=0)


class Model:
    def __init__(self):
        self.input_size = int(os.environ.get("DFGC_INPUT_SIZE", "300"))
        self.tta_mode = os.environ.get("DFGC_TTA", "strong").lower()
        if self.tta_mode not in {"none", "fast", "strong"}:
            self.tta_mode = "strong"

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = TransferModel("efficientnet-b3", num_out_classes=3)
        weight_path = os.path.join(THIS_DIR, "efn-b3_3c_60_acc0.9975.pth")
        if not os.path.isfile(weight_path):
            raise FileNotFoundError("Missing model weights: %s" % weight_path)

        state = torch.load(weight_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        default_batch = "4" if self.tta_mode == "strong" else "16"
        if self.device.type == "cpu":
            default_batch = "1"
        self.batchsize = int(os.environ.get("DFGC_BATCH_SIZE", default_batch))

    def run(self, input_dir, json_file):
        with open(json_file, "r", encoding="utf-8") as load_f:
            json_info = json.load(load_f)

        dataset_eval = FolderDataset(
            input_dir,
            json_info,
            input_size=self.input_size,
            tta_mode=self.tta_mode,
        )
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
            for batch in loader:
                batch_size, views, channels, height, width = batch.shape
                imgs = batch.reshape(batch_size * views, channels, height, width).to(self.device)
                outputs = softmax(self.model(imgs))
                fake_probs = 1.0 - outputs[:, 0]
                fake_probs = fake_probs.reshape(batch_size, views).mean(dim=1)
                prediction.append(fake_probs.cpu())

        if prediction:
            prediction = torch.cat(prediction, dim=0).numpy().reshape(-1).tolist()
        else:
            prediction = []

        prediction = [float(min(max(p, 0.0), 1.0)) for p in prediction]
        img_names = list(dataset_eval.img_names)
        assert isinstance(prediction, list)
        assert isinstance(img_names, list)
        assert len(prediction) == len(img_names)
        return img_names, prediction
