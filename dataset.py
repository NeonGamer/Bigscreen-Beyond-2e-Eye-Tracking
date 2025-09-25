import os
from typing import List
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

def find_extrema(data_dir: str):
    x_vals, y_vals = [], []

    for root, _, filenames in os.walk(data_dir):
        for f in filenames:
            parts = f.replace(".jpg", "").split("_")
            if len(parts) != 5:
                continue

            _, x, y, blink, _ = parts
            x_vals.append(float(x))
            y_vals.append(float(y))

    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    x_max_abs = max(abs(x_min), abs(x_max))
    y_max_abs = max(abs(y_min), abs(y_max))

    return x_max_abs, y_max_abs

class EyeGazeDataset(Dataset):
    def __init__(self, data_dir: str, x_max_abs: float, y_max_abs: float, eye: str, img_transform: Compose):
        self.files: List[str] = []
        for root, _, filenames in os.walk(data_dir):
            for f in filenames:
                if f.lower().endswith(".jpg"):
                    self.files.append(os.path.join(root, f))
        self.files.sort()
        self.data_dir = data_dir
        self.eye = eye
        self.transform = img_transform
        self.x_max_abs = x_max_abs
        self.y_max_abs = y_max_abs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        filename = self.files[idx]
        path = str(os.path.join(self.data_dir, filename))

        img = cv2.imread(path)
        h, w, _ = img.shape
        mid = w // 2
        eye_img = img[:, :mid] if self.eye == "left" else img[:, mid:]
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
        eye_img = Image.fromarray(eye_img)
        if self.transform:
            eye_img = self.transform(eye_img)

        parts = filename[:-4].split("_")
        x = float(parts[-4])
        y = float(parts[-3])
        x_norm = 1 - symmetric_normalize(x, self.x_max_abs)
        y_norm = symmetric_normalize(y, self.y_max_abs)
        blink = float(parts[-2])
        dilation = float(parts[-1])

        gaze_target = torch.tensor([x_norm, y_norm], dtype=torch.float32)
        if blink == .75:
            blink = .65
        elif blink == .375:
            blink = .25
        eyelid_target = torch.tensor([blink], dtype=torch.float32)
        #.6 - 1 widen
        # .3 - .65 squint
        #  0 - .25 closed
        dilation_target = torch.tensor([dilation], dtype=torch.float32)
        return eye_img, gaze_target, eyelid_target, dilation_target

def symmetric_normalize(value: float, max_abs: float):
    return (value + max_abs) / (2 * max_abs)