import torch

import variables
import threading
from typing import Callable, List, Tuple

import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from dataset import EyeGazeDataset

class EyeCNN(nn.Module):
    def __init__(self):
        pass

TARGET_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

left_eye_model: EyeCNN
right_eye_model: EyeCNN

left_eye_graph: List[float] = []
right_eye_graph: List[float] = []
left_eye_val_graph: List[float] = []
right_eye_val_graph: List[float] = []

runtime_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

training_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor()
])

class EyeCNN(nn.Module):
    def __init__(self):
        super(EyeCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 96, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.gaze_conv4 = nn.Conv2d(96, 165, 3, padding=1)
        self.gaze_conv5 = nn.Conv2d(165, 256, 3, padding=1)
        self.gaze_adaptive = nn.AdaptiveAvgPool2d((1, 1))
        self.gaze_head = nn.Linear(256, 2)

        self.eyelid_conv4 = nn.Conv2d(96, 165, 3, padding=1)
        self.eyelid_conv5 = nn.Conv2d(165, 256, 3, padding=1)
        self.eyelid_adaptive = nn.AdaptiveAvgPool2d((1, 1))
        self.eyelid_head = nn.Linear(256, 1)

        self.dilation_conv4 = nn.Conv2d(96, 165, 3, padding=1)
        self.dilation_conv5 = nn.Conv2d(165, 256, 3, padding=1)
        self.dilation_adaptive = nn.AdaptiveAvgPool2d((1, 1))
        self.dilation_head = nn.Linear(256, 1)

        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = self.pool(self.act(self.conv3(x)))

        g = self.pool(self.act(self.gaze_conv4(x)))
        g = self.pool(self.act(self.gaze_conv5(g)))
        g = self.gaze_adaptive(g)
        g = torch.flatten(g, 1)
        gaze_out = self.sigmoid(self.gaze_head(g))

        e = self.pool(self.act(self.eyelid_conv4(x)))
        e = self.pool(self.act(self.eyelid_conv5(e)))
        e = self.eyelid_adaptive(e)
        e = torch.flatten(e, 1)
        eyelid_out = self.sigmoid(self.eyelid_head(e))

        p = self.pool(self.act(self.dilation_conv4(x)))
        p = self.pool(self.act(self.dilation_conv5(p)))
        p = self.dilation_adaptive(p)
        p = torch.flatten(p, 1)
        dilation_out = self.sigmoid(self.dilation_head(p))

        return gaze_out, eyelid_out, dilation_out

def train_model(model: EyeCNN, train_loader: DataLoader, val_loader: DataLoader, optimizer: Adam, criterion: Callable, epochs: int, eye_name: str, gaze_weight: float, eyelid_weight: float, dilation_weight: float):
    model.to(TARGET_DEVICE)
    for epoch in range(epochs):
        model.train()
        total_training_loss = 0.0
        for imgs, gaze_targets, eyelid_targets, dilation_targets in train_loader:
            imgs, gaze_targets, eyelid_targets, dilation_targets = imgs.to(TARGET_DEVICE), gaze_targets.to( TARGET_DEVICE), eyelid_targets.to(TARGET_DEVICE), dilation_targets.to(TARGET_DEVICE)
            optimizer.zero_grad()
            gaze_pred, eyelid_pred, dilation_pred = model(imgs)
            gaze_loss = criterion(gaze_pred, gaze_targets)
            eyelid_loss = criterion(eyelid_pred, eyelid_targets)
            dilation_loss = criterion(dilation_pred, dilation_targets)
            loss = gaze_weight * gaze_loss + eyelid_weight * eyelid_loss + dilation_weight * dilation_loss
            loss.backward()
            optimizer.step()
            total_training_loss += loss.item()

        avg_train_loss = total_training_loss / max(1, len(train_loader))

        if eye_name == "Left Eye":
            left_eye_graph.append(avg_train_loss)
        else:
            right_eye_graph.append(avg_train_loss)

        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for imgs, gaze_targets, eyelid_targets, dilation_targets in val_loader:
                    imgs, gaze_targets, eyelid_targets, dilation_targets = imgs.to(TARGET_DEVICE), gaze_targets.to(TARGET_DEVICE), eyelid_targets.to(TARGET_DEVICE), dilation_targets.to(TARGET_DEVICE)
                    optimizer.zero_grad()
                    gaze_pred, eyelid_pred, dilation_pred = model(imgs)
                    gaze_loss = criterion(gaze_pred, gaze_targets)
                    eyelid_loss = criterion(eyelid_pred, eyelid_targets)
                    dilation_loss = criterion(dilation_pred, dilation_targets)
                    loss = gaze_weight * gaze_loss + eyelid_weight * eyelid_loss + dilation_weight * dilation_loss
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / max(1, len(val_loader))

            if eye_name == "Left Eye":
                left_eye_val_graph.append(avg_val_loss)
            else:
                right_eye_val_graph.append(avg_val_loss)

        print(f"{eye_name} Epoch {epoch + 1}/{epochs} | Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")

def train_both_models(model_left: EyeCNN, model_right: EyeCNN, x_max_abs: float, y_max_abs: float, gaze_weight: float, eyelid_weight: float, dilation_weight: float):
    dataset_left = EyeGazeDataset(variables.DATA_DIR, x_max_abs, y_max_abs, "left", training_transform)
    dataset_right = EyeGazeDataset(variables.DATA_DIR, x_max_abs, y_max_abs, "right", training_transform)

    val_split = 0.2
    train_size_left = int((1 - val_split) * len(dataset_left))
    val_size_left = len(dataset_left) - train_size_left
    train_dataset_left, val_dataset_left = random_split(dataset_left, [train_size_left, val_size_left])

    train_size_right = int((1 - val_split) * len(dataset_right))
    val_size_right = len(dataset_right) - train_size_right
    train_dataset_right, val_dataset_right = random_split(dataset_right, [train_size_right, val_size_right])

    train_loader_left = DataLoader(train_dataset_left, batch_size=variables.BATCH_SIZE, shuffle=True)
    val_loader_left = DataLoader(val_dataset_left, batch_size=variables.BATCH_SIZE, shuffle=False)

    train_loader_right = DataLoader(train_dataset_right, batch_size=variables.BATCH_SIZE, shuffle=True)
    val_loader_right = DataLoader(val_dataset_right, batch_size=variables.BATCH_SIZE, shuffle=False)

    opt_l = torch.optim.Adam(model_left.parameters(), lr=variables.LR)
    opt_r = torch.optim.Adam(model_right.parameters(), lr=variables.LR)

    t1 = threading.Thread(target=train_model,
                          args=(model_left, train_loader_left, val_loader_left, opt_l, nn.MSELoss(), variables.EPOCHS, "Left Eye", gaze_weight, eyelid_weight, dilation_weight), name="TrainThread_Left")
    t2 = threading.Thread(target=train_model,
                          args=(model_right, train_loader_right, val_loader_right, opt_r, nn.MSELoss(), variables.EPOCHS, "Right Eye", gaze_weight, eyelid_weight, dilation_weight), name="TrainThread_Right")

    t1.start()
    t2.start()

    t1.join()
    t2.join()

def save_eye_models():
    torch.save(left_eye_model.state_dict(), variables.MODEL_LEFT_PATH)
    torch.save(right_eye_model.state_dict(), variables.MODEL_RIGHT_PATH)