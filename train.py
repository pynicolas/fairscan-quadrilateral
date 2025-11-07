#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import glob
import os
import shutil
import time

from model import QuadRegressorLite

QUAD_DATASET_PATH = "dataset/quad_dataset"
BUILD_DIR = "build"
MODEL_FILE_PATH = f"{BUILD_DIR}/fairscan-quadrilateral.pt"
TFLITE_MODEL_FILE_PATH = f"{BUILD_DIR}/fairscan-quadrilateral.tflite"
EPOCHS = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.isdir(BUILD_DIR):
    shutil.rmtree(BUILD_DIR)
os.makedirs(BUILD_DIR)
shutil.copy("LICENSE", f"{BUILD_DIR}/LICENSE.txt")

def load_quad_from_txt(txt_path):
    """
    Reads a .txt file with 8 normalized coordinates (x1 y1 x2 y2 x3 y3 x4 y4)
    and returns a (4, 2) numpy array sorted in clockwise order starting from top-left.
    """
    with open(txt_path, "r") as f:
        parts = f.read().strip().split()
    
    if len(parts) != 8:
        raise ValueError(f"Expected 8 numbers in {txt_path}, got {len(parts)}")

    coords = np.array(list(map(float, parts))).reshape(-1, 2)

    # Sort the points in clockwise order
    # Compute centroid
    cx, cy = coords.mean(axis=0)
    
    # Compute angles relative to the centroid
    angles = np.arctan2(coords[:,1] - cy, coords[:,0] - cx)
    
    # Sort by angle (counter-clockwise) and reorder to start from top-left
    sorted_idx = np.argsort(angles)
    sorted_coords = coords[sorted_idx]

    # Ensure top-left first (smallest y + x)
    top_left_idx = np.argmin(sorted_coords[:,0] + sorted_coords[:,1])
    sorted_coords = np.roll(sorted_coords, -top_left_idx, axis=0)

    return sorted_coords


class QuadDataset(Dataset):
    def __init__(self, folder):
        self.masks = sorted(glob.glob(os.path.join(folder, "*.npy")))

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        mask_path = self.masks[idx]
        pts_path = mask_path.replace(".npy", ".txt")

        mask = np.load(mask_path).astype(np.float32)
        mask = cv2.resize(mask, (256, 256))
        mask = np.expand_dims(mask, 0)  # (1, 256, 256)

        points = load_quad_from_txt(pts_path)
        points = points.reshape(-1)

        return torch.tensor(mask), torch.tensor(points, dtype=torch.float32)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = criterion(pred, y.view(y.size(0), -1))
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

train_ds = QuadDataset(f"{QUAD_DATASET_PATH}/train")
val_ds = QuadDataset(f"{QUAD_DATASET_PATH}/val")

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32)

model = QuadRegressorLite().to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
best_val_loss = float("inf")
best_state = None

for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    val_loss = evaluate(model, val_loader, criterion)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict()
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch}: train_loss={total_loss / len(train_ds):.5f}, val_loss={val_loss:.5f}, time={elapsed_time:.1f}s")

print(f"Best validation loss: {best_val_loss:.5f}")
torch.save(best_state, MODEL_FILE_PATH)

# Convert to TFLite

import ai_edge_torch

model = QuadRegressorLite().to(DEVICE)
model.load_state_dict(torch.load(MODEL_FILE_PATH, map_location="cpu"))

sample_inputs = (torch.randn(1, 1, 256, 256),)
edge_model = ai_edge_torch.convert(model.eval(), sample_inputs)
edge_model.export(TFLITE_MODEL_FILE_PATH)
