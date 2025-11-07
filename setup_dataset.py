#!/usr/bin/env python3
from pathlib import Path
import os
import shutil
import zipfile
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import tensorflow as tf
from urllib.request import urlretrieve
import csv

DATASET_VERSION = "v2.0"
DATASET_ZIP_URL = f'https://github.com/pynicolas/fairscan-dataset/releases/download/{DATASET_VERSION}/fairscan-dataset-{DATASET_VERSION}.zip'
DATASET_TOP_DIR = Path("dataset")
FAIRSCAN_DATASET_ZIP_PATH = DATASET_TOP_DIR / "fairscan-dataset.zip"
SEG_DATASET_DIR = DATASET_TOP_DIR / "fairscan-dataset"
QUAD_DATASET_DIR = DATASET_TOP_DIR / "quad_dataset"
FILTERED_OUT_DIR = DATASET_TOP_DIR / "filtered_out"

SEG_MODEL_URL = "https://github.com/pynicolas/fairscan-segmentation-model/releases/download/v1.1.0/fairscan-segmentation-model.tflite"
SEG_MODEL_FILE_PATH = DATASET_TOP_DIR / "fairscan-segmentation-model.tflite"

INPUT_WIDTH = 256
INPUT_HEIGHT = 256
DICE_THRESHOLD = 0.95

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((INPUT_WIDTH, INPUT_HEIGHT), Image.BILINEAR)
    img_np = np.asarray(img).astype(np.float32)
    img_np = (img_np - 127.5) / 127.5  # Normalize to [-1, 1]
    return np.expand_dims(img_np, axis=0)

def postprocess_output(output: np.ndarray) -> np.ndarray:
    output = np.squeeze(output).astype(np.float32)  # Shape: (256, 256)
    output = np.clip(output, 0, 1)
    return output  # float32 array, values in [0,1]

def get_segmentation_mask(image_path):
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    input_tensor = preprocess_image(img)
    interpreter.set_tensor(input_details['index'], input_tensor)
    interpreter.invoke()
    output_tensor = interpreter.get_tensor(output_details['index'])
    return postprocess_output(output_tensor)

def load_ground_truth_mask(mask_path):
    mask = Image.open(mask_path).convert("L").resize((INPUT_WIDTH, INPUT_HEIGHT))
    mask_np = np.asarray(mask).astype(np.float32) / 255.0
    return np.clip(mask_np, 0, 1)

def dice_score(mask1, mask2):
    mask1 = (mask1 > 0.5).astype(np.float32)
    mask2 = (mask2 > 0.5).astype(np.float32)
    intersection = np.sum(mask1 * mask2)
    denom = np.sum(mask1) + np.sum(mask2)
    if denom == 0:
        return 1.0  # both empty
    return (2.0 * intersection) / denom

print('Prepare dataset directory...')
if os.path.isdir(DATASET_TOP_DIR):
    shutil.rmtree(DATASET_TOP_DIR)
os.makedirs(DATASET_TOP_DIR, exist_ok=True)
os.makedirs(QUAD_DATASET_DIR, exist_ok=True)
os.makedirs(FILTERED_OUT_DIR, exist_ok=True)

print('Download and extract dataset...')
urlretrieve(DATASET_ZIP_URL, FAIRSCAN_DATASET_ZIP_PATH)
with zipfile.ZipFile(FAIRSCAN_DATASET_ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(DATASET_TOP_DIR)

print('Download and initialize segmentation model...')
urlretrieve(SEG_MODEL_URL, SEG_MODEL_FILE_PATH)
interpreter = tf.lite.Interpreter(model_path=str(SEG_MODEL_FILE_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

csv_path = DATASET_TOP_DIR / "dice_scores.csv"
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["subset", "image", "dice_score", "kept"])

print('Assemble dataset...')
for subdir_name in ["train", "val"]:
    img_input_dir = (SEG_DATASET_DIR / subdir_name) / "images"
    mask_input_dir = (SEG_DATASET_DIR / subdir_name) / "masks"
    quad_input_dir = (SEG_DATASET_DIR / subdir_name) / "quads"

    output_dir = QUAD_DATASET_DIR / subdir_name
    filtered_dir = FILTERED_OUT_DIR / subdir_name
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(filtered_dir, exist_ok=True)

    for image_path in sorted(img_input_dir.glob("*.jpg")):
        mask_path = mask_input_dir / (image_path.stem + ".png")
        quad_path = quad_input_dir / (image_path.stem + ".txt")

        if not quad_path.exists():
            print(f"Skipping {image_path.name}: missing quad")
            continue

        pred_mask = get_segmentation_mask(image_path)
        if mask_path.exists():
            gt_mask = load_ground_truth_mask(mask_path)
            dice = dice_score(gt_mask, pred_mask)
            dice_str = f"{dice:.4f}"
            keep = dice >= DICE_THRESHOLD
        else:
            print(f"Warning: no ground truth mask {mask_path}")
            dice_str = "unknown: no ground truth"
            keep = True
        csv_writer.writerow([subdir_name, image_path.name, dice_str, "yes" if keep else "no"])

        # Choose target dir
        target_dir = output_dir if keep else filtered_dir

        # Save predicted mask
        np.save(target_dir / (image_path.stem + ".npy"), pred_mask)
        Image.fromarray((pred_mask * 255).astype(np.uint8)).save(target_dir / (image_path.stem + ".png"))

        # Copy quad annotation
        shutil.copy2(quad_path, target_dir / (image_path.stem + ".txt"))

csv_file.close()
print(f"Done. Dice scores written to {csv_path}")
