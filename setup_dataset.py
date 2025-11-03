from pathlib import Path
import os
import shutil
import zipfile
import numpy as np
from PIL import Image
from PIL import ImageOps
import tensorflow as tf
from urllib.request import urlretrieve

DATASET_VERSION = "v2.0"
DATASET_ZIP_URL = f'https://github.com/pynicolas/fairscan-dataset/releases/download/{DATASET_VERSION}/fairscan-dataset-{DATASET_VERSION}.zip'
DATASET_TOP_DIR = Path("dataset")
FAIRSCAN_DATASET_ZIP_PATH = DATASET_TOP_DIR / "fairscan-dataset.zip"
SEG_DATASET_DIR = DATASET_TOP_DIR / "fairscan-dataset"
QUAD_DATASET_DIR = DATASET_TOP_DIR / "quad_dataset"

SEG_MODEL_URL = "https://github.com/pynicolas/fairscan-segmentation-model/releases/download/v1.0.0/fairscan-segmentation-model.tflite"
SEG_MODEL_FILE_PATH = DATASET_TOP_DIR / "fairscan-segmentation-model.tflite"

INPUT_WIDTH = 256
INPUT_HEIGHT = 256

def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((INPUT_WIDTH, INPUT_HEIGHT), Image.BILINEAR)
    img_np = np.asarray(img).astype(np.float32)
    img_np = (img_np - 127.5) / 127.5  # Normalize to [-1, 1]
    return np.expand_dims(img_np, axis=0)  # Shape: (1, 256, 256, 3)

def postprocess_output(output: np.ndarray) -> Image.Image:
    output = np.squeeze(output)  # Shape: (256, 256)
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)
    return Image.fromarray(output, mode="L")  # "L" = 8-bit grayscale

def generate_segmentation_mask(image_path, output_dir):
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    input_tensor = preprocess_image(img)

    interpreter.set_tensor(input_details['index'], input_tensor)
    interpreter.invoke()
    output_tensor = interpreter.get_tensor(output_details['index'])

    mask = postprocess_output(output_tensor)
    mask.save(output_dir / (image_path.stem + ".png"))

print('Prepare dataset directory...')
if os.path.isdir(DATASET_TOP_DIR):
    shutil.rmtree(DATASET_TOP_DIR)
os.makedirs(DATASET_TOP_DIR, exist_ok=True)
os.makedirs(QUAD_DATASET_DIR, exist_ok=True)

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

print('Assemble dataset...')
for subdir_name in ["train", "val"]:
    img_input_dir = (SEG_DATASET_DIR / subdir_name) / "images"
    quad_input_dir = (SEG_DATASET_DIR / subdir_name) / "quads"
    output_dir = QUAD_DATASET_DIR / subdir_name
    os.makedirs(output_dir, exist_ok=True)
    for image_path in sorted(img_input_dir.glob("*.jpg")):
        quad_path = quad_input_dir / (image_path.stem + ".txt")
        if not quad_path.exists():
            print(f"{quad_path} not found")
            continue
        print(f"Adding {image_path} + {quad_path}")
        generate_segmentation_mask(image_path, output_dir)
        shutil.copy2(quad_path, output_dir / (image_path.stem + ".txt"))
