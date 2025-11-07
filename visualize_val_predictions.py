#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import cv2

# --- Paths ---
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
VAL_DIR = DATASET_DIR / "quad_dataset" / "val"
SEG_DATASET_DIR = DATASET_DIR / "fairscan-dataset"
MODEL_PATH = ROOT / "build" / "fairscan-quadrilateral.tflite"
OUTPUT_DIR = ROOT / "build" / "quad_val_viz"

# --- Settings ---
INPUT_SIZE = (256, 256)
IMG_WIDTH = 400

# --- Output setup ---
if OUTPUT_DIR.exists():
    for f in OUTPUT_DIR.glob("*"):
        f.unlink()
else:
    OUTPUT_DIR.mkdir(parents=True)

def load_quad_from_txt(txt_path: Path):
    with open(txt_path, "r") as f:
        values = list(map(float, f.read().strip().split()))
    if len(values) != 8:
        raise ValueError(f"Expected 8 floats in {txt_path}")
    return np.array(values, dtype=np.float32).reshape(4, 2)

def preprocess_mask(mask_path: Path) -> np.ndarray:
    mask = np.load(mask_path).astype(np.float32)
    mask = cv2.resize(mask, (256, 256))
    mask = np.expand_dims(mask, axis=(0, 1))
    return mask

def run_inference(interpreter, mask_np: np.ndarray) -> np.ndarray:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details["index"], mask_np)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details["index"])
    return output_data.reshape(4, 2)

def overlay_mask_and_quads(orig_img: Image.Image, mask_path: Path, gt_quad: np.ndarray, pred_quad: np.ndarray):
    mask = Image.open(mask_path).convert("L").resize(orig_img.size)
    mask_color = ImageOps.colorize(mask, black="black", white="lime")
    composite = Image.blend(orig_img, mask_color, alpha=0.4)
    draw = ImageDraw.Draw(composite)

    w, h = orig_img.size
    gt_pts = [(x * w, y * h) for x, y in gt_quad]
    pred_pts = [(x * w, y * h) for x, y in pred_quad]

    # Draw ground truth (blue) and prediction (red)
    draw.line(gt_pts + [gt_pts[0]], fill="blue", width=3)
    draw.line(pred_pts + [pred_pts[0]], fill="red", width=3)
    return composite


# --- Load model ---
print(f"Loading TFLite model: {MODEL_PATH}")
# --- Load model ---
print(f"Loading TFLite model: {MODEL_PATH}")
interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()

# 🔍 Print input/output info
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input details:", input_details)
print("Output details:", output_details)

html_lines = [
    "<html><head><meta charset='utf-8'><title>Quadrilateral Validation Results</title>",
    "<style>body{font-family:sans-serif;background:#111;color:#ddd;} img{width:400px;margin:4px;border-radius:8px;box-shadow:0 0 6px #000;} </style></head><body>",
    "<h1>FairScan Quadrilateral Model — Validation Visualization</h1>",
    "<p><span style='color:red;'>Red</span> = Predicted | <span style='color:blue;'>Blue</span> = Ground Truth</p>"
]

for mask_path in sorted(VAL_DIR.glob("*.npy")):
    txt_path = mask_path.with_suffix(".txt")
    if not txt_path.exists():
        continue
    image_path = (SEG_DATASET_DIR / "val" / "images" / f"{mask_path.stem}.jpg")
    if not image_path.exists():
        continue

    try:
        gt_quad = load_quad_from_txt(txt_path)
        mask_np = preprocess_mask(mask_path)
        pred_quad = run_inference(interpreter, mask_np)
        orig_img = Image.open(image_path).convert("RGB")
        composite = overlay_mask_and_quads(orig_img, mask_path.with_suffix(".png"), gt_quad, pred_quad)
    except Exception as e:
        print(f"❌ Failed for {mask_path.name}: {e}")
        continue

    out_path = OUTPUT_DIR / f"{mask_path.stem}.png"
    composite.save(out_path)
    html_lines.append(f"<img src='{out_path.name}' alt='{mask_path.stem}'>")

html_lines.append("</body></html>")
html_path = OUTPUT_DIR / "index.html"
html_path.write_text("\n".join(html_lines), encoding="utf-8")

print(f"✅ Visualization generated in {OUTPUT_DIR}")
print(f"👉 Open {html_path} in your browser.")
