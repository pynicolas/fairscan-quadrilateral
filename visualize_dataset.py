#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

# --- Paths ---
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
QUAD_DATASET_DIR = DATASET_DIR / "quad_dataset"
SEG_DATASET_DIR = DATASET_DIR / "fairscan-dataset"
OUTPUT_DIR = ROOT / "build" / "visualize_dataset"

# --- Output setup ---
if OUTPUT_DIR.exists():
    for f in OUTPUT_DIR.glob("*"):
        f.unlink()
else:
    OUTPUT_DIR.mkdir(parents=True)

def load_quad_from_txt(txt_path: Path):
    """Read normalized quad coordinates and return np.ndarray (4,2)."""
    with open(txt_path, "r") as f:
        values = list(map(float, f.read().strip().split()))
    assert len(values) == 8, f"Expected 8 floats in {txt_path}"
    pts = np.array(values, dtype=np.float32).reshape(4, 2)
    return pts

def find_original_image_path(mask_path: Path, split: str) -> Path:
    """Find the original image (jpg) matching this mask."""
    orig_path = (SEG_DATASET_DIR / split / "images" / f"{mask_path.stem}.jpg")
    if not orig_path.exists():
        raise FileNotFoundError(f"Original image not found for {mask_path}")
    return orig_path

def overlay_mask_and_quad(orig_img: Image.Image, mask_path: Path, quad_pts: np.ndarray) -> Image.Image:
    """Combine original RGB image, mask, and quad annotation into one composite image."""
    mask = Image.open(mask_path).convert("L").resize(orig_img.size)
    mask_color = ImageOps.colorize(mask, black="black", white="lime")

    composite = Image.blend(orig_img, mask_color, alpha=0.4)
    draw = ImageDraw.Draw(composite)

    # Scale normalized quad points to image coordinates
    w, h = orig_img.size
    pts_scaled = [(float(x) * w, float(y) * h) for x, y in quad_pts]
    draw.line(pts_scaled + [pts_scaled[0]], fill="red", width=3)

    return composite

html_lines = [
    "<html><head><meta charset='utf-8'><title>Quadrilateral Dataset Visualization</title>",
    "<style>body{font-family:sans-serif;background:#111;color:#ddd;} img{width:400px;margin:4px;border-radius:8px;box-shadow:0 0 6px #000;} .split{margin-top:2em;} </style></head><body>",
    "<h1>FairScan Quadrilateral Dataset Visualization</h1>"
]

for split in ["train", "val"]:
    html_lines.append(f"<div class='split'><h2>{split}</h2>")
    for mask_path in sorted((QUAD_DATASET_DIR / split).glob("*.png")):
        txt_path = mask_path.with_suffix(".txt")
        if not txt_path.exists():
            print(f"⚠ Missing {txt_path}")
            continue

        try:
            quad_pts = load_quad_from_txt(txt_path)
            orig_path = find_original_image_path(mask_path, split)
            orig_img = Image.open(orig_path).convert("RGB")
            composite = overlay_mask_and_quad(orig_img, mask_path, quad_pts)
        except Exception as e:
            print(f"❌ Failed for {mask_path.name}: {e}")
            continue

        out_path = OUTPUT_DIR / f"{split}_{mask_path.stem}.png"
        composite.save(out_path)
        html_lines.append(f"<img src='{out_path.name}' alt='{mask_path.stem}'>")

    html_lines.append("</div>")

html_lines.append("</body></html>")

html_path = OUTPUT_DIR / "index.html"
html_path.write_text("\n".join(html_lines), encoding="utf-8")

print(f"✅ Visualization generated in {OUTPUT_DIR}")
print(f"👉 Open {html_path} in your browser.")
