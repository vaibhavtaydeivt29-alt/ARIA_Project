"""Load REAL datasets (MSTAR / OpenSARShip chips) arranged as image folders:
    data/<name>/<class_name>/*.png   e.g. data/mstar/Ground Vehicle/img001.png
Class folder names must match aria.config.CLASSES (unused classes can be missing)."""
import os, numpy as np
from PIL import Image
from .config import CLASSES, IMG_SIZE

def load_image_folder(root):
    X, y = [], []
    for ci, name in enumerate(CLASSES):
        d = os.path.join(root, name)
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                im = Image.open(os.path.join(d, f)).convert("L").resize((IMG_SIZE, IMG_SIZE))
                X.append(np.asarray(im, np.float32)[..., None] / 255.0); y.append(ci)
    if not X: raise FileNotFoundError(f"No images found under {root}. See README (Real data).")
    return np.stack(X), np.array(y)
