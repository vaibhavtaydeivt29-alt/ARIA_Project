"""Data normalisation, augmentation and 70/15/15 split (thesis: Data Preprocessing Pipeline)."""
import numpy as np
from scipy.ndimage import rotate, zoom
from sklearn.model_selection import train_test_split
from .config import SPLIT, SEED, IMG_SIZE

def normalise(X):
    """Scale every image to the 0-1 range."""
    X = X.astype(np.float32); mn = X.min(axis=(1, 2, 3), keepdims=True); mx = X.max(axis=(1, 2, 3), keepdims=True)
    return (X - mn) / (mx - mn + 1e-8)

def augment(img, rng):
    """Rotate, scale, flip, crop and add noise (thesis: Data Augmentation)."""
    a = rotate(img, rng.uniform(-20, 20), reshape=False, order=1, mode="nearest")
    if rng.random() < .5: a = a[:, ::-1]
    s = rng.uniform(.9, 1.1); z = zoom(a, (s, s, 1), order=1)
    out = np.zeros_like(a); h = min(IMG_SIZE, z.shape[0]); o = (IMG_SIZE - h) // 2; zo = (z.shape[0] - h) // 2
    out[o:o+h, o:o+h] = z[zo:zo+h, zo:zo+h]
    return np.clip(out + rng.normal(0, .03, out.shape), 0, 1).astype(np.float32)

def augment_set(X, y, copies=1, seed=SEED):
    rng = np.random.default_rng(seed)
    Xa = [augment(x, rng) for _ in range(copies) for x in X]
    return np.concatenate([X, np.stack(Xa)]), np.concatenate([y] + [y] * copies)

def split_data(X, y, seed=SEED):
    """70% train / 15% validation / 15% test (stratified)."""
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=1 - SPLIT["train"], stratify=y, random_state=seed)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=SPLIT["test"] / (SPLIT["val"] + SPLIT["test"]), stratify=ytmp, random_state=seed)
    return (Xtr, ytr), (Xva, yva), (Xte, yte)
