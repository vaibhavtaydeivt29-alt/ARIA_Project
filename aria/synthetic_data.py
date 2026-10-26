"""Synthetic stand-in data so the whole project runs WITHOUT MSTAR / OpenSARShip.
These are NOT real radar data. Use data_loader.py with real datasets for real results."""
import numpy as np
from scipy.ndimage import rotate, gaussian_filter
from .config import IMG_SIZE, NUM_CLASSES, DT, HORIZON_STEPS, LSTM

def _shape(cls, rng):
    img = np.zeros((IMG_SIZE, IMG_SIZE), np.float32); c = IMG_SIZE // 2
    if cls == 0:      # Aircraft: fuselage + wings (cross)
        img[c-30:c+30, c-3:c+3] = 1; img[c-6:c-2, c-28:c+28] = 1; img[c+22:c+25, c-10:c+10] = 1
    elif cls == 1:    # UAV: small X
        for k in range(-12, 13): img[c+k-1:c+k+1, c+k-1:c+k+1] = 1; img[c+k-1:c+k+1, c-k-1:c-k+1] = 1
    elif cls == 2:    # Ship: long bar with bright spots
        img[c-4:c+4, c-40:c+40] = .7; img[c-4:c+4, c-40:c-30] = 1; img[c-4:c+4, c+10:c+14] = 1
    elif cls == 3:    # Ground vehicle: compact block + turret
        img[c-10:c+10, c-16:c+16] = .8; img[c-5:c+5, c+4:c+22] = 1
    else:             # Unknown: scattered blobs
        for _ in range(rng.integers(3, 7)):
            y, x = rng.integers(20, IMG_SIZE-20, 2); img[y-4:y+4, x-4:x+4] = rng.uniform(.5, 1)
    return img

def make_sar_image(cls, rng):
    img = _shape(cls, rng)
    img = rotate(img, rng.uniform(0, 360), reshape=False, order=1)
    img = np.roll(img, rng.integers(-10, 10, 2), axis=(0, 1))
    speckle = rng.rayleigh(0.25, img.shape).astype(np.float32)     # SAR-like speckle noise
    img = gaussian_filter(img, 1.0) * 1.2 + speckle
    return np.clip(img / img.max(), 0, 1)[..., None].astype(np.float32)

def make_image_dataset(n_per_class=200, seed=0):
    rng = np.random.default_rng(seed)
    X = np.stack([make_sar_image(c, rng) for c in range(NUM_CLASSES) for _ in range(n_per_class)])
    y = np.repeat(np.arange(NUM_CLASSES), n_per_class)
    p = rng.permutation(len(y)); return X[p], y[p]

def make_trajectories(n=3000, seed=0):
    """Targets move with constant speed + random gentle turns. Returns (X, y, last_pos).
    X: (n, 20, 4) = [x, y, vx, vy] relative to the last observed position.
    y: (n, 2) = displacement after the 1.2 s horizon, in metres."""
    rng = np.random.default_rng(seed); L = LSTM["seq_len"]; T = L + HORIZON_STEPS
    X, Y = [], []
    for _ in range(n):
        speed = rng.uniform(50, 300); ang = rng.uniform(0, 2*np.pi); turn = rng.normal(0, 0.05)
        pos = rng.uniform(-5000, 5000, 2); path = []
        for _ in range(T):
            ang += turn * DT + rng.normal(0, 0.002)
            v = speed * np.array([np.cos(ang), np.sin(ang)]); pos = pos + v * DT; path.append(np.r_[pos, v])
        path = np.array(path); obs = path[:L].copy(); obs[:, :2] += rng.normal(0, 3, (L, 2))   # 3 m sensor noise
        last = obs[-1, :2].copy(); obs[:, :2] -= last
        X.append(obs); Y.append(path[-1, :2] - last)
    return np.array(X, np.float32), np.array(Y, np.float32)
