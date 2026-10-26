"""Train + evaluate the CNN target classifier.
   python train_cnn.py                       -> synthetic demo data
   python train_cnn.py --data data/mstar     -> real images in class folders"""
import argparse, os, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from aria.config import CLASSES, CNN, NUM_CLASSES, SEED
from aria.preprocessing import normalise, augment_set, split_data
from aria.cnn_model import build_cnn

ap = argparse.ArgumentParser(); ap.add_argument("--data", default="synthetic"); ap.add_argument("--epochs", type=int, default=CNN["epochs"])
ap.add_argument("--n-per-class", type=int, default=300); a = ap.parse_args()
keras.utils.set_random_seed(SEED); os.makedirs("models", exist_ok=True); os.makedirs("results", exist_ok=True)
if a.data == "synthetic":
    from aria.synthetic_data import make_image_dataset; X, y = make_image_dataset(a.n_per_class)
else:
    from aria.data_loader import load_image_folder; X, y = load_image_folder(a.data)
X = normalise(X); (Xtr, ytr), (Xva, yva), (Xte, yte) = split_data(X, y)
Xtr, ytr = augment_set(Xtr, ytr, copies=1)
oh = lambda v: keras.utils.to_categorical(v, NUM_CLASSES)
m = build_cnn(); m.summary()
h = m.fit(Xtr, oh(ytr), validation_data=(Xva, oh(yva)), epochs=a.epochs, batch_size=CNN["batch_size"], verbose=2)
pred = m.predict(Xte, verbose=0).argmax(1)
acc = accuracy_score(yte, pred); p, r, f, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
print(f"\nTEST  accuracy={acc*100:.1f}%  precision={p*100:.1f}%  recall={r*100:.1f}%  F1={f*100:.1f}%")
m.save("models/cnn.keras")
plt.figure(); plt.plot(h.history["loss"], label="Training loss"); plt.plot(h.history["val_loss"], "--", label="Validation loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("CNN Training and Validation Loss"); plt.savefig("results/cnn_loss.png", dpi=150); plt.close()
cm = confusion_matrix(yte, pred, labels=range(NUM_CLASSES), normalize="true") * 100
plt.figure(figsize=(6, 5)); plt.imshow(cm, cmap="Blues"); plt.xticks(range(NUM_CLASSES), CLASSES, rotation=30); plt.yticks(range(NUM_CLASSES), CLASSES)
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES): plt.text(j, i, f"{cm[i,j]:.0f}", ha="center", va="center")
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix (%)"); plt.tight_layout(); plt.savefig("results/cnn_confusion.png", dpi=150)
