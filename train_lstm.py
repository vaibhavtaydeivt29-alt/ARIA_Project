"""Train + evaluate the LSTM trajectory predictor (synthetic trajectories by default).
   For real data, replace make_trajectories() with your own (N,20,4) sequences and (N,2) targets."""
import argparse, os, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras
from aria.config import LSTM, SEED
from aria.synthetic_data import make_trajectories
from aria.lstm_model import build_lstm, INPUT_SCALE, Y_SCALE

ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=LSTM["epochs"]); ap.add_argument("--n", type=int, default=3000); a = ap.parse_args()
keras.utils.set_random_seed(SEED); os.makedirs("models", exist_ok=True); os.makedirs("results", exist_ok=True)
X, Y = make_trajectories(a.n); n = len(X); i1, i2 = int(.7 * n), int(.85 * n)
Xs, Ys = X / INPUT_SCALE, Y / Y_SCALE
m = build_lstm(); m.summary()
h = m.fit(Xs[:i1], Ys[:i1], validation_data=(Xs[i1:i2], Ys[i1:i2]), epochs=a.epochs, batch_size=LSTM["batch_size"], verbose=2)
pred = m.predict(Xs[i2:], batch_size=LSTM["eval_batch_size"], verbose=0) * Y_SCALE
err = np.linalg.norm(pred - Y[i2:], axis=1)
print(f"\nTEST  mean position error = {err.mean():.1f} m   median = {np.median(err):.1f} m   (horizon 1.2 s)")
m.save("models/lstm.keras")
k = 0; plt.figure(); plt.plot(*X[i2+k, :, :2].T, "o-", label="Observed path"); plt.plot(*Y[i2+k], "g*", ms=14, label="Actual (t+1.2 s)"); plt.plot(*pred[k], "rx", ms=10, label="LSTM prediction")
plt.legend(); plt.axis("equal"); plt.title("LSTM Prediction (relative to last position)"); plt.savefig("results/lstm_prediction.png", dpi=150)
