"""End-to-end ARIA demo (synthetic data). Trains tiny models if none exist, then shows all 5 modules.
   python run_demo.py            (first run trains quickly, takes a few minutes)"""
import os, subprocess, sys, numpy as np
from aria.config import CLASSES
from aria.synthetic_data import make_sar_image
from aria.sensor_mesh import Node, SensorMesh
from aria.eccm import FrequencyHopper, PulseAuthenticator, detect_jamming
from aria.pipeline import ARIA

if not (os.path.exists("models/cnn.keras") and os.path.exists("models/lstm.keras")):
    print(">> No trained models found: quick training on synthetic data...")
    subprocess.run([sys.executable, "train_cnn.py", "--epochs", "8", "--n-per-class", "150"], check=True)
    subprocess.run([sys.executable, "train_lstm.py", "--epochs", "10", "--n", "2000"], check=True)

print("\n=== Module 3: Distributed Sensor Mesh ===")
nodes = [Node("Node 1", (0, 0)), Node("Node 2", (-3000, 3000)), Node("Node 3", (3500, -3000))]
mesh = SensorMesh(nodes, hill_center=(2000, 1500), hill_radius=900)
behind_hill = (4000, 3000)
_, seen = mesh.fuse(behind_hill); print("Target behind hill is seen by:", seen)
nodes[0].alive = False; _, seen = mesh.fuse((-1000, 500)); print("Node 1 destroyed -> still seen by:", seen)
nodes[0].alive = True

print("\n=== Module 4: ECCM ===")
hop = FrequencyHopper(); freqs = hop.freqs; spec = np.random.default_rng(1).uniform(.8, 1.2, len(freqs)); spec[freqs.index(hop.current)] = 30
jam = detect_jamming(spec, freqs); print(f"Jamming detected on {sorted(jam)} GHz; radar hops {hop.current} -> {hop.hop_away_from(jam)} GHz")
au = PulseAuthenticator(); old = au.new_pulse(); cur = au.new_pulse()
echo = cur + np.random.default_rng(2).normal(0, .5, au.L)
print("Real echo accepted:", au.verify(echo), "| Replayed (spoof) echo accepted:", au.verify(old))

print("\n=== Modules 1, 2, 5: CNN + LSTM + Threat Engine ===")
from aria.synthetic_data import make_trajectories
aria = ARIA(); Xt, Yt = make_trajectories(3, seed=7); rng = np.random.default_rng(5)
rows = []
for k, cls_id in enumerate([0, 1, 2]):                     # Aircraft, UAV, Ship images
    img = make_sar_image(cls_id, rng); last = rng.uniform(-8000, 8000, 2)
    rows.append(aria.process_target(img, Xt[k], last))
print(f"{'#':<3}{'Class':<10}{'Conf':>6}{'Speed m/s':>11}{'Threat':>8}  {'Category':<8} Action")
for i, r in enumerate(sorted(rows, key=lambda r: -r['score']), 1):
    print(f"{i:<3}{r['cls']:<10}{r['conf']*100:>5.0f}%{r['speed']:>11.0f}{r['score']:>8}  {r['category']:<8} {r['action']}")
print("\nDone. (Synthetic data only: numbers are NOT thesis results.)")
