# ARIA: Adaptive Radar Intelligence Architecture: Prototype Code

Python/TensorFlow code for the 5 modules in the thesis. Hyperparameters come from Tables 4.3 to 4.5
(CNN: 3 conv layers 3x3, ReLU, MaxPool 2x2, dropout 0.5, Adam 0.001, batch 32, 50 epochs;
LSTM: 20 steps, 2 layers x 64 units, dropout 0.2, Adam 0.001, batch 16, 100 epochs; split 70/15/15).

## Files (module -> file)
| Thesis module | File |
|---|---|
| 1. CNN Target Classification | `aria/cnn_model.py`, `train_cnn.py` |
| 2. LSTM Predictive Tracking | `aria/lstm_model.py`, `train_lstm.py` |
| 3. Distributed Sensor Mesh | `aria/sensor_mesh.py` |
| 4. ECCM Layer | `aria/eccm.py` |
| 5. Threat Assessment Engine | `aria/threat_engine.py` |
| Whole pipeline | `aria/pipeline.py`, `run_demo.py` |
| Preprocessing (normalise, augment, split) | `aria/preprocessing.py` |
| Settings | `aria/config.py` |

## Run
```bash
pip install -r requirements.txt
python run_demo.py                 # quick end-to-end demo (trains tiny models first, a few minutes)
python train_cnn.py                # full CNN training (50 epochs) on synthetic data
python train_lstm.py               # full LSTM training (100 epochs) on synthetic trajectories
python train_cnn.py --data data/mstar   # REAL images (see data/mstar/README.txt)
```
Works on Google Colab too (upload the folder, `!pip install -r requirements.txt`, run the same commands).
Outputs: `models/*.keras`, `results/cnn_loss.png`, `cnn_confusion.png`, `lstm_prediction.png`.

## Important: be honest in viva
* Default data is **synthetic** (generated shapes + noise). It lets the code run anywhere but is NOT real radar data.
  Numbers printed by the demo are **not** the thesis numbers (94.2%, 89.5% ...).
* To report real results, download MSTAR / OpenSARShip yourself, put images in class folders and retrain.
  Your accuracy will be whatever the real run gives.
* The Threat Engine weights, the 3 m sensor noise, 0.1 s time step and the ECCM models are simple
  illustrative assumptions (the thesis gives no formulas for them).
* This is a research prototype of the architecture, not software for a live radar.
