"""ARIA end-to-end pipeline: ECCM -> Sensor Mesh -> CNN -> LSTM -> Threat Engine -> Operator."""
import numpy as np
from tensorflow import keras
from .config import CLASSES
from .lstm_model import INPUT_SCALE, Y_SCALE
from .threat_engine import assess

class ARIA:
    def __init__(self, cnn_path="models/cnn.keras", lstm_path="models/lstm.keras", mesh=None, hopper=None, auth=None):
        self.cnn, self.lstm = keras.models.load_model(cnn_path), keras.models.load_model(lstm_path)
        self.mesh, self.hopper, self.auth = mesh, hopper, auth

    def classify(self, img):                       # Module 1
        p = self.cnn.predict(img[None], verbose=0)[0]; k = int(p.argmax()); return CLASSES[k], float(p[k])

    def predict_future(self, obs):                 # Module 2: obs (20,4) relative to last position
        d = self.lstm.predict((obs / INPUT_SCALE)[None], verbose=0)[0] * Y_SCALE
        return d                                    # displacement in metres after 1.2 s

    def process_target(self, img, obs, last_pos, asset_pos=(0.0, 0.0)):
        cls, conf = self.classify(img); disp = self.predict_future(obs)
        vel = obs[-1, 2:4]; speed = float(np.linalg.norm(vel)); last = np.array(last_pos, float); to_asset = np.array(asset_pos) - last
        th = assess(cls, speed, float(np.linalg.norm(to_asset)), vel, to_asset)       # Module 5
        return dict(cls=cls, conf=conf, predicted_pos=last + disp, speed=speed, **th)
