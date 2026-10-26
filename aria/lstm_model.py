"""Module 2: LSTM-based Predictive Tracking (Table 4.5)."""
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from .config import LSTM

INPUT_SCALE = np.array([1000, 1000, 300, 300], np.float32)   # metres, metres, m/s, m/s
Y_SCALE = 500.0                                               # metres

def build_lstm(n_features=4):
    m = keras.Sequential([layers.Input((LSTM["seq_len"], n_features))])
    for i in range(LSTM["layers"]):
        m.add(layers.LSTM(LSTM["hidden"], return_sequences=(i < LSTM["layers"] - 1), dropout=LSTM["dropout"]))
    m.add(layers.Dense(32, activation="relu")); m.add(layers.Dense(2))               # future (dx, dy) in scaled units
    m.compile(optimizer=keras.optimizers.Adam(LSTM["lr"]), loss="mse")                # Mean Squared Error
    return m
