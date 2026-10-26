"""Module 1: CNN-based Target Classification (Table 4.4)."""
from tensorflow import keras
from tensorflow.keras import layers
from .config import IMG_SIZE, CHANNELS, NUM_CLASSES, CNN

def build_cnn():
    m = keras.Sequential([layers.Input((IMG_SIZE, IMG_SIZE, CHANNELS))])
    for f in (32, 64, 128)[:CNN["conv_layers"]]:
        m.add(layers.Conv2D(f, CNN["kernel"], activation="relu", padding="same"))   # feature extraction
        m.add(layers.MaxPooling2D(CNN["pool"]))                                       # dimensionality reduction
    m.add(layers.Flatten()); m.add(layers.Dense(128, activation="relu")); m.add(layers.Dropout(CNN["dropout"]))
    m.add(layers.Dense(NUM_CLASSES, activation="softmax"))                           # classification
    m.compile(optimizer=keras.optimizers.Adam(CNN["lr"]), loss="categorical_crossentropy", metrics=["accuracy"])
    return m
