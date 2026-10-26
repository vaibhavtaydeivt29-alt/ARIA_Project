"""All settings come from the thesis (Chapter 4: Table 4.3, 4.4, 4.5)."""
SEED = 42
CLASSES = ["Aircraft", "UAV", "Ship", "Ground Vehicle", "Unknown"]   # 5 target categories
NUM_CLASSES = len(CLASSES)

# Dataset split (Table 4.3)
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}

# CNN (Table 4.4)
IMG_SIZE = 128
CHANNELS = 1                      # SAR images are grayscale
CNN = dict(conv_layers=3, kernel=(3, 3), pool=(2, 2), dropout=0.5,
           optimizer="adam", batch_size=32, epochs=50, lr=0.001)

# LSTM (Table 4.5)
LSTM = dict(seq_len=20, hidden=64, layers=2, dropout=0.2, optimizer="adam",
            batch_size=16, eval_batch_size=64, epochs=100, lr=0.001)
DT = 0.1                          # seconds between radar observations (assumption)
HORIZON_STEPS = 12                # 12 x 0.1 s = 1.2 s prediction horizon (thesis Table 5.2)
