"""Module 4: ECCM layer: jamming detection, frequency hopping and signal authentication."""
import numpy as np

class FrequencyHopper:
    def __init__(self, freqs=(9.0, 9.2, 9.4, 9.6), seed=0):      # GHz (X-band style example values)
        self.freqs, self.i, self.rng = list(freqs), 0, np.random.default_rng(seed)
    @property
    def current(self): return self.freqs[self.i]
    def hop_away_from(self, jammed):
        ok = [k for k, f in enumerate(self.freqs) if f not in jammed]
        if ok and self.freqs[self.i] in jammed: self.i = int(self.rng.choice(ok))
        return self.current

def detect_jamming(spectrum, freqs, k=5.0):
    """Frequencies whose power is far above the median noise floor are treated as jammed."""
    floor = np.median(spectrum); return {f for f, p in zip(freqs, spectrum) if p > k * floor}

def adaptive_filter(signal, win=5):
    """Simple adaptive clipping: limit samples that exceed 3x the local median (impulsive noise)."""
    sig = np.asarray(signal, float); med = np.array([np.median(sig[max(0, i - win):i + win + 1]) for i in range(len(sig))])
    return np.where(np.abs(sig) > 3 * (np.abs(med) + 1e-9), med, sig)

class PulseAuthenticator:
    """Every pulse carries a fresh random +/-1 code. A DRFM spoofer that replays an OLD pulse
    fails the correlation test with the code of the CURRENT pulse."""
    def __init__(self, length=128, thr=0.5, seed=0): self.L, self.thr, self.rng = length, thr, np.random.default_rng(seed)
    def new_pulse(self): self.code = self.rng.choice([-1.0, 1.0], self.L); return self.code
    def verify(self, echo): return float(np.dot(echo, self.code) / (np.linalg.norm(echo) * np.linalg.norm(self.code) + 1e-9)) >= self.thr
