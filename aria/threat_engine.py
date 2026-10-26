"""Module 5: Autonomous Threat Assessment Engine.
NOTE: the thesis describes WHICH factors are used (class, speed, trajectory, proximity) but gives no
formula. The weights below are an illustrative design choice; change them freely."""
import numpy as np

CLASS_RISK = {"Aircraft": .85, "UAV": .65, "Ship": .30, "Ground Vehicle": .35, "Unknown": .75}
W = dict(cls=.35, speed=.20, proximity=.25, heading=.20)

def assess(cls, speed_mps, dist_m, heading_vec, to_asset_vec, max_speed=400.0, max_range=20000.0):
    h = np.array(heading_vec, float); a = np.array(to_asset_vec, float)
    cosang = float(np.dot(h, a) / (np.linalg.norm(h) * np.linalg.norm(a) + 1e-9))
    f = dict(cls=CLASS_RISK.get(cls, .75), speed=min(speed_mps / max_speed, 1), proximity=max(0, 1 - dist_m / max_range), heading=max(0, cosang))
    score = round(100 * sum(W[k] * f[k] for k in W))
    cat = "HIGH" if score >= 70 else "MEDIUM" if score >= 40 else "LOW"
    act = {"HIGH": "Alert operator and track first", "MEDIUM": "Monitor closely", "LOW": "Routine monitoring"}[cat]
    return dict(score=score, category=cat, action=act)
