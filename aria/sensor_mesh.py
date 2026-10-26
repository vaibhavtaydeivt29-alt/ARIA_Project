"""Module 3: Distributed Sensor Mesh. Several radar nodes see the same target; a data-fusion
centre combines them (inverse-variance weighting). Terrain (a hill) blocks line of sight."""
import numpy as np

class Node:
    def __init__(self, name, pos, rng_m=8000, sigma=15.0):
        self.name, self.pos, self.range, self.sigma, self.alive = name, np.array(pos, float), rng_m, sigma, True

class SensorMesh:
    def __init__(self, nodes, hill_center=None, hill_radius=0.0, seed=0):
        self.nodes, self.hc, self.hr, self.rng = nodes, (None if hill_center is None else np.array(hill_center, float)), hill_radius, np.random.default_rng(seed)

    def _blocked(self, a, b):            # does the segment a->b pass through the hill?
        if self.hc is None: return False
        d = b - a; t = np.clip(np.dot(self.hc - a, d) / (np.dot(d, d) + 1e-9), 0, 1)
        return np.linalg.norm(a + t * d - self.hc) < self.hr

    def visible_nodes(self, target):
        t = np.array(target, float)
        return [n for n in self.nodes if n.alive and np.linalg.norm(t - n.pos) <= n.range and not self._blocked(n.pos, t)]

    def fuse(self, target):
        """Return (fused_position or None, names of nodes that saw the target)."""
        vis = self.visible_nodes(target)
        if not vis: return None, []
        meas = [np.array(target) + self.rng.normal(0, n.sigma, 2) for n in vis]
        w = np.array([1 / n.sigma ** 2 for n in vis]); fused = (np.array(meas) * w[:, None]).sum(0) / w.sum()
        return fused, [n.name for n in vis]
