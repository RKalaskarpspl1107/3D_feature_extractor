"""Fast ray casting against the part's triangle mesh (vectorised Möller-Trumbore).

Exact B-spline ray intersection costs ~0.1-0.2 s per ray on large models;
the mesh answers "is anything in the way" in milliseconds, which is all
the accessibility / through / inside tests need.
"""

import numpy as np

from .viewer import mesh_arrays


class MeshRays:
    def __init__(self, faces):
        pos, idx, _ = mesh_arrays(faces)
        tri = pos[idx].astype(np.float64)           # (T, 3, 3)
        self.v0 = tri[:, 0]
        self.e1 = tri[:, 1] - self.v0
        self.e2 = tri[:, 2] - self.v0
        self.lo = tri.min(axis=1)
        self.hi = tri.max(axis=1)
        self._grids = {}

    GRID = 96

    def _grid(self, axis):
        """2D bucket grid over the two coordinates perpendicular to `axis`
        (built lazily): an axis-aligned ray only needs the triangles in its cell."""
        if axis in self._grids:
            return self._grids[axis]
        a, b = [k for k in range(3) if k != axis]
        glo = np.array([self.lo[:, a].min(), self.lo[:, b].min()])
        ghi = np.array([self.hi[:, a].max(), self.hi[:, b].max()])
        cell = np.maximum((ghi - glo) / self.GRID, 1e-9)
        i0 = np.clip(((self.lo[:, [a, b]] - glo) / cell).astype(int), 0, self.GRID - 1)
        i1 = np.clip(((self.hi[:, [a, b]] - glo) / cell).astype(int), 0, self.GRID - 1)
        buckets = [[[] for _ in range(self.GRID)] for _ in range(self.GRID)]
        for t, (x0, y0), (x1, y1) in zip(range(len(i0)), i0, i1):
            for x in range(x0, x1 + 1):
                row = buckets[x]
                for y in range(y0, y1 + 1):
                    row[y].append(t)
        grid = (a, b, glo, cell, [[np.array(c, dtype=np.int64) for c in row] for row in buckets])
        self._grids[axis] = grid
        return grid

    def _candidates(self, o, d):
        k = int(np.argmax(np.abs(d)))
        if abs(d[k]) < 1 - 1e-9:
            return None                      # not axis-aligned: test everything
        a, b, glo, cell, buckets = self._grid(k)
        x = int((o[a] - glo[0]) // cell[0])
        y = int((o[b] - glo[1]) // cell[1])
        if not (0 <= x < self.GRID and 0 <= y < self.GRID):
            return np.empty(0, dtype=np.int64)
        return buckets[x][y]

    def hits(self, origin, direction, t_max=np.inf, t_min=1e-6):
        """Distances along the ray at which it crosses the mesh (sorted)."""
        o = np.asarray(origin, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        pre = self._candidates(o, d)
        lo, hi = (self.lo, self.hi) if pre is None else (self.lo[pre], self.hi[pre])
        # cheap slab test against each triangle's bounding box first
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(np.abs(d) > 1e-12, 1.0 / d, np.inf)
            t1 = (lo - o) * inv
            t2 = (hi - o) * inv
        t1 = np.nan_to_num(t1, nan=-np.inf)
        t2 = np.nan_to_num(t2, nan=np.inf)
        tnear = np.minimum(t1, t2).max(axis=1)
        tfar = np.maximum(t1, t2).min(axis=1)
        cand = np.nonzero((tnear <= tfar + 1e-9) & (tfar >= t_min) & (tnear <= t_max))[0]
        if pre is not None:
            cand = pre[cand]
        if len(cand) == 0:
            return np.empty(0)
        e1, e2, v0 = self.e1[cand], self.e2[cand], self.v0[cand]
        p = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-12
        inv_det = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = o - v0
        u = np.einsum("ij,ij->i", s, p) * inv_det
        q = np.cross(s, e1)
        v = (q @ d) * inv_det
        t = np.einsum("ij,ij->i", e2, q) * inv_det
        mask = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > t_min) & (t < t_max)
        return np.sort(t[mask])

    def blocked(self, origin, direction, t_max=np.inf):
        return len(self.hits(origin, direction, t_max)) > 0

    def inside(self, point):
        """Parity vote over three skew directions (robust to a stray hit on a shared edge)."""
        votes = 0
        for d in ((0.267, 0.534, 0.802), (-0.816, 0.408, 0.408), (0.371, -0.928, 0.0)):
            t = self.hits(point, d)
            if len(t):
                # hits on a shared edge are reported once per triangle: merge near-duplicates
                t = t[np.concatenate(([True], np.diff(t) > 1e-6))]
            votes += len(t) % 2
        return votes >= 2
