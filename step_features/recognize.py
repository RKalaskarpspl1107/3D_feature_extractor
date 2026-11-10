"""Rule-based machining-feature recognition on analysed faces + adjacency.

Order matters: each recogniser claims ("consumes") the faces it explains
so later recognisers don't double-count them.
  holes -> slots -> pockets/steps -> bosses -> chamfers -> fillets/rounds
Anything left is reported as an unclassified surface, so nothing is lost.
"""

import math
from collections import defaultdict

import numpy as np
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepTools import BRepTools
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_IN, TopAbs_ON, TopAbs_SOLID, TopAbs_WIRE
from OCP.TopExp import TopExp
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape
from OCP.TopExp import TopExp_Explorer
from OCP.gp import gp_Pnt

from .raycast import MeshRays

AXIS_DIR_TOL = 1e-3        # |1 - |d1.d2||
AXIS_POS_TOL = 0.05        # mm, coaxial distance
RADIUS_TOL = 0.02          # mm
FULL_CIRCLE_DEG = 300.0

# Tap-drill and clearance diameters (ISO metric coarse) used for *hints* only:
# STEP geometry does not carry thread data, so these are inferences.
TAP_DRILL = {2.5: "M3", 3.3: "M4", 4.2: "M5", 5.0: "M6", 6.8: "M8", 8.5: "M10",
             10.2: "M12", 12.0: "M14", 14.0: "M16", 17.5: "M20", 21.0: "M24"}
# Some CAD systems model tapped holes at the thread's minor diameter instead.
THREAD_MINOR = {2.459: "M3", 3.242: "M4", 4.134: "M5", 4.917: "M6", 6.647: "M8", 8.376: "M10",
                10.106: "M12", 11.835: "M14", 13.835: "M16", 17.294: "M20", 20.752: "M24"}
CLEARANCE = {3.4: "M3", 4.5: "M4", 5.5: "M5", 6.6: "M6", 9.0: "M8", 11.0: "M10",
             13.5: "M12", 17.5: "M16", 22.0: "M20", 26.0: "M24"}


def _r(x, n=3):
    return None if x is None else round(float(x), n)


def _pt(v):
    return [round(float(c), 3) for c in v]


def canon_dir(d):
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d)
    k = int(np.argmax(np.abs(d)))
    return -d if d[k] < 0 else d


def _frame(d):
    e1 = np.cross(d, [1.0, 0, 0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(d, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(d, e1)


def angular_coverage(face_infos, loc, d, bin_deg=5):
    e1, e2 = _frame(d)
    bins = set()
    for f in face_infos:
        rel = f.points - loc
        ang = np.degrees(np.arctan2(rel @ e2, rel @ e1)) % 360
        bins.update((ang // bin_deg).astype(int).tolist())
        # fill small gaps between samples along the face
    covered = sorted(bins)
    if not covered:
        return 0.0
    # close 1-bin gaps caused by sparse sampling
    full = set(covered)
    for b in covered:
        if (b + 2) % (360 // bin_deg) in full:
            full.add((b + 1) % (360 // bin_deg))
    return len(full) * bin_deg


def dir_name(d):
    d = np.asarray(d)
    k = int(np.argmax(np.abs(d)))
    if abs(d[k]) > 0.999:
        return ("+" if d[k] > 0 else "-") + "XYZ"[k]
    return "[" + ", ".join(f"{c:.3f}" for c in d) + "]"


class Recognizer:
    def __init__(self, shape, faces, infos, neighbours, edges, is_solid=True):
        self.is_solid = is_solid
        self.shape = shape
        self.faces = faces
        self.info = infos
        self.nb = neighbours
        self.edges = edges
        self.used = set()
        self.chamfer_faces = set()
        self.all_points = np.vstack([i.points for i in infos])
        self.lo, self.hi = self.all_points.min(axis=0), self.all_points.max(axis=0)
        # build point-in-solid classifiers once; constructing one is the expensive part
        solids = []
        ex = TopExp_Explorer(shape, TopAbs_SOLID)
        while ex.More():
            solids.append(ex.Current())
            ex.Next()
        self.classifiers = [BRepClass3d_SolidClassifier(s) for s in (solids or [shape])]
        self.fmap = TopTools_IndexedMapOfShape()
        for f in faces:
            self.fmap.Add(f)
        self.edge_faces = TopTools_IndexedDataMapOfShapeListOfShape()
        TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, self.edge_faces)
        self.rays = MeshRays(faces)

    # ------------------------------------------------------------------ utils
    def in_material(self, p):
        if not self.is_solid:
            return self._in_material_by_rays(p)
        pnt = gp_Pnt(*map(float, p))
        for c in self.classifiers:
            c.Perform(pnt, 1e-6)
            if c.State() in (TopAbs_IN, TopAbs_ON):
                return True
        return False

    def _in_material_by_rays(self, p):
        return self.rays.inside(p)

    def end_capped(self, p, d, t_end, outward, delta):
        """Surface-model hole-end test: is there a surface capping the hole just past this end?
        Casts a short ray along the axis from just inside the hole."""
        start = p + d * (t_end - outward * 0.2)
        return self.rays.blocked(start, outward * d, t_max=0.2 + delta + 0.5)

    def on_stock_boundary(self, p, tol=0.1):
        p = np.asarray(p)
        return bool(np.any(np.abs(p - self.lo) < tol) or np.any(np.abs(p - self.hi) < tol))

    def rel(self, p):
        """Position relative to stock minimum corner (the datum)."""
        return _pt(np.asarray(p) - self.lo)

    def colors(self, fids):
        cols = defaultdict(int)
        for i in fids:
            c = self.info[i].color
            if c is not None:
                cols[c] += 1
        return [{"rgb": list(c), "faces": n} for c, n in sorted(cols.items(), key=lambda x: -x[1])]

    def edge_kinds(self, a, b):
        return [e.kind for e in self.nb.get(a, {}).get(b, [])]

    # ------------------------------------------------------------------ holes
    def _coaxial_clusters(self, fids):
        clusters = []
        for i in fids:
            f = self.info[i]
            d = canon_dir(f.axis_dir)
            p = f.axis_loc - (f.axis_loc @ d) * d
            for c in clusters:
                if 1 - abs(float(c["d"] @ d)) < AXIS_DIR_TOL and np.linalg.norm(c["p"] - p) < AXIS_POS_TOL:
                    c["fids"].append(i)
                    break
            else:
                clusters.append({"d": d, "p": p, "fids": [i]})
        return clusters

    def _split_axial(self, cluster):
        d, p = cluster["d"], cluster["p"]
        spans = []
        for i in cluster["fids"]:
            t0, t1 = self.info[i].axial_range(p, d)
            spans.append((t0, t1, i))
        spans.sort()
        comps, cur, cur_end = [], [], -math.inf
        for t0, t1, i in spans:
            if cur and t0 > cur_end + 0.05:
                comps.append(cur)
                cur, cur_end = [], -math.inf
            cur.append((t0, t1, i))
            cur_end = max(cur_end, t1)
        if cur:
            comps.append(cur)
        return comps

    def recognise_holes(self):
        cands = [f.id for f in self.info
                 if f.kind in ("cylinder", "cone") and f.concave and f.axis_dir is not None]
        holes = []
        for cl in self._coaxial_clusters(cands):
            d, p = cl["d"], cl["p"]
            for comp in self._split_axial(cl):
                h = self._build_hole(d, p, comp)
                if h:
                    holes.append(h)
        return holes

    def _build_hole(self, d, p, comp):
        infos = {i: self.info[i] for _, _, i in comp}
        cyl_ids = [i for i in infos if infos[i].kind == "cylinder"]
        if not cyl_ids:
            return None
        # group cylinder faces by radius (halves of a split cylinder share radius)
        by_r = defaultdict(list)
        for i in cyl_ids:
            r = infos[i].radius
            key = next((k for k in by_r if abs(k - r) < RADIUS_TOL), r)
            by_r[key].append(i)
        full = {r: ids for r, ids in by_r.items()
                if angular_coverage([infos[i] for i in ids], p, d) >= FULL_CIRCLE_DEG}
        if not full:
            return None     # partial arcs only: corner radii / slot ends, not a hole
        member_ids = [i for i in infos if infos[i].kind == "cone" or
                      any(i in ids for ids in full.values())]
        t_lo = min(infos[i].axial_range(p, d)[0] for i in member_ids)
        t_hi = max(infos[i].axial_range(p, d)[1] for i in member_ids)
        r_max = max(full)
        delta = min(0.5, 0.2 * min(full))
        if self.is_solid:
            closed_lo = self.in_material(p + d * (t_lo - delta))
            closed_hi = self.in_material(p + d * (t_hi + delta))
        else:
            closed_lo = self.end_capped(p, d, t_lo, -1.0, delta)
            closed_hi = self.end_capped(p, d, t_hi, 1.0, delta)
        # A drill point the CAD wrote as a free-form face isn't in the component, so the end
        # looks open. Look down the axis for a cap within a 118° tip length (~0.6 r) of the end.
        tip_lo = tip_hi = False
        r_small = min(full)
        tip_len = 0.2 + 0.65 * r_small + 0.5
        if not closed_lo and self.rays.blocked(p + d * (t_lo + 0.2), -d, t_max=tip_len):
            closed_lo = tip_lo = True
        if not closed_hi and self.rays.blocked(p + d * (t_hi - 0.2), d, t_max=tip_len):
            closed_hi = tip_hi = True
        if closed_lo and closed_hi:
            return None     # enclosed void; not machinable as a hole
        through = not closed_lo and not closed_hi
        # An "open" end may still have a flat shoulder with a narrower opening in it
        # (e.g. Ø16 down to a floor, smaller bore continuing): probe off-axis at 0.8 r.
        # Probe 4 points around the rim: a real shoulder is a full ring (>= 3 hits); a hole
        # that merely breaks out into a cavity at an angle only hits on one side.
        def shoulder(t_end, outward):
            e1, e2 = _frame(d)
            r_end = min(full)
            hits = 0
            for off in (e1, -e1, e2, -e2):
                start = p + d * (t_end - outward * 0.2) + off * 0.8 * r_end
                hits += self.rays.blocked(start, outward * d, t_max=0.2 + delta + 0.5)
            return hits >= 3
        shoulder_lo = not closed_lo and shoulder(t_lo, -1.0)
        shoulder_hi = not closed_hi and shoulder(t_hi, 1.0)

        def radius_at_end(t_end):
            best = None
            for r, ids in full.items():
                for i in ids:
                    a, b = infos[i].axial_range(p, d)
                    dist = min(abs(a - t_end), abs(b - t_end))
                    if best is None or dist < best[0] - 1e-6 or (abs(dist - best[0]) < 1e-6 and r > best[1]):
                        best = (dist, r)
            return best[1]

        if through:
            r_lo_end, r_hi_end = radius_at_end(t_lo), radius_at_end(t_hi)
            on_stock_lo = self.on_stock_boundary(p + d * t_lo)
            on_stock_hi = self.on_stock_boundary(p + d * t_hi)
            if abs(r_lo_end - r_hi_end) > RADIUS_TOL:
                entry_hi = r_hi_end > r_lo_end         # counterbore side is the entry
            elif on_stock_lo != on_stock_hi:
                entry_hi = on_stock_hi                 # drilled from the outside face, exits into a cavity
            else:
                entry_hi = (d @ np.ones(3)) >= 0        # canonical dir points +; enter from + side
        else:
            entry_hi = closed_lo                       # open end is the entry
        drill = -d if entry_hi else d                  # direction into material
        t_entry = t_hi if entry_hi else t_lo

        def depth_of(t):
            return (t_entry - t) if entry_hi else (t - t_entry)

        # ---- ordered profile from the entry
        segs = []
        for r, ids in full.items():
            a = min(infos[i].axial_range(p, d)[0] for i in ids)
            b = max(infos[i].axial_range(p, d)[1] for i in ids)
            s0, s1 = sorted((depth_of(a), depth_of(b)))
            segs.append({"kind": "cylinder", "diameter": 2 * r, "from": s0, "to": s1, "ids": ids})
        for i in member_ids:
            f = infos[i]
            if f.kind != "cone":
                continue
            a, b = f.axial_range(p, d)
            s0, s1 = sorted((depth_of(a), depth_of(b)))
            rel = f.points - p
            rho = np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)
            segs.append({"kind": "cone", "angle": 2 * f.semi_angle, "from": s0, "to": s1,
                         "d_min": 2 * float(rho.min()), "d_max": 2 * float(rho.max()), "ids": [i]})
        segs.sort(key=lambda s: (s["from"], s["to"]))
        cyls = [s for s in segs if s["kind"] == "cylinder"]
        cones = [s for s in segs if s["kind"] == "cone"]
        main = min(cyls, key=lambda s: s["diameter"])
        total_depth = max(s["to"] for s in segs)
        cyl_depth = max(s["to"] for s in cyls)

        hole = {
            "type": "hole",
            "subtype": "through" if through else "blind",
            "diameter": _r(main["diameter"]),
            "depth": _r(cyl_depth),
            "entry_point": _pt(p + d * t_entry),
            "direction": _pt(drill),
            "direction_name": dir_name(drill),
        }
        form = "simple"
        entry_cyl = cyls[0]
        if len(cyls) > 1:
            larger_first = entry_cyl["diameter"] > main["diameter"] + 2 * RADIUS_TOL
            if larger_first:
                form = "counterbore"
                hole["counterbore"] = {"diameter": _r(entry_cyl["diameter"]),
                                       "depth": _r(entry_cyl["to"] - entry_cyl["from"])}
            else:
                form = "stepped"
            hole["steps"] = [{"diameter": _r(s["diameter"]), "from_depth": _r(s["from"]),
                              "to_depth": _r(s["to"])} for s in cyls]

        for c in cones:
            at_entry = c["from"] < 0.05
            at_bottom = (not through) and c["to"] >= total_depth - 0.05 and c["d_min"] < 0.5
            if at_bottom:
                hole["bottom"] = f"conical, {c['angle']:.0f} deg" + (" (drill point)" if abs(c["angle"] - 118) < 2 else "")
                hole["depth_including_tip"] = _r(total_depth)
            elif at_entry:
                ref = entry_cyl["diameter"]
                size = (c["d_max"] - ref) / 2
                if size <= max(1.0, 0.15 * ref):
                    hole["entry_chamfer"] = {"size": _r(size), "angle": _r(c["angle"] / 2, 1),
                                             "outer_diameter": _r(c["d_max"])}
                else:
                    form = "countersink" if form == "simple" else form + "+countersink"
                    hole["countersink"] = {"diameter": _r(c["d_max"]), "angle": _r(c["angle"], 1)}
            elif through and c["to"] >= total_depth - 0.05:
                hole["exit_chamfer"] = {"size": _r((c["d_max"] - main["diameter"]) / 2),
                                        "angle": _r(c["angle"] / 2, 1), "outer_diameter": _r(c["d_max"])}
            else:
                cone = {"angle": _r(c["angle"], 1), "from_depth": _r(c["from"]), "to_depth": _r(c["to"]),
                        "d_min": _r(c["d_min"]), "d_max": _r(c["d_max"])}
                if cone not in hole.get("internal_cones", []):     # split halves of one cone
                    hole.setdefault("internal_cones", []).append(cone)
        exit_shoulder = shoulder_lo if entry_hi else shoulder_hi
        if through and exit_shoulder:
            hole["subtype"] = "stepped through"
            hole["bottom"] = "flat shoulder with a smaller opening continuing through"
        bottom_is_tip = tip_lo if entry_hi else tip_hi
        if "bottom" not in hole and bottom_is_tip:
            hole["bottom"] = "conical (drill point)"
        if "bottom" not in hole:
            hole["bottom"] = "open (through)" if through else "flat"
        hole["form"] = form

        dia = main["diameter"]
        tap = next((m for k, m in TAP_DRILL.items() if abs(k - dia) < 0.06), None)
        minor = next((m for k, m in THREAD_MINOR.items() if abs(k - dia) < 0.02), None)
        clr = next((m for k, m in CLEARANCE.items() if abs(k - dia) < 0.06), None)
        if minor:
            hole["thread_hint"] = f"likely tapped {minor} (Ø{dia:.3f} is the {minor} thread minor diameter)"
        elif tap:
            hole["thread_hint"] = f"possibly tapped {tap} (Ø{dia:.1f} is the {tap} tap-drill size; thread not modelled in STEP)"
        elif clr:
            hole["thread_hint"] = f"Ø{dia:.1f} matches {clr} bolt clearance"

        fids = set(i for s in segs for i in s["ids"])
        # claim flat bottoms / counterbore shoulders: planes perpendicular to the axis inside the hole
        for i in list(fids):
            for j in self.nb.get(i, {}):
                g = self.info[j]
                if g.kind == "plane" and j not in fids and abs(abs(g.normal @ d) - 1) < 1e-3:
                    rel = g.points - p
                    rho = np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)
                    if rho.max() <= r_max + 0.05:
                        fids.add(j)
        hole["face_ids"] = sorted(fids)
        self.used.update(fids)
        return hole

    # ------------------------------------------------------------------ slots
    def recognise_slots(self):
        cands = [f.id for f in self.info if f.id not in self.used and f.kind == "cylinder"
                 and f.concave and f.axis_dir is not None]
        ends = []
        for cl in self._coaxial_clusters(cands):
            for comp in self._split_axial(cl):
                ids = [i for _, _, i in comp]
                r = float(np.median([self.info[i].radius for i in ids]))
                cov = angular_coverage([self.info[i] for i in ids], cl["p"], cl["d"])
                if 150 <= cov <= 215:
                    t0 = min(c[0] for c in comp)
                    t1 = max(c[1] for c in comp)
                    ends.append({"ids": ids, "r": r, "d": cl["d"], "p": cl["p"], "t0": t0, "t1": t1,
                                 "mid": np.mean(np.vstack([self.info[i].points for i in ids]), axis=0)})
        slots, taken = [], set()
        for a in range(len(ends)):
            if a in taken:
                continue
            best = None
            for b in range(a + 1, len(ends)):
                if b in taken:
                    continue
                A, B = ends[a], ends[b]
                if abs(A["r"] - B["r"]) > RADIUS_TOL or 1 - abs(float(A["d"] @ B["d"])) > AXIS_DIR_TOL:
                    continue
                if abs(A["t0"] - B["t0"]) > 0.05 or abs(A["t1"] - B["t1"]) > 0.05:
                    continue
                ca = A["p"] + A["d"] * A["t0"]
                cb = B["p"] + A["d"] * A["t0"]
                sep = cb - ca
                L = float(np.linalg.norm(sep))
                if L < 1e-3:
                    continue
                u = sep / L
                # the half-circles must bulge away from each other
                if float((A["mid"] - ca) @ u) > 0 or float((B["mid"] - cb) @ u) < 0:
                    continue
                # must share a straight wall with each other
                wall = set()
                for i in A["ids"]:
                    for j in self.nb.get(i, {}):
                        if self.info[j].kind == "plane" and any(j in self.nb.get(k, {}) for k in B["ids"]):
                            wall.add(j)
                if not wall:
                    continue
                if best is None or L < best[0]:
                    best = (L, b, u, wall, ca, cb)
            if best is None:
                continue
            L, b, u, wall, ca, cb = best
            A, B = ends[a], ends[b]
            taken.update((a, b))
            fids = set(A["ids"]) | set(B["ids"]) | wall
            floor = [j for i in fids for j in self.nb.get(i, {})
                     if self.info[j].kind == "plane" and abs(abs(self.info[j].normal @ A["d"]) - 1) < 1e-3
                     and "concave" in self.edge_kinds(i, j)]
            depth = A["t1"] - A["t0"]
            centre = (ca + cb) / 2 + A["d"] * depth / 2          # mid-length, mid-depth
            slot = {
                "type": "slot",
                "subtype": "blind" if floor else "through",
                "width": _r(2 * A["r"]),
                "length": _r(L + 2 * A["r"]),
                "centre_distance": _r(L),
                "depth": _r(depth),
                "centre": _pt(centre),
                "length_direction": _pt(u),
                "depth_axis": dir_name(A["d"]),
            }
            fids |= set(floor)
            slot["face_ids"] = sorted(fids)
            self.used.update(fids)
            slots.append(slot)
        return slots

    # ------------------------------------------------------------------ pockets / steps
    def accessible(self, fid, n, samples=7):
        """True if most sample points on the face can 'see' open air along n
        (i.e. a tool coming from direction n can reach it)."""
        pts = self.info[fid].points
        if len(pts) > samples:
            pts = pts[np.linspace(0, len(pts) - 1, samples).astype(int)]
        free = 0
        for p in pts:
            if not self.rays.blocked(p + n * 0.05, n):
                free += 1
        return free >= 0.6 * len(pts)

    def recognise_pockets(self):
        results = []
        wall_owner = set()     # faces already used as pocket walls may be shared between stacked pockets
        floors = [f.id for f in self.info if f.kind == "plane" and f.id not in self.used
                  and float(np.abs(f.normal).max()) > 0.999]      # 2.5D: floors face a principal axis
        done = set()
        for fid in floors:
            if fid in done:
                continue
            n = self.info[fid].normal
            level = float(self.info[fid].points.mean(axis=0) @ n)
            stock = float((self.all_points @ n).max())
            if stock - level < 0.05:
                continue            # the stock face itself
            # merge coplanar connected floor faces
            floor = {fid}
            stack = [fid]
            while stack:
                k = stack.pop()
                for j in self.nb.get(k, {}):
                    g = self.info[j]
                    if (j not in floor and j not in self.used and g.kind == "plane"
                            and float(g.normal @ n) > 1 - 1e-4
                            and abs(float(g.points.mean(axis=0) @ n) - level) < 0.02):
                        floor.add(j)
                        stack.append(j)
            done |= floor
            if not any(self.accessible(j, n) for j in floor):
                continue            # a wall of something else, not a floor
            # walls: faces meeting the floor at a concave edge, rising above it;
            # then continue only across smooth edges (corner radii between walls)
            walls = set()
            for k in floor:
                for j, es in self.nb.get(k, {}).items():
                    if j in floor or (j in self.used and j not in wall_owner):
                        continue
                    kinds = {e.kind for e in es}
                    g = self.info[j]
                    if float((g.points @ n).min()) < level - 0.05:
                        continue
                    if "concave" in kinds or ("smooth" in kinds and g.concave):
                        walls.add(j)
            stack = list(walls)
            while stack:
                k = stack.pop()
                for j, es in self.nb.get(k, {}).items():
                    if j in floor or j in walls or (j in self.used and j not in wall_owner):
                        continue
                    g = self.info[j]
                    if ({e.kind for e in es} == {"smooth"} and float(np.abs(g.normals @ n).mean()) < 0.3
                            and float((g.points @ n).min()) >= level - 0.05):
                        walls.add(j)
                        stack.append(j)
            if not walls:
                continue
            wall_normals = []
            for j in walls:
                g = self.info[j]
                ip = g.normals - np.outer(g.normals @ n, n)
                ln = np.linalg.norm(ip, axis=1)
                wall_normals.extend((ip[ln > 0.5] / ln[ln > 0.5, None]).tolist())
            if not wall_normals:
                continue
            wn = np.array(wall_normals)
            e1, e2 = _frame(n)
            # align frame to a principal axis if possible
            for ax in np.eye(3):
                if abs(ax @ n) < 1e-3:
                    e1 = ax
                    e2 = np.cross(n, e1)
                    break
            # walls face INTO the pocket; count which of 4 in-plane sides are enclosed
            sides = set()
            for v in wn:
                a = math.degrees(math.atan2(v @ e2, v @ e1)) % 360
                sides.add(int(((a + 45) % 360) // 90))
            region = floor | walls
            pts = np.vstack([self.info[j].points for j in region])
            top = float((np.vstack([self.info[j].points for j in walls]) @ n).max())
            depth = top - level
            u, v = pts @ e1, pts @ e2
            size1, size2 = float(u.max() - u.min()), float(v.max() - v.min())
            cu, cv = (u.max() + u.min()) / 2, (v.max() + v.min()) / 2
            centre_top = e1 * cu + e2 * cv + n * top
            floor_area = sum(self.info[j].area for j in floor)
            corner_r = sorted({_r(self.info[j].radius, 2) for j in walls
                               if self.info[j].kind == "cylinder" and self.info[j].concave
                               and abs(abs(self.info[j].axis_dir @ n) - 1) < 1e-2})
            floor_r = sorted({_r(self.info[j].radius, 2) for j in walls
                              if self.info[j].kind in ("cylinder", "blend") and self.info[j].concave
                              and (self.info[j].axis_dir is None or abs(self.info[j].axis_dir @ n) < 1e-2)})
            if len(sides) == 4 and corner_r and abs(size1 - size2) < 0.1 and abs(size1 / 2 - max(corner_r)) < 0.1:
                subtype = "circular pocket / spot face"
            elif len(sides) == 4:
                subtype = "closed pocket"
            elif len(sides) == 3:
                subtype = "open pocket"
            elif len(sides) == 2 and (sides == {0, 2} or sides == {1, 3}):
                subtype = "through channel"
            else:
                subtype = "step / shoulder"
            feat = {
                "type": "pocket" if "pocket" in subtype else ("channel" if "channel" in subtype else "step"),
                "subtype": subtype,
                "open_direction": dir_name(n),
                "depth": _r(depth),
                "length": _r(max(size1, size2)),
                "width": _r(min(size1, size2)),
                "floor_area": _r(floor_area, 2),
                "floor_level": _r(level),
                "centre_at_top": _pt(centre_top),
            }
            if corner_r:
                feat["corner_radius"] = corner_r
            if floor_r:
                feat["floor_fillet_radius"] = floor_r
            feat["face_ids"] = sorted(region)
            self.used.update(region)
            wall_owner.update(walls)
            results.append(feat)
        return results

    # ------------------------------------------------------------------ bosses
    def _inner_wire_neighbours(self, fid):
        """For each inner wire (hole in the face boundary): the set of faces across its edges."""
        face = self.faces[fid]
        outer = BRepTools.OuterWire_s(face)
        result = []
        ex = TopExp_Explorer(face, TopAbs_WIRE)
        while ex.More():
            w = ex.Current()
            ex.Next()
            if w.IsSame(outer):
                continue
            nbrs = set()
            ee = TopExp_Explorer(w, TopAbs_EDGE)
            while ee.More():
                e = ee.Current()
                ee.Next()
                k = self.edge_faces.FindIndex(e)
                if k == 0:
                    continue
                for f in self.edge_faces.FindFromIndex(k):
                    j = self.fmap.FindIndex(f) - 1
                    if j >= 0 and j != fid:
                        nbrs.add(j)
            result.append(nbrs)
        return result

    def recognise_cutouts(self):
        """Through openings (windows) with no floor: found as inner loops of planar faces
        whose walls run parallel to the face normal and let a ray pass right through."""
        out = []
        for f in self.info:
            if f.kind != "plane" or float(np.abs(f.normal).max()) < 0.999:
                continue
            n = f.normal
            for nbrs in self._inner_wire_neighbours(f.id):
                # openings are usually edge-chamfered: step across the chamfer ring to the walls
                ring = {j for j in nbrs if j in self.chamfer_faces}
                for j in ring:
                    nbrs = nbrs | {k for k in self.nb.get(j, {}) if k != f.id and k not in self.chamfer_faces}
                walls = {j for j in nbrs if j not in self.used}
                if not walls:
                    continue
                # grow across smooth/concave edges through wall-like faces
                stack = list(walls)
                while stack:
                    k = stack.pop()
                    for j, es in self.nb.get(k, {}).items():
                        g = self.info[j]
                        if (j in walls or j in self.used or
                                "convex" in {e.kind for e in es} or
                                float(np.abs(g.normals @ n).mean()) > 0.3):
                            continue
                        walls.add(j)
                        stack.append(j)
                walls = {j for j in walls if float(np.abs(self.info[j].normals @ n).mean()) < 0.3}
                if not walls:
                    continue
                # through? from a point in the opening's air, a ray must escape both ways along n.
                # Try the opening centre and points stepped off the largest walls (drafted
                # walls lean, so a point just off the wall can be shadowed by the wall itself).
                pts = np.vstack([self.info[j].points for j in walls])
                height = float(np.ptp(pts @ n))
                starts = [pts.min(axis=0) / 2 + pts.max(axis=0) / 2]
                for j in sorted(walls, key=lambda j: -self.info[j].area)[:4]:
                    w = self.info[j]
                    k = len(w.points) // 2
                    starts.append(w.points[k] + w.normals[k] * (0.3 + 0.1 * height))
                through = False
                for start in starts:
                    if self.in_material(start):
                        continue
                    blocked = False
                    for sgn in (1, -1):
                        blocked |= self.rays.blocked(start, sgn * n)
                    if not blocked:
                        through = True
                        break
                if not through:
                    continue
                e1 = next(ax for ax in np.eye(3) if abs(ax @ n) < 1e-3)
                e2 = np.cross(n, e1)
                u, v, h = pts @ e1, pts @ e2, pts @ n
                size1, size2 = float(np.ptp(u)), float(np.ptp(v))
                centre = e1 * (u.max() + u.min()) / 2 + e2 * (v.max() + v.min()) / 2 + n * h.max()
                corner_r = sorted({_r(self.info[j].radius, 2) for j in walls
                                   if self.info[j].kind == "cylinder" and self.info[j].concave})
                feat = {"type": "cutout", "subtype": "through opening",
                        "axis": dir_name(canon_dir(n)),
                        "length": _r(max(size1, size2)), "width": _r(min(size1, size2)),
                        "depth": _r(float(np.ptp(h))), "centre": _pt(centre),
                        "wall_faces": len(walls)}
                if corner_r:
                    feat["corner_radius"] = corner_r
                feat["face_ids"] = sorted(walls)
                self.used.update(walls)
                out.append(feat)
                if ring:
                    feat["edge_chamfer_faces"] = sorted(ring)
        return out

    def recognise_bosses(self):
        cands = [f.id for f in self.info if f.id not in self.used and f.kind == "cylinder"
                 and f.concave is False and f.axis_dir is not None]
        out = []
        for cl in self._coaxial_clusters(cands):
            for comp in self._split_axial(cl):
                ids = [i for _, _, i in comp]
                cov = angular_coverage([self.info[i] for i in ids], cl["p"], cl["d"])
                if cov < FULL_CIRCLE_DEG:
                    continue
                t0 = min(c[0] for c in comp)
                t1 = max(c[1] for c in comp)
                r = float(np.median([self.info[i].radius for i in ids]))
                out.append({
                    "type": "boss",
                    "subtype": "cylindrical boss / pin",
                    "diameter": _r(2 * r),
                    "height": _r(t1 - t0),
                    "axis": dir_name(cl["d"]),
                    "base_centre": _pt(cl["p"] + cl["d"] * t0),
                    "face_ids": sorted(ids),
                })
                self.used.update(ids)
        return out

    # ------------------------------------------------------------------ chamfers
    def recognise_chamfers(self):
        out = []
        for f in self.info:
            if f.id in self.used:
                continue
            if f.kind == "plane":
                nbrs = [(j, es) for j, es in self.nb.get(f.id, {}).items()]
                convex_nbrs = [j for j, es in nbrs if all(e.kind == "convex" for e in es)]
                if len(convex_nbrs) < 2:
                    continue
                # two neighbours whose normals bracket this one symmetrically-ish
                best = None
                for a in range(len(convex_nbrs)):
                    for b in range(a + 1, len(convex_nbrs)):
                        A, B = self.info[convex_nbrs[a]], self.info[convex_nbrs[b]]
                        if A.kind != "plane" or B.kind != "plane":
                            continue
                        ang_ab = math.degrees(math.acos(np.clip(A.normal @ B.normal, -1, 1)))
                        if ang_ab < 30 or ang_ab > 170:     # opposite faces can't bracket a chamfer
                            continue
                        a1 = math.degrees(math.acos(np.clip(f.normal @ A.normal, -1, 1)))
                        a2 = math.degrees(math.acos(np.clip(f.normal @ B.normal, -1, 1)))
                        if abs(a1 + a2 - ang_ab) < 1.0:
                            best = (A, B, a1, a2)
                if best is None:
                    continue
                A, B, a1, a2 = best
                # width across the chamfer: extent perpendicular to the shared edge direction
                edge_dir = np.cross(A.normal, B.normal)
                edge_dir /= np.linalg.norm(edge_dir)
                across = np.cross(f.normal, edge_dir)
                w = float(np.ptp(f.points @ across))
                length = float(np.ptp(f.points @ edge_dir))
                if w > 20 or w > 0.5 * length:
                    continue
                leg1 = w * math.sin(math.radians(a2)) / math.sin(math.radians(a1 + a2)) if a1 + a2 > 0 else w
                leg2 = w * math.sin(math.radians(a1)) / math.sin(math.radians(a1 + a2)) if a1 + a2 > 0 else w
                out.append({"type": "chamfer", "subtype": "edge chamfer",
                            "size": _r(min(leg1, leg2)) if abs(leg1 - leg2) < 0.01 else f"{leg1:.2f} x {leg2:.2f}",
                            "angle": _r(min(a1, a2), 1), "width": _r(w), "length": _r(length),
                            "position": _pt(f.centroid), "face_ids": [f.id]})
                self.used.add(f.id)
                self.chamfer_faces.add(f.id)
            elif f.kind == "cone" and f.semi_angle and 20 < f.semi_angle < 50:
                rel = f.points - f.axis_loc
                d = canon_dir(f.axis_dir)
                rho = np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)
                w = float(rho.max() - rho.min())
                if w > 10:
                    continue
                full = angular_coverage([f], f.axis_loc, d) >= FULL_CIRCLE_DEG
                kind = "internal" if f.concave else "external"
                out.append({"type": "chamfer",
                            "subtype": f"{kind} circular chamfer" if full else f"{kind} arc edge chamfer",
                            "size": _r(w), "angle": _r(f.semi_angle, 1),
                            "diameter_max": _r(2 * rho.max()), "diameter_min": _r(2 * rho.min()),
                            "axis": dir_name(d), "position": _pt(f.centroid), "face_ids": [f.id]})
                self.used.add(f.id)
                self.chamfer_faces.add(f.id)
        return out

    # ------------------------------------------------------------------ fillets
    def recognise_fillets(self):
        out = []
        for f in self.info:
            if f.id in self.used or f.kind not in ("cylinder", "blend", "cone", "sphere", "freeform"):
                continue
            if f.kind == "cone" or (f.kind == "freeform"):
                continue
            kinds = [e.kind for es in self.nb.get(f.id, {}).values() for e in es]
            if "smooth" not in kinds and f.kind != "blend":
                continue
            r = f.radius
            if r is None or r > 50:
                continue
            length = f.area / max(r * (math.pi / 2), 1e-9)
            out.append({"type": "fillet" if f.concave else "round",
                        "subtype": ("internal fillet / corner radius" if f.concave else "external round / edge radius"),
                        "radius": _r(r, 2), "approx_length": _r(length, 1),
                        "axis": dir_name(canon_dir(f.axis_dir)) if f.axis_dir is not None else None,
                        "position": _pt(f.centroid), "face_ids": [f.id]})
            self.used.add(f.id)
        return out

    def leftovers(self):
        names = {"+Z": "top", "-Z": "bottom", "+X": "right side", "-X": "left side",
                 "+Y": "back side", "-Y": "front side"}
        out = []
        for f in self.info:
            if f.id in self.used:
                continue
            item = {"type": "surface", "area": _r(f.area, 2), "position": _pt(f.centroid), "face_ids": [f.id]}
            if f.area < 1.0:
                item["subtype"] = "sliver"
            elif f.kind == "plane":
                dn = dir_name(f.normal)
                level = float(f.points.mean(axis=0) @ f.normal)
                item["normal"] = dn
                if dn in names:
                    at_stock = float((self.all_points @ f.normal).max()) - level < 0.05
                    item["subtype"] = f"stock face ({names[dn]})" if at_stock else "planar face"
                    if not at_stock:
                        item["level"] = _r(level)
                else:
                    k = int(np.argmax(np.abs(f.normal)))
                    tilt = math.degrees(math.acos(min(1.0, abs(float(f.normal[k])))))
                    item["subtype"] = "inclined face"
                    item["tilt_deg"] = _r(tilt, 2)
                    item["tilt_from"] = "XYZ"[k]
            elif f.kind == "cone" and f.semi_angle and abs(f.semi_angle - 59) < 1.5:
                item["subtype"] = "drill-point cone (partial)"
            elif f.kind == "cone":
                item["subtype"] = "conical face"
                item["half_angle"] = _r(f.semi_angle, 2)
            else:
                item["subtype"] = f"{f.kind} face"
            if f.radius is not None:
                item["radius"] = _r(f.radius)
            if f.concave is not None:
                item["concave"] = bool(f.concave)
            out.append(item)
        return out
