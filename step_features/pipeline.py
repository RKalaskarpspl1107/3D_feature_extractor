"""End-to-end: STEP file -> structured feature list."""

import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

from .faces import analyse_face, mesh_shape
from .loader import load_step
from .recognize import Recognizer, _pt, _r
from .topology import build_adjacency

# Keys that describe a feature's *shape*; features sharing them form a pattern.
_SIGNATURE_KEYS = {
    "hole": ("subtype", "form", "diameter", "depth", "direction_name", "counterbore", "countersink",
             "bottom", "entry_chamfer"),
    "slot": ("subtype", "width", "length", "depth", "depth_axis"),
    "pocket": ("subtype", "length", "width", "depth", "open_direction"),
    "step": ("subtype", "length", "width", "depth", "open_direction"),
    "channel": ("subtype", "length", "width", "depth", "open_direction"),
    "boss": ("diameter", "height", "axis"),
    "cutout": ("length", "width", "depth", "axis"),
    "chamfer": ("subtype", "size", "angle"),
    "fillet": ("radius",),
    "round": ("radius",),
}

_POSITION_KEYS = ("entry_point", "centre", "centre_at_top", "base_centre", "position")


def _norm(v):
    """Round away float noise (63.5 vs 63.501, -0.0) so identical features compare equal."""
    if isinstance(v, float):
        return round(v, 2) + 0.0
    if isinstance(v, dict):
        return tuple(sorted((k, _norm(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_norm(x) for x in v)
    return v


def _signature(f):
    keys = _SIGNATURE_KEYS.get(f["type"])
    if not keys:
        return None
    return (f["type"],) + tuple(repr(_norm(f.get(k))) for k in keys)


def _position(f):
    for k in _POSITION_KEYS:
        if k in f:
            return f[k]
    return None


def describe(f):
    """One-line human-readable callout, drawing-style."""
    t = f["type"]
    if t == "hole":
        s = f"Ø{f['diameter']:g}"
        if f["subtype"] == "through":
            s += " THRU"
        elif f["subtype"] == "stepped through":
            s += f" ↧{f['depth']:g} to shoulder, smaller opening THRU"
        else:
            s += f" ↧{f['depth']:g}"
        if "counterbore" in f:
            s += f", ⌴ Ø{f['counterbore']['diameter']:g} ↧{f['counterbore']['depth']:g}"
        if "countersink" in f:
            s += f", ⌵ Ø{f['countersink']['diameter']:g} × {f['countersink']['angle']:g}°"
        if "entry_chamfer" in f:
            s += f", C{f['entry_chamfer']['size']:g}"
        if f.get("bottom", "").startswith("conical"):
            s += f", {f['bottom']}"
        if f.get("thread_hint", "").startswith(("possibly", "likely")):
            s += f" [{f['thread_hint'].split(' (')[0]}]"
        return s
    if t == "slot":
        return f"Slot {f['length']:g} × {f['width']:g}, depth {f['depth']:g} ({f['subtype']})"
    if t == "pocket" and f["subtype"].startswith("circular"):
        return f"Circular pocket / spot face Ø{f['length']:g}, depth {f['depth']:g}, opens {f['open_direction']}"
    if t in ("pocket", "step", "channel"):
        s = f"{f['subtype'].capitalize()} {f['length']:g} × {f['width']:g}, depth {f['depth']:g}, opens {f['open_direction']}"
        if f.get("corner_radius"):
            s += f", corner R{'/'.join(f'{r:g}' for r in f['corner_radius'])}"
        return s
    if t == "cutout":
        s = f"Through opening {f['length']:g} × {f['width']:g}, thickness {f['depth']:g}, along {f['axis']}"
        if f.get("corner_radius"):
            s += f", corner R{'/'.join(f'{r:g}' for r in f['corner_radius'])}"
        return s
    if t == "boss":
        return f"Boss Ø{f['diameter']:g} × {f['height']:g} along {f['axis']}"
    if t == "chamfer":
        size = f["size"] if isinstance(f["size"], str) else f"{f['size']:g}"
        return f"Chamfer {size} × {f['angle']:g}° ({f['subtype']})"
    if t in ("fillet", "round"):
        return f"{t.capitalize()} R{f['radius']:g} ({f['subtype']})"
    if t == "surface":
        s = f"{f['subtype'][0].upper()}{f['subtype'][1:]}"
        if "tilt_deg" in f:
            s += f", {f['tilt_deg']:g}° from {f['tilt_from']}"
        if "half_angle" in f:
            s += f", half-angle {f['half_angle']:g}°"
        return s + f", area {f['area']:g} mm²"
    return t


def extract_features(step_path, mesh_deflection=0.2, viewer_path=None):
    """Run the full pipeline. If viewer_path is given, also write the HTML 3D viewer there."""
    t0 = time.time()
    model = load_step(step_path)
    mesh_shape(model.shape, mesh_deflection)
    infos = [analyse_face(i, f, model.face_colors.get(i)) for i, f in enumerate(model.faces)]
    edges, neighbours = build_adjacency(model.shape, model.faces)

    rec = Recognizer(model.shape, model.faces, infos, neighbours, edges, model.is_solid)
    raw = []
    raw += rec.recognise_holes()
    raw += rec.recognise_slots()
    raw += rec.recognise_chamfers()
    raw += rec.recognise_pockets()
    raw += rec.recognise_cutouts()
    raw += rec.recognise_bosses()
    raw += rec.recognise_fillets()
    leftovers = rec.leftovers()
    slivers = [f for f in leftovers if f["subtype"] == "sliver"]
    raw += [f for f in leftovers if f["subtype"] != "sliver"]

    # position relative to stock datum + colour + callout
    for f in raw:
        p = _position(f)
        if p is not None:
            f["position_from_datum"] = rec.rel(p)
        f["colors"] = rec.colors(f["face_ids"])
        f["callout"] = describe(f)

    # ---- patterns: identical features
    groups = defaultdict(list)
    for f in raw:
        sig = _signature(f)
        if sig is not None:
            groups[sig].append(f)
    order = {"hole": 0, "slot": 1, "pocket": 2, "channel": 3, "step": 4, "cutout": 4.5, "boss": 5,
             "chamfer": 6, "fillet": 7, "round": 8, "surface": 9}

    def sort_key(f):
        p = f.get("position_from_datum") or [0, 0, 0]
        return (order.get(f["type"], 99), -(f.get("diameter") or f.get("depth") or f.get("radius") or 0), p)

    raw.sort(key=sort_key)
    patterns = []
    pid_of = {}
    for sig, members in groups.items():
        if len(members) < 2:
            continue
        pid = f"P{len(patterns) + 1}"
        for m in members:
            pid_of[id(m)] = pid
        patterns.append({"pattern_id": pid, "type": members[0]["type"], "count": len(members),
                         "callout": f"{len(members)}× {members[0]['callout']}",
                         "positions_from_datum": [m.get("position_from_datum") for m in members]})

    for n, f in enumerate(raw, 1):
        f["id"] = n
        if id(f) in pid_of:
            f["pattern_id"] = pid_of[id(f)]

    # ---- part summary
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(model.shape, props)
    sprops = GProp_GProps()
    BRepGProp.SurfaceProperties_s(model.shape, sprops)
    size = rec.hi - rec.lo
    face_kinds = Counter(i.kind for i in infos)
    color_summary = defaultdict(lambda: Counter())
    for f in raw:
        for c in f["colors"]:
            color_summary[tuple(c["rgb"])][f["type"]] += c["faces"]

    summary = {
        "file": Path(step_path).name,
        "units": "mm",
        "stock_size": {"x": _r(size[0]), "y": _r(size[1]), "z": _r(size[2])},
        "bounding_box_min": _pt(rec.lo),
        "bounding_box_max": _pt(rec.hi),
        "datum_note": "position_from_datum is measured from the bounding-box minimum corner (min X, min Y, min Z)",
        "volume_mm3": _r(props.Mass(), 1),
        "surface_area_mm2": _r(sprops.Mass(), 1),
        "stock_volume_mm3": _r(float(np.prod(size)), 1),
        "material_removed_pct": _r(100 * (1 - props.Mass() / float(np.prod(size))), 1),
        "face_count": len(infos),
        "face_types": dict(face_kinds),
        "edge_count": len(edges),
        "model_type": "solid" if model.is_solid else "surface model (sewn)",
        "warnings": model.warnings,
        "tiny_faces_omitted": {"count": len(slivers), "note": "faces under 1 mm² not explained by any feature",
                               "face_ids": sorted(i for f in slivers for i in f["face_ids"])},
        "feature_counts": dict(Counter(f["type"] for f in raw)),
        "colors": [{"rgb": list(c), "faces_by_feature_type": dict(v)} for c, v in
                   sorted(color_summary.items(), key=lambda x: -sum(x[1].values()))],
        "processing_seconds": _r(time.time() - t0, 1),
    }
    result = {"part": summary, "features": raw, "patterns": patterns}
    if viewer_path:
        from .viewer import write_viewer
        write_viewer(result, model.faces, model.face_colors, viewer_path)
    return result
