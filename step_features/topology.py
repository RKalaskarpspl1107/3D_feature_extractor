"""Face adjacency graph with edge convexity.

An edge between faces A and B is:
  convex  - material angle < 180 deg (outside corner, e.g. top edge of a block)
  concave - material angle > 180 deg (inside corner, e.g. pocket floor-to-wall)
  smooth  - tangent-continuous (fillet / blend boundary)
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Curve2d, BRepAdaptor_Surface
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.GCPnts import GCPnts_AbscissaPoint
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Pnt, gp_Vec
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape

SMOOTH_ANGLE_DEG = 5.0


@dataclass
class Edge:
    faces: tuple          # (fid_a, fid_b)
    kind: str             # convex | concave | smooth
    angle: float          # angle between outward normals, degrees
    midpoint: np.ndarray
    length: float


def _vec(p):
    return np.array([p.X(), p.Y(), p.Z()], dtype=float)


_adaptor_cache = {}


def _normal_at(fid, face, edge, t):
    """Outward normal of `face` at parameter t of `edge`, via the edge's own pcurve
    (exact and much faster than projecting the 3D point onto a B-spline surface)."""
    if fid not in _adaptor_cache:
        _adaptor_cache[fid] = BRepAdaptor_Surface(face)
    uv = BRepAdaptor_Curve2d(edge, face).Value(t)
    props = BRepLProp_SLProps(_adaptor_cache[fid], uv.X(), uv.Y(), 1, 1e-7)
    if not props.IsNormalDefined():
        return None
    n = _vec(props.Normal())
    return -n if face.Orientation() == TopAbs_REVERSED else n


def _edge_in_face(face, edge):
    """The edge as oriented inside this face (orientation matters for convexity)."""
    ex = TopExp_Explorer(face, TopAbs_EDGE)
    while ex.More():
        e = ex.Current()
        if e.IsSame(edge):
            return TopoDS.Edge_s(e)
        ex.Next()
    return TopoDS.Edge_s(edge)


def build_adjacency(shape, faces):
    """Returns (edges list, neighbours dict fid -> {fid: [Edge,...]})."""
    fmap = TopTools_IndexedMapOfShape()
    for f in faces:
        fmap.Add(f)

    def fid_of(f):
        i = fmap.FindIndex(f)
        return i - 1 if i > 0 else None

    _adaptor_cache.clear()
    emap = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, emap)

    edges = []
    neighbours = defaultdict(lambda: defaultdict(list))
    for i in range(1, emap.Extent() + 1):
        edge = TopoDS.Edge_s(emap.FindKey(i))
        if BRep_Tool.Degenerated_s(edge):
            continue
        anc = [fid_of(TopoDS.Face_s(f)) for f in emap.FindFromIndex(i)]
        anc = [a for a in dict.fromkeys(anc) if a is not None]
        if len(anc) != 2:
            continue
        a, b = anc
        curve = BRepAdaptor_Curve(edge)
        t0, t1 = curve.FirstParameter(), curve.LastParameter()
        kinds, angles = [], []
        flip = (_edge_in_face(faces[a], edge).Orientation() == TopAbs_REVERSED) !=                (faces[a].Orientation() == TopAbs_REVERSED)
        for frac in (0.25, 0.5, 0.75):
            t = t0 + (t1 - t0) * frac
            pt, vt = gp_Pnt(), gp_Vec()
            curve.D1(t, pt, vt)
            if vt.Magnitude() < 1e-12:
                continue
            tan = _vec(vt) / vt.Magnitude()
            if flip:
                tan = -tan
            na, nb = _normal_at(a, faces[a], edge, t), _normal_at(b, faces[b], edge, t)
            if na is None or nb is None:
                continue
            ang = float(np.degrees(np.arccos(np.clip(na @ nb, -1, 1))))
            angles.append(ang)
            if ang < SMOOTH_ANGLE_DEG:
                kinds.append("smooth")
            else:
                kinds.append("convex" if float(np.cross(na, nb) @ tan) > 0 else "concave")
        if not kinds:
            continue
        kind = max(set(kinds), key=kinds.count)
        mid = _vec(curve.Value((t0 + t1) / 2))
        length = GCPnts_AbscissaPoint.Length_s(curve)
        e = Edge(faces=(a, b), kind=kind, angle=float(np.median(angles)), midpoint=mid, length=length)
        edges.append(e)
        neighbours[a][b].append(e)
        neighbours[b][a].append(e)
    return edges, neighbours
