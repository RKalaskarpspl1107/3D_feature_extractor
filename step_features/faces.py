"""Per-face geometric analysis.

Many CAD exporters (ZW3D here) write every face as a B-spline, so the
analytic type (plane/cylinder/cone/...) has to be recovered:
  1. OCCT canonical recognition at increasing tolerances;
  2. a curvature-sampling fallback for trimmed/partial faces the
     recogniser misses (e.g. quarter-cylinder pocket corners, fillets).
All normals are oriented OUT of the material.
"""

import math
from dataclasses import dataclass, field

import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GProp import GProp_GProps
from OCP.ShapeAnalysis import ShapeAnalysis_CanonicalRecognition
from OCP.TopAbs import TopAbs_REVERSED
from OCP.TopLoc import TopLoc_Location
from OCP.gp import gp_Cone, gp_Cylinder, gp_Dir, gp_Pln, gp_Sphere

RECOGNITION_TOLS = (1e-4, 1e-3, 5e-3, 2e-2)


@dataclass
class FaceInfo:
    id: int
    kind: str                         # plane | cylinder | cone | sphere | blend | freeform
    area: float
    centroid: np.ndarray
    points: np.ndarray                # sample points (N,3)
    normals: np.ndarray               # outward unit normals at points (N,3)
    color: tuple | None = None
    normal: np.ndarray | None = None  # plane: outward normal
    origin: np.ndarray | None = None  # plane: point on plane
    axis_loc: np.ndarray | None = None
    axis_dir: np.ndarray | None = None
    radius: float | None = None
    semi_angle: float | None = None   # cone, degrees
    apex: np.ndarray | None = None    # cone
    concave: bool | None = None       # cylinder/cone/blend: True = material outside (hole-like)
    recognised_by: str = ""
    extra: dict = field(default_factory=dict)

    def axial_range(self, loc, direction):
        t = (self.points - loc) @ direction
        return float(t.min()), float(t.max())


def _vec(p):
    return np.array([p.X(), p.Y(), p.Z()], dtype=float)


def mesh_shape(shape, linear_deflection=0.2, angular_deflection=0.3):
    BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)


def _sample_face(face, max_points=400):
    """Sample points + outward normals from the face triangulation's UV nodes."""
    loc = TopLoc_Location()
    tri = BRep_Tool.Triangulation_s(face, loc)
    adaptor = BRepAdaptor_Surface(face)
    reversed_ = face.Orientation() == TopAbs_REVERSED
    uvs = []
    if tri is not None and tri.HasUVNodes():
        n = tri.NbNodes()
        step = max(1, n // max_points)
        uvs = [tri.UVNode(i) for i in range(1, n + 1, step)]
        uvs = [(p.X(), p.Y()) for p in uvs]
    if not uvs:
        u0, u1 = adaptor.FirstUParameter(), adaptor.LastUParameter()
        v0, v1 = adaptor.FirstVParameter(), adaptor.LastVParameter()
        uvs = [(u0 + (u1 - u0) * a, v0 + (v1 - v0) * b)
               for a in (0.1, 0.5, 0.9) for b in (0.1, 0.5, 0.9)]
    props = BRepLProp_SLProps(adaptor, 2, 1e-7)
    pts, nrm, kmax, kmin, dmin, dmax = [], [], [], [], [], []
    for u, v in uvs:
        props.SetParameters(u, v)
        if not props.IsNormalDefined():
            continue
        n = _vec(props.Normal())
        if reversed_:
            n = -n
        pts.append(_vec(props.Value()))
        nrm.append(n)
        if props.IsCurvatureDefined():
            k1, k2 = props.MaxCurvature(), props.MinCurvature()
            if reversed_:
                k1, k2 = -k2, -k1
            kmax.append(k1)
            kmin.append(k2)
            gd1, gd2 = gp_Dir(), gp_Dir()
            props.CurvatureDirections(gd1, gd2)
            d1, d2 = _vec(gd1), _vec(gd2)
            if reversed_:
                d1, d2 = d2, d1
            dmax.append(d1)
            dmin.append(d2)
    return (np.array(pts), np.array(nrm), np.array(kmax), np.array(kmin),
            np.array(dmax), np.array(dmin))


def _is_concave(points, normals, axis_loc, axis_dir):
    radial = points - axis_loc
    radial -= np.outer(radial @ axis_dir, axis_dir)
    # Outward (from material) normal pointing towards the axis => hole-like
    return float(np.mean(np.sum(radial * normals, axis=1))) < 0


def analyse_face(fid, face, color=None) -> FaceInfo:
    g = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, g)
    area = g.Mass()
    centroid = _vec(g.CentreOfMass())
    pts, nrm, kmax, kmin, dmax, dmin = _sample_face(face)
    info = FaceInfo(id=fid, kind="freeform", area=area, centroid=centroid,
                    points=pts if len(pts) else centroid[None, :],
                    normals=nrm if len(nrm) else np.zeros((1, 3)), color=color)

    cr = ShapeAnalysis_CanonicalRecognition(face)

    def attempt(test, tol, target):
        try:
            return test(tol, target)
        except Exception:   # OCCT raises on degenerate fits
            return False

    for tol in RECOGNITION_TOLS:
        pl = gp_Pln()
        if attempt(cr.IsPlane, tol, pl):
            n = _vec(pl.Axis().Direction())
            if len(nrm) and float(np.mean(nrm @ n)) < 0:
                n = -n
            info.kind, info.normal, info.origin = "plane", n, _vec(pl.Location())
            info.recognised_by = f"canonical@{tol:g}"
            return info
        cy = gp_Cylinder()
        if attempt(cr.IsCylinder, tol, cy):
            ax = cy.Axis()
            info.kind, info.radius = "cylinder", cy.Radius()
            info.axis_loc, info.axis_dir = _vec(ax.Location()), _vec(ax.Direction())
            info.concave = _is_concave(info.points, info.normals, info.axis_loc, info.axis_dir)
            info.recognised_by = f"canonical@{tol:g}"
            return info
        co = gp_Cone()
        if attempt(cr.IsCone, tol, co):
            ang = abs(math.degrees(co.SemiAngle()))
            ax = co.Axis()
            info.axis_loc, info.axis_dir = _vec(ax.Location()), _vec(ax.Direction())
            info.recognised_by = f"canonical@{tol:g}"
            if ang < 0.5:          # numerically-tapered cylinder
                info.kind, info.radius = "cylinder", co.RefRadius()
            else:
                info.kind, info.semi_angle, info.apex = "cone", ang, _vec(co.Apex())
                info.radius = co.RefRadius()
            info.concave = _is_concave(info.points, info.normals, info.axis_loc, info.axis_dir)
            return info
        sp = gp_Sphere()
        if attempt(cr.IsSphere, tol, sp):
            info.kind, info.radius = "sphere", sp.Radius()
            info.axis_loc = _vec(sp.Location())
            info.recognised_by = f"canonical@{tol:g}"
            return info

    return _curvature_fallback(info, kmax, kmin, dmax, dmin)


def _fit_revolution(points, normals):
    """Fit a cylinder or cone to sample points using their normals.

    Every normal line of a surface of revolution meets its axis, and for
    cylinders/cones n.d is constant, so d is the least-variance direction
    of the (centred) normals. Returns dict or None.
    """
    if len(points) < 6:
        return None
    nc = normals - normals.mean(axis=0)
    _, sv, vt = np.linalg.svd(nc)
    d = vt[2]
    if sv[1] < 1e-3 * max(sv[0], 1e-12):     # normals nearly parallel: too little arc
        return None
    # 2D frame perpendicular to d
    e1 = np.cross(d, [1.0, 0, 0])
    if np.linalg.norm(e1) < 0.1:
        e1 = np.cross(d, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    P = np.stack([points @ e1, points @ e2], axis=1)
    N = np.stack([normals @ e1, normals @ e2], axis=1)
    ln = np.linalg.norm(N, axis=1)
    ok = ln > 1e-6
    if ok.sum() < 6:
        return None
    N = N[ok] / ln[ok, None]
    P2, t = P[ok], (points @ d)[ok]
    # least-squares intersection of 2D lines (p + s n): minimise perp distance
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for p_, n_ in zip(P2, N):
        M = np.eye(2) - np.outer(n_, n_)
        A += M
        b += M @ p_
    try:
        c2 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    rho = np.linalg.norm(P2 - c2, axis=1)
    X = np.stack([np.ones_like(t), t - t.mean()], axis=1)
    (a, slope), *_ = np.linalg.lstsq(X, rho, rcond=None)
    resid = np.abs(rho - X @ np.array([a, slope]))
    if resid.max() > max(0.015 * a, 0.03) or a <= 0:
        return None
    loc = c2[0] * e1 + c2[1] * e2 + t.mean() * d
    return {"dir": d, "loc": loc, "radius": float(a), "slope": float(slope),
            "residual": float(resid.max())}


def _curvature_fallback(info, kmax, kmin, dmax, dmin):
    if len(kmax) < 3:
        return info
    info.recognised_by = "fit"
    big = np.maximum(np.abs(kmax), np.abs(kmin))
    if big.max() < 1e-4:
        # planar B-spline not caught by recogniser: least-squares plane
        c = info.points.mean(axis=0)
        _, _, vt = np.linalg.svd(info.points - c)
        n = vt[2]
        if float(np.mean(info.normals @ n)) < 0:
            n = -n
        info.kind, info.normal, info.origin = "plane", n, c
        return info

    fit = _fit_revolution(info.points, info.normals)
    if fit is not None:
        d, loc = fit["dir"], fit["loc"]
        info.axis_dir, info.axis_loc = d, loc
        info.concave = _is_concave(info.points, info.normals, loc, d)
        ang = math.degrees(math.atan(abs(fit["slope"])))
        if ang < 0.5:
            info.kind, info.radius = "cylinder", fit["radius"]
        else:
            # orient d so the cone widens along +d, apex where radius hits 0
            if fit["slope"] < 0:
                d = -d
                info.axis_dir = d
            info.kind, info.semi_angle = "cone", ang
            info.radius = fit["radius"]
            info.apex = loc - d * (fit["radius"] / math.tan(math.radians(ang)))
        return info

    radii = 1.0 / np.maximum(big, 1e-12)
    r_med = float(np.median(radii))
    if float(np.std(radii)) / r_med < 0.1:
        # one principal radius ~constant, surface not a revolution => rolling-ball blend (fillet)
        info.kind, info.radius = "blend", r_med
        # dominant curvature > 0 <=> centre of curvature on the air side (concave)
        dom = np.where(np.abs(kmax) >= np.abs(kmin), kmax, kmin)
        info.concave = bool(np.median(dom) > 0)
        return info
    info.extra["min_radius"] = float(radii.min())
    info.extra["max_radius"] = float(radii.max())
    return info
