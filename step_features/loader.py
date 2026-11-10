"""Load a STEP file into an OpenCascade shape, keeping per-face colours."""

from dataclasses import dataclass, field

from OCP.IFSelect import IFSelect_RetDone
from OCP.Quantity import Quantity_Color
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDocStd import TDocStd_Document
from OCP.BRep import BRep_Builder, BRep_Tool
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
from OCP.ShapeFix import ShapeFix_Solid
from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
from OCP.TopExp import TopExp
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shape
from OCP.TopTools import TopTools_IndexedMapOfShape
from OCP.XCAFDoc import XCAFDoc_ColorCurv, XCAFDoc_ColorGen, XCAFDoc_ColorSurf, XCAFDoc_DocumentTool


SEWING_TOL = 0.01   # mm; enough to join faces exported with separate edge copies


@dataclass
class LoadedModel:
    shape: TopoDS_Shape
    faces: list                      # TopoDS_Face, index == face id
    face_colors: dict = field(default_factory=dict)   # face id -> (r, g, b) 0-255
    is_solid: bool = True
    open_edges: int = 0
    warnings: list = field(default_factory=list)


def _to_srgb(linear: float) -> int:
    # OCCT stores colours in linear RGB; STEP files (and CAD UIs) use sRGB.
    v = linear * 12.92 if linear <= 0.0031308 else 1.055 * linear ** (1 / 2.4) - 0.055
    return round(max(0.0, min(1.0, v)) * 255)


def _color_of(color_tool, shape):
    c = Quantity_Color()
    for kind in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen, XCAFDoc_ColorCurv):
        if color_tool.GetColor(shape, kind, c):
            return (_to_srgb(c.Red()), _to_srgb(c.Green()), _to_srgb(c.Blue()))
    return None


def load_step(path: str) -> LoadedModel:
    doc = TDocStd_Document(TCollection_ExtendedString("step"))
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetNameMode(True)
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError(f"Could not read STEP file: {path}")
    if not reader.Transfer(doc):
        raise ValueError(f"Could not transfer STEP file: {path}")

    color_tool = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    # The XCAF document shares TShapes with the underlying reader's result,
    # so colours can be looked up on faces of this shape directly.
    shape = reader.Reader().OneShape()
    if shape.IsNull():
        raise ValueError("STEP file contains no shapes")

    face_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, face_map)
    body_color = _color_of(color_tool, shape)
    orig_colors = []
    for i in range(1, face_map.Extent() + 1):
        orig_colors.append(_color_of(color_tool, face_map.FindKey(i)) or body_color)

    model = LoadedModel(shape=shape, faces=[])
    solids = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_SOLID, solids)
    sewing = None
    if solids.Extent() == 0:
        # Surface model (loose faces/shells): stitch shared edges so faces know their
        # neighbours. Colours are carried over through the sewing history.
        sewing = BRepBuilderAPI_Sewing(SEWING_TOL)
        sewing.Add(shape)
        sewing.Perform()
        sewn = sewing.SewedShape()
        model.is_solid = False
        model.open_edges = sewing.NbFreeEdges()
        shell_map = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(sewn, TopAbs_SHELL, shell_map)
        model.warnings.append(
            f"Surface model (no solid): faces were sewn at {SEWING_TOL} mm into {shell_map.Extent()} "
            f"shell(s) with {model.open_edges} open edge(s). Inside/outside tests use ray voting, "
            f"so hole ends and volumes are less certain than for a solid.")
        # Sewing picks each shell's orientation arbitrarily; an inside-out shell makes holes
        # look like pins. Negative enclosed volume means the normals point inward: flip it.
        builder = BRep_Builder()
        oriented = TopoDS_Compound()
        builder.MakeCompound(oriented)
        flipped = 0
        for k in range(1, shell_map.Extent() + 1):
            sh = shell_map.FindKey(k)
            props = GProp_GProps()
            BRepGProp.VolumeProperties_s(sh, props)
            if props.Mass() < 0:
                sh = sh.Reversed()
                flipped += 1
            builder.Add(oriented, sh)
        if flipped:
            sewn = oriented
            shell_map = TopTools_IndexedMapOfShape()
            TopExp.MapShapes_s(sewn, TopAbs_SHELL, shell_map)
        if shell_map.Extent() == 1 and BRep_Tool.IsClosed_s(shell_map.FindKey(1)):
            fix = ShapeFix_Solid(BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(shell_map.FindKey(1))).Solid())
            fix.Perform()
            sewn = fix.Solid()
            model.is_solid = True
            model.warnings[-1] = f"Surface model sewn into a closed solid at {SEWING_TOL} mm."
        model.shape = sewn

    new_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(model.shape, TopAbs_FACE, new_map)
    color_by_index = {}
    for i in range(1, face_map.Extent() + 1):
        f = face_map.FindKey(i)
        if sewing is not None:
            f = sewing.ModifiedSubShape(f)   # the sewn face that replaced this original face
        k = new_map.FindIndex(f)
        if k > 0 and orig_colors[i - 1] is not None:
            color_by_index[k] = orig_colors[i - 1]
    for k in range(1, new_map.Extent() + 1):
        fid = len(model.faces)
        model.faces.append(TopoDS.Face_s(new_map.FindKey(k)))
        if k in color_by_index:
            model.face_colors[fid] = color_by_index[k]
    return model
