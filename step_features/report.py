"""Write extraction results as JSON + CSV/TXT feature tables."""

import csv
import json
from pathlib import Path

CSV_COLUMNS = ["id", "pattern_id", "type", "subtype", "callout",
               "diameter", "depth", "length", "width", "radius", "size", "angle",
               "x", "y", "z", "x_from_datum", "y_from_datum", "z_from_datum",
               "direction", "thread_hint", "colors", "face_ids"]


def _position(f):
    for k in ("entry_point", "centre", "centre_at_top", "base_centre", "position"):
        if k in f:
            return f[k]
    return [None, None, None]


def feature_row(f):
    x, y, z = _position(f)
    dx, dy, dz = f.get("position_from_datum") or [None, None, None]
    direction = (f.get("direction_name") or f.get("open_direction") or f.get("axis")
                 or f.get("depth_axis") or f.get("normal") or "")
    return {
        "id": f["id"], "pattern_id": f.get("pattern_id", ""), "type": f["type"],
        "subtype": f.get("subtype", ""), "callout": f["callout"],
        "diameter": f.get("diameter", ""), "depth": f.get("depth", ""),
        "length": f.get("length", ""), "width": f.get("width", ""),
        "radius": f.get("radius", ""), "size": f.get("size", ""), "angle": f.get("angle", ""),
        "x": x, "y": y, "z": z, "x_from_datum": dx, "y_from_datum": dy, "z_from_datum": dz,
        "direction": direction, "thread_hint": f.get("thread_hint", ""),
        "colors": " ".join("rgb(%d,%d,%d)x%d" % (*c["rgb"], c["faces"]) for c in f.get("colors", [])),
        "face_ids": " ".join(map(str, f["face_ids"])),
    }


def write_outputs(result, out_dir, stem=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = stem or Path(result["part"]["file"]).stem
    paths = {
        "json": out / f"{stem}_features.json",
        "csv": out / f"{stem}_FEATURE_TABLE.csv",
        "txt": out / f"{stem}_FEATURE_TABLE.txt",
    }
    paths["json"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = [feature_row(f) for f in result["features"]]
    with open(paths["csv"], "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    part = result["part"]
    s = part["stock_size"]
    lines = [
        f"FEATURE TABLE - {part['file']}",
        f"Stock (bounding box): {s['x']:g} x {s['y']:g} x {s['z']:g} {part['units']}",
        f"Datum: bounding-box min corner {part['bounding_box_min']}",
        f"Volume: {part['volume_mm3']:g} mm3   Material removed from stock: {part['material_removed_pct']:g}%",
        f"Faces: {part['face_count']}   Feature counts: {part['feature_counts']}",
    ]
    for w in part.get("warnings", []):
        lines.append(f"WARNING: {w}")
    if part.get("tiny_faces_omitted", {}).get("count"):
        lines.append(f"Note: {part['tiny_faces_omitted']['count']} tiny faces (<1 mm²) not listed as features")
    lines += [
        "",
        "PATTERNS (identical features)",
    ]
    for p in result["patterns"]:
        lines.append(f"  {p['pattern_id']:>4}  {p['callout']}")
    lines += ["", f"{'ID':>4} {'PAT':>4}  {'TYPE':<8} {'X':>9} {'Y':>9} {'Z':>9}  {'DIR':<6} CALLOUT",
              "     (X/Y/Z measured from datum)"]
    for r in rows:
        def fmt(v):
            return f"{v:9.3f}" if isinstance(v, (int, float)) else f"{'':>9}"
        dirn = r["direction"] if len(str(r["direction"])) <= 6 else "incl."
        lines.append(f"{r['id']:>4} {r['pattern_id']:>4}  {r['type']:<8} {fmt(r['x_from_datum'])} "
                     f"{fmt(r['y_from_datum'])} {fmt(r['z_from_datum'])}  {dirn:<6} {r['callout']}")
    paths["txt"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths
