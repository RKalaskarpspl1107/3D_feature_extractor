"""CLI: python extract.py <file.stp> [--out output_dir]"""

import argparse
import sys
from pathlib import Path

from step_features import extract_features
from step_features.report import write_outputs


def main():
    ap = argparse.ArgumentParser(description="Extract machining features from a STEP file")
    ap.add_argument("step_file")
    ap.add_argument("--out", default="output", help="output directory (default: output)")
    ap.add_argument("--deflection", type=float, default=0.2,
                    help="mesh sampling deflection in mm (smaller = slower, more precise)")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    stem = Path(args.step_file).stem.replace(" ", "_")
    viewer = Path(args.out) / f"{stem}_viewer.html"
    viewer.parent.mkdir(parents=True, exist_ok=True)
    result = extract_features(args.step_file, mesh_deflection=args.deflection, viewer_path=viewer)
    paths = write_outputs(result, args.out, stem)
    paths["viewer"] = viewer
    part = result["part"]
    print(f"{part['file']}: {part['face_count']} faces -> {len(result['features'])} features "
          f"in {part['processing_seconds']}s")
    for k, v in part["feature_counts"].items():
        print(f"  {k:<8} {v}")
    for p in paths.values():
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
