# STEP 3D Feature Extraction

Reads a 3D CAD model in STEP format (`.stp` / `.step`) and identifies
its machining features. For every feature it reports the type, its
dimensions, its position and its direction. This is the 3D counterpart
of the 2D drawing-detection API. Instead of reading symbols off a
drawing, it reads the solid geometry directly, so sizes and positions
are exact, not OCR'd.

## What it finds

| Feature | Reported |
|---|---|
| **Holes** | Through or blind, diameter, depth, entry point, drilling direction. Also counterbore (Ø, depth), countersink (Ø, angle), entry and exit chamfer, drill-point bottom (118°) or flat bottom, and stepped-hole steps. A *thread hint* is given when the diameter matches a metric tap drill (e.g. Ø6.8 → M8) or clearance size. |
| **Slots** | Width, length, depth, centre, length direction, blind or through. |
| **Pockets** | Closed, open or circular (spot face) pockets and through channels: length × width, depth, opening direction, corner radius, floor fillet radius, floor area. |
| **Through openings** | Windows that go right through a wall: size, thickness, corner radius. |
| **Bosses** | Diameter, height, axis. |
| **Chamfers** | Straight edge chamfers, and circular or arc-edge chamfers: size × angle. |
| **Fillets / rounds** | Internal fillets and corner radii, and external rounds: radius. |
| **Surfaces** | Everything else, with an explanation: stock faces (top, bottom, sides), inclined or draft faces (with tilt angle), conical faces, slivers. Nothing is silently dropped. |

Identical features are grouped into **patterns**, e.g. `12× Ø6.8 ↧16, drill point [possibly tapped M8]`.

Each feature also lists the **CAD face colours** it carries. Many tool
and mould shops colour faces by finish or machining class. In
`ZT2126 CORE PLATE` the colours are red for machined flat faces, blue
for bores and walls, and cyan for tapped holes. So the colours can be
mapped to surface-finish requirements once the colour standard is known.

### Coordinates

* `entry_point` / `centre` / `position` are in the model's own coordinates.
* `position_from_datum` is measured from the **bounding-box minimum corner**
  (min X, min Y, min Z). That's usually how a machinist sets up the part.
* Units are mm.

## Outputs

For `part.stp` you get:

| File | Contents |
|---|---|
| `part_features.json` | Everything: part summary, every feature, patterns |
| `part_FEATURE_TABLE.csv` | One row per feature (open in Excel) |
| `part_FEATURE_TABLE.txt` | Printable table with patterns listed first |
| `part_viewer.html` | Interactive 3D viewer. Click a feature in the list to highlight it on the model, or click the model to find a face's feature. Link to a feature directly with `#f=<id>`. |

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python extract.py "ZT2126 CORE PLATE (1).stp" --out output
```

`ZT2126 CORE PLATE` (650 × 475 × 70 mm, 1,037 faces) takes about 25 s on a laptop CPU.

## Web UI

Start the server (below) and open **http://localhost:8000/** in a browser.

1. Enter the API key once. It's remembered in that browser.
2. Drop a `.stp` / `.step` file on the page, or click to choose one.
3. You get a summary, the interactive 3D viewer, a searchable feature
   table and the pattern list, plus download buttons for the CSV table,
   TXT table, full JSON and the standalone 3D viewer HTML.

The last 10 uploads are listed at the bottom of the page to reopen, as
long as the server still has those jobs.

## API

Same conventions as the 2D API: every endpoint except `/health` needs an
`X-API-Key` header, and the server won't start without `API_KEY` set.

```powershell
$env:API_KEY = "<your key>"
.\.venv\Scripts\uvicorn api:app --port 8000
```

| Method | Path | Returns |
|---|---|---|
| GET | `/` | Web upload page (no key needed to load it) |
| GET | `/health` | `{"status":"ok"}` |
| POST | `/extract` (multipart `file`) | `job_id` + part summary + features + patterns |
| GET | `/jobs/{job_id}` | The same result again |
| GET | `/jobs/{job_id}/features/{id}` | One feature |
| GET | `/jobs/{job_id}/feature_table.csv` | CSV table |
| GET | `/jobs/{job_id}/feature_table.txt` | Text table |
| GET | `/jobs/{job_id}/features.json` | Full result as a file |
| GET | `/jobs/{job_id}/viewer.html` | 3D viewer |

```bash
curl -H "X-API-Key: $API_KEY" -F "file=@part.stp" http://localhost:8000/extract
```

### Docker

```bash
docker build -t step-feature-extractor .
docker run -p 8000:8000 -e API_KEY=<value> step-feature-extractor
```

Jobs are stored in `/app/jobs` inside the container. Mount a volume
there to keep them across restarts. Nothing deletes old jobs yet.

## How it works

1. **Load** (`loader.py`): OpenCascade reads the STEP file, including per-face colours.
2. **Face analysis** (`faces.py`): each face's true type is recovered. Many
   exporters (ZW3D included) write *every* face as a B-spline, so this
   step first tries OpenCascade's canonical recognition, then fits a
   cylinder or cone to the face normals. Faces left over are classified
   as rolling-ball blends (fillets) or freeform.
3. **Topology** (`topology.py`): builds the face-adjacency graph and marks
   each shared edge as convex, concave or smooth.
4. **Recognition** (`recognize.py`): rule-based, in order: holes → slots →
   chamfers → pockets → through openings → bosses → fillets. Each step
   claims its faces so none are counted twice. Pocket floors must pass an
   accessibility test: a ray from the floor must reach open air, which
   separates floors from walls.
5. **Report** (`pipeline.py`, `report.py`, `viewer.py`): groups patterns,
   numbers the features and writes the outputs.

## Known limits

* **Surface finish (Ra), tolerances and GD&T** are not in these STEP files.
  They are AP214 files with no PMI, so they can't be read directly. The
  face colours are the practical source until AP242 files with PMI are available.
* **Threads** are never modelled in STEP geometry. `thread_hint` is an
  inference from the drill diameter, not a fact.
* Pocket and step logic assumes 2.5D machining along the principal X/Y/Z
  axes. Inclined planes are reported as inclined faces, not pockets.
* **Surface models** (loose faces, no solid, e.g. some Autodesk exports) are
  sewn together on load. The output then carries a warning, because hole
  ends and volumes are less certain than for a true solid.
* Tuned on `ZT2126 CORE PLATE`, then run on all 30 Zoom Technologies mould plates of the estimation dataset
  (cavity, core, bottom, ejector, manifold, stripper and top plates; up to 850 × 450 mm and 5,400 faces, which takes
  several minutes). Check new part families in the 3D viewer before trusting them.

## Files

```
extract.py            command line: python extract.py part.stp --out output
api.py                web API + upload page (FastAPI), needs API_KEY
web/index.html        the upload page
step_features/
  loader.py           read STEP (OpenCascade), face colours, sew surface models
  faces.py            true face type (plane / cylinder / cone / fillet / freeform), also for B-spline exports
  topology.py         face adjacency, convex / concave / smooth edges
  raycast.py          accessibility test for pocket floors
  recognize.py        holes, slots, chamfers, pockets, openings, bosses, fillets
  pipeline.py         extract_features(): runs the steps, groups patterns
  report.py           JSON / CSV / TXT outputs
  viewer.py           standalone interactive 3D viewer (HTML)
requirements.txt      Python 3.12; OpenCascade pinned to cadquery-ocp 7.8.1.1.post1
Dockerfile            CPU image for the API
```

Also used inside the **humac** estimator (github `humacpspl/Estimation_model`, `services/features3d`), where the
features drive the process plan, cycle time and cost.

No customer STEP files are in this repository; use your own parts.

