"""
FastAPI wrapper around the STEP feature-extraction pipeline.

Same conventions as the 2D drawing-detection API: every endpoint except
/health needs the X-API-Key header, the key comes from the API_KEY
environment variable, and the server refuses to start without one.

Each /extract call becomes a persistent job (job_id + folder under
JOBS_DIR) holding the uploaded STEP file and the JSON/CSV/TXT outputs,
so results can be fetched again later.

Known limitation: JOBS_DIR lives on the container's filesystem; without
a mounted volume, jobs are lost on restart, and nothing expires old jobs.
"""
import json
import os
import re
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import APIKeyHeader

from step_features import extract_features
from step_features.report import write_outputs

API_KEY = os.environ.get("API_KEY")
JOBS_DIR = Path(os.environ.get("JOBS_DIR", Path(__file__).resolve().parent / "jobs"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))
ALLOWED_SUFFIXES = {".stp", ".step"}

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Depends(_api_key_header)) -> None:
    # constant-time comparison so the check doesn't leak how much of the key matched
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status_code=401,
                            detail="Missing or invalid API key (expected in X-API-Key header).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEY:
        raise RuntimeError("API_KEY environment variable is not set; refusing to start unauthenticated.")
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="STEP 3D Feature Extraction API", version="1.0.0", lifespan=lifespan)


WEB_DIR = Path(__file__).resolve().parent / "web"


@app.get("/", include_in_schema=False)
def upload_page():
    """Browser upload page. The page itself is public; every call it makes sends the API key."""
    return FileResponse(WEB_DIR / "index.html", media_type="text/html")


@app.get("/health")
def health():
    return {"status": "ok"}


def job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    d = JOBS_DIR / job_id
    if not d.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    return d


@app.post("/extract", dependencies=[Depends(verify_api_key)])
async def extract(file: UploadFile = File(...)):
    """Upload a STEP file (.stp/.step); returns part summary, features and patterns."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Upload must be a .stp or .step file.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_MB} MB.")
    if not data.lstrip().startswith(b"ISO-10303-21"):
        raise HTTPException(status_code=400, detail="Not a STEP (ISO-10303-21) file.")

    job_id = uuid.uuid4().hex
    d = JOBS_DIR / job_id
    d.mkdir(parents=True)
    # keep the original name (it appears in the viewer and outputs), minus anything path-like
    safe = re.sub(r"[^A-Za-z0-9 ._()-]", "_", Path(file.filename).name).strip() or f"input{suffix}"
    step_path = d / safe
    step_path.write_bytes(data)

    try:
        result = await run_in_threadpool(extract_features, str(step_path), 0.2, d / "viewer.html")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")
    result["part"]["file"] = file.filename
    write_outputs(result, d, "result")
    (d / "meta.json").write_text(json.dumps({"filename": file.filename}), encoding="utf-8")
    return {"job_id": job_id, **result}


@app.get("/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
def get_job(job_id: str):
    d = job_dir(job_id)
    path = d / "result_features.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found for this job.")
    return {"job_id": job_id, **json.loads(path.read_text(encoding="utf-8"))}


@app.get("/jobs/{job_id}/features/{feature_id}", dependencies=[Depends(verify_api_key)])
def get_feature(job_id: str, feature_id: int):
    result = get_job(job_id)
    for f in result["features"]:
        if f["id"] == feature_id:
            return f
    raise HTTPException(status_code=404, detail=f"Unknown feature id: {feature_id}")


@app.get("/jobs/{job_id}/feature_table.csv", dependencies=[Depends(verify_api_key)])
def get_feature_table_csv(job_id: str):
    path = job_dir(job_id) / "result_FEATURE_TABLE.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Feature table not found for this job.")
    return FileResponse(path, media_type="text/csv", filename=f"{job_id}_FEATURE_TABLE.csv")


@app.get("/jobs/{job_id}/feature_table.txt", dependencies=[Depends(verify_api_key)])
def get_feature_table_txt(job_id: str):
    path = job_dir(job_id) / "result_FEATURE_TABLE.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Feature table not found for this job.")
    return FileResponse(path, media_type="text/plain; charset=utf-8", filename=f"{job_id}_FEATURE_TABLE.txt")


@app.get("/jobs/{job_id}/viewer.html", dependencies=[Depends(verify_api_key)])
def get_viewer(job_id: str):
    """Interactive 3D viewer: click a feature to highlight its faces on the part."""
    path = job_dir(job_id) / "viewer.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Viewer not found for this job.")
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/jobs/{job_id}/features.json", dependencies=[Depends(verify_api_key)])
def get_features_json(job_id: str):
    path = job_dir(job_id) / "result_features.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Result not found for this job.")
    return FileResponse(path, media_type="application/json", filename=f"{job_id}_features.json")
