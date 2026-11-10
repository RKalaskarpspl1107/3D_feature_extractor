# CPU-only image for the STEP feature-extraction API.
FROM python:3.12-slim

WORKDIR /app

# Stream logs straight to `docker logs` instead of block-buffering them.
ENV PYTHONUNBUFFERED=1

# OpenCascade (cadquery-ocp) and its vtk dependency load these shared
# libraries at import time even though nothing is rendered server-side.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt

COPY api.py extract.py ./
COPY step_features/ ./step_features/
COPY web/ ./web/

# Fail the build (not the first request) if the geometry kernel can't load.
RUN python -c "from step_features import extract_features; print('OCP ok')"

EXPOSE 8000

# api.py refuses to start without API_KEY; pass it at run time:
#   docker run -p 8000:8000 -e API_KEY=<value> step-feature-extractor
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
