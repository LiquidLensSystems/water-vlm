FROM python:3.11-slim

# System deps: GDAL for geopandas, OpenGL libs for opencv-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full event_radar directory into /workspace
COPY event_radar/ .
