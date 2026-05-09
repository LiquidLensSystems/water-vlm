#!/usr/bin/env bash
# Liquid Lens — one-time asset download
# Run once before `docker compose up`.
# Requires: wget, unzip

set -euo pipefail

MODELS_DIR="models"
DATA_DIR="data/ne_10m_lakes"

# ── Model weights ─────────────────────────────────────────────────────────────
echo "Downloading LFM2.5-VL model weights..."
mkdir -p "$MODELS_DIR"

BASE_URL="https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main"

for FILE in "LFM2.5-VL-450M-Q4_0.gguf" "mmproj-LFM2.5-VL-450m-F16.gguf"; do
    if [ -f "$MODELS_DIR/$FILE" ]; then
        echo "  $FILE already exists, skipping."
    else
        echo "  Fetching $FILE..."
        wget -q --show-progress -O "$MODELS_DIR/$FILE" "$BASE_URL/$FILE"
    fi
done

echo "Model weights ready."

# ── Natural Earth lakes shapefile ─────────────────────────────────────────────
echo "Downloading Natural Earth 10m lakes shapefile..."
mkdir -p "$DATA_DIR"

SHP_ZIP="ne_10m_lakes.zip"
SHP_URL="https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes.zip"

if [ -f "$DATA_DIR/ne_10m_lakes.shp" ]; then
    echo "  Shapefile already exists, skipping."
else
    wget -q --show-progress -O "$SHP_ZIP" "$SHP_URL"
    unzip -q "$SHP_ZIP" -d "$DATA_DIR"
    rm "$SHP_ZIP"
    echo "  Shapefile extracted to $DATA_DIR/"
fi

echo ""
echo "Setup complete. Run:  docker compose up --build"
