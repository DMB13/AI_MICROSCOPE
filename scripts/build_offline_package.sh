#!/usr/bin/env bash
# Build Docker image and save as tar for offline transfer.
set -euo pipefail

TAG=ai_microscope:offline
OUT_DIR=dist
OUT_TAR=${OUT_DIR}/ai_microscope_offline.tar

mkdir -p "$OUT_DIR"

echo "Building Docker image ${TAG}..."
docker build -t ${TAG} .

echo "Saving image to ${OUT_TAR} (this may take a while)..."
docker save -o "${OUT_TAR}" ${TAG}

echo "Offline package created: ${OUT_TAR}"
echo "Transfer this tar to the offline machine and run the install script to load and run the container."
