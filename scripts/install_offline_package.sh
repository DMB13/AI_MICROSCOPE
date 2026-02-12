#!/usr/bin/env bash
# Install and run the offline package on a target machine (requires Docker installed)
set -euo pipefail

TAR_PATH=${1:-dist/ai_microscope_offline.tar}
TAG=ai_microscope:offline

if [ ! -f "$TAR_PATH" ]; then
  echo "Tar file not found: $TAR_PATH"
  echo "Usage: $0 /path/to/ai_microscope_offline.tar"
  exit 1
fi

echo "Loading Docker image from $TAR_PATH..."
docker load -i "$TAR_PATH"

echo "Image loaded. To run the GUI container (requires X11), run:" \
     "\n  docker run --rm -it \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v \$PWD/model/records:/home/appuser/AI_MICROSCOPE/model/records \
    ai_microscope:offline"

echo "Or run headless scripts inside the container as documented in README.md"
