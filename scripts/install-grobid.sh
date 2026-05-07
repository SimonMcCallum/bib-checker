#!/usr/bin/env bash
set -euo pipefail
IMAGE="lfoppiano/grobid:0.8.1"
echo "Pulling ${IMAGE} (~500MB on first run)..."
docker pull "${IMAGE}"
echo "Done. Run scripts/start-grobid.sh to start the service."
