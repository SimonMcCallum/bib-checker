#!/usr/bin/env bash
set -euo pipefail
IMAGE="lfoppiano/grobid:0.8.1"
NAME="grobid"

if docker ps -a --filter "name=^/${NAME}$" --format '{{.Names}}' | grep -q "^${NAME}$"; then
    echo "Container '${NAME}' exists; removing..."
    docker rm -f "${NAME}" >/dev/null
fi

echo "Starting ${IMAGE} on http://localhost:8070 ..."
docker run --rm -d -p 8070:8070 --name "${NAME}" "${IMAGE}" >/dev/null

deadline=$(( $(date +%s) + 120 ))
while [ "$(date +%s)" -lt "${deadline}" ]; do
    if curl -sf http://localhost:8070/api/isalive >/dev/null 2>&1; then
        echo "GROBID is alive."
        exit 0
    fi
    sleep 2
done
echo "GROBID did not become healthy within 120s" >&2
exit 1
