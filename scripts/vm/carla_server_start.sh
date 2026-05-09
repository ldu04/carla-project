#!/usr/bin/env bash
# CARLA 0.9.13 서버 기동 (호스트 네트워크). 이미 떠 있으면 건너뜀.
set -euo pipefail

DOCKER="${DOCKER:-sudo docker}"
NAME="${CARLA_CONTAINER_NAME:-carla_server}"
IMAGE="${CARLA_IMAGE:-carlasim/carla:0.9.13}"

if "${DOCKER}" ps --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "[ok] ${NAME} already running"
  exit 0
fi

if "${DOCKER}" ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "[info] removing stopped container ${NAME}"
  "${DOCKER}" rm -f "${NAME}"
fi

echo "[start] ${NAME} (${IMAGE}) ..."
"${DOCKER}" run -d --name "${NAME}" \
  --gpus all \
  --net=host \
  --shm-size=2g \
  --entrypoint "" \
  "${IMAGE}" \
  bash -lc '/home/carla/CarlaUE4.sh -RenderOffScreen -nosound'

echo "[hint] 2000/2001 대기: bash scripts/vm/carla_ports_wait.sh"
echo "[hint] 로그: ${DOCKER} logs -f ${NAME}"
