#!/usr/bin/env bash
# client_poster_top.sh 와 동일하나 TTY 없이 실행 (nohup/파이프라인용).
set -euo pipefail

DOCKER="${DOCKER:-sudo docker}"
PROJECT="${CARLA_PROJECT:-${HOME}/carla-project}"
WHEEL="${CARLA_WHEEL:-carla-0.9.13-cp37-cp37m-manylinux_2_27_x86_64.whl}"
V2V="${V2V:-0}"
PHOTO_OUT="${PHOTO_OUT:-logs/photos/poster_run}"

cd "${PROJECT}"

if [[ ! -f "${WHEEL}" ]]; then
  echo "[error] wheel not found: ${PROJECT}/${WHEEL}"
  exit 1
fi

CMD="pip install -q -r requirements.txt && pip install -q ${WHEEL} && python scenario_emergency_brake.py --v2v ${V2V} --record-photos 1 --photo-cams top --photo-out-dir ${PHOTO_OUT}"

echo "[run] V2V=${V2V} PHOTO_OUT=${PHOTO_OUT} (no TTY)"
"${DOCKER}" run --rm -i \
  --gpus all \
  --net=host \
  --shm-size=2g \
  -v "${PROJECT}:/work" \
  -w /work \
  python:3.7-slim \
  bash -lc "${CMD}"
