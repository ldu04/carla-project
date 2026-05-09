#!/usr/bin/env bash
# Python 3.7 클라이언트 컨테이너에서 시나리오 실행 (--photo-cams top).
# 사용 전: CARLA 서버 기동 + 2000 포트 열림 확인.
set -euo pipefail

DOCKER="${DOCKER:-sudo docker}"
PROJECT="${CARLA_PROJECT:-${HOME}/carla-project}"
WHEEL="${CARLA_WHEEL:-carla-0.9.13-cp37-cp37m-manylinux_2_27_x86_64.whl}"
V2V="${V2V:-0}"
PHOTO_OUT="${PHOTO_OUT:-logs/photos/poster_run}"

cd "${PROJECT}"

if [[ ! -f "${WHEEL}" ]]; then
  echo "[error] wheel not found: ${PROJECT}/${WHEEL}"
  echo "        copy from CARLA image: docker cp <carla>:/home/carla/PythonAPI/carla/dist/${WHEEL} ."
  exit 1
fi

RUN="${DOCKER} run --rm -it --gpus all --net=host --shm-size=2g"
RUN="${RUN} -v ${PROJECT}:/work -w /work python:3.7-slim"

CMD="pip install -q -r requirements.txt && pip install -q ${WHEEL} && python scenario_emergency_brake.py --v2v ${V2V} --record-photos 1 --photo-cams top --photo-out-dir ${PHOTO_OUT}"

echo "[run] V2V=${V2V} PHOTO_OUT=${PHOTO_OUT}"
${RUN} bash -lc "${CMD}"
