#!/usr/bin/env bash
# Python 3.7 클라이언트에서 scenario_poster_v2v.py 실행.
# 사전 조건: CARLA 서버 기동 + 2000/2001 대기.
set -euo pipefail

DOCKER="${DOCKER:-sudo docker}"
PROJECT="${CARLA_PROJECT:-${HOME}/carla-project}"
WHEEL="${CARLA_WHEEL:-carla-0.9.13-cp37-cp37m-manylinux_2_27_x86_64.whl}"

cd "${PROJECT}"

if [[ ! -f "${WHEEL}" ]]; then
  echo "[error] wheel not found: ${PROJECT}/${WHEEL}"
  echo "        docker cp <carla_server>:/home/carla/PythonAPI/carla/dist/${WHEEL} ."
  exit 1
fi

# 기본: 확인용(스크립트 기본 인자) — 최종은 POSTER_ARGS 로 해상도만 올리면 됨.
# 예: POSTER_ARGS="--output-root /work/output --map Town04 --weather ClearNoon --img-w 1920 --img-h 1080" bash ...
POSTER_ARGS=${POSTER_ARGS:---output-root /work/output --map Town04 --weather ClearNoon}

RUN="${DOCKER} run --rm -it --net=host --shm-size=2g"
RUN="${RUN} -v ${PROJECT}:/work -w /work python:3.7-slim"

APT='apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends'
APT_PKGS='libxcb1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libgl1'
CMD="${APT} ${APT_PKGS} && pip install -q -r requirements.txt && pip install -q ${WHEEL} && python scenario_poster_v2v.py ${POSTER_ARGS}"

echo "[run] scenario_poster_v2v.py ${POSTER_ARGS}"
${RUN} bash -lc "${CMD}"
