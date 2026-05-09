#!/usr/bin/env bash
# CARLA RPC 포트(2000)가 열릴 때까지 대기 (최대 약 4분)
set -euo pipefail

MAX_ROUNDS="${CARLA_WAIT_ROUNDS:-120}"
SLEEP_SEC="${CARLA_WAIT_SLEEP:-2}"

for ((i = 1; i <= MAX_ROUNDS; i++)); do
  if ss -lntp 2>/dev/null | grep -q ':2000'; then
    echo "[ok] listening:"
    ss -lntp 2>/dev/null | grep -E ':2000|:2001' || true
    exit 0
  fi
  if ((i % 15 == 1)); then
    echo "[wait] round ${i}/${MAX_ROUNDS} (CARLA boot can take several minutes) ..."
  fi
  sleep "${SLEEP_SEC}"
done

echo "[error] timeout: port 2000 not listening"
exit 1
